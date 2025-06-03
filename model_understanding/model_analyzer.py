#!/usr/bin/env python3
"""
Detectron2 Model Architecture Analyzer
Provides detailed analysis of model structure, layer inputs/outputs, and data flow.
"""

import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any
import json

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T

class LayerHook:
    """Hook to capture inputs and outputs of layers"""
    def __init__(self, name: str):
        self.name = name
        self.input_shapes = []
        self.output_shapes = []
        self.input_data = None
        self.output_data = None
        
    def __call__(self, module, input_data, output_data):
        # Handle different input types
        if isinstance(input_data, (tuple, list)):
            self.input_shapes = [self._get_shape(x) for x in input_data]
            self.input_data = input_data
        else:
            self.input_shapes = [self._get_shape(input_data)]
            self.input_data = [input_data]
            
        # Handle different output types
        if isinstance(output_data, (tuple, list)):
            self.output_shapes = [self._get_shape(x) for x in output_data]
            self.output_data = output_data
        else:
            self.output_shapes = [self._get_shape(output_data)]
            self.output_data = [output_data]
    
    def _get_shape(self, data):
        """Get shape of various data types"""
        if isinstance(data, torch.Tensor):
            return list(data.shape)
        elif isinstance(data, dict):
            return {k: self._get_shape(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._get_shape(x) for x in data]
        else:
            return str(type(data))

class ModelAnalyzer:
    """Comprehensive model architecture analyzer"""
    
    def __init__(self, config_file: str, weights_path: str = None):
        self.cfg = self._setup_config(config_file, weights_path)
        self.model = build_model(self.cfg)
        
        if weights_path:
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(weights_path)
        
        self.model.eval()
        self.hooks = {}
        self.layer_info = OrderedDict()
        self._analyze_architecture()
    
    def _setup_config(self, config_file: str, weights_path: str = None):
        """Setup detectron2 configuration"""
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        if weights_path:
            cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = "cpu"  # Use CPU for analysis
        return cfg
    
    def _analyze_architecture(self):
        """Analyze the model architecture and collect layer information"""
        print("Analyzing model architecture...")
        
        # Get all named modules
        for name, module in self.model.named_modules():
            if name:  # Skip the root module
                self.layer_info[name] = {
                    'type': type(module).__name__,
                    'module': module,
                    'parameters': self._count_parameters(module),
                    'input_shape': None,
                    'output_shape': None,
                    'parent': self._get_parent_name(name)
                }
    
    def _count_parameters(self, module):
        """Count parameters in a module"""
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {'total': total_params, 'trainable': trainable_params}
    
    def _get_parent_name(self, name):
        """Get parent module name"""
        parts = name.split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else 'root'
    
    def register_hooks(self, layer_names: List[str] = None):
        """Register forward hooks for specified layers"""
        if layer_names is None:
            layer_names = list(self.layer_info.keys())
        
        for name in layer_names:
            if name in self.layer_info:
                hook = LayerHook(name)
                handle = self.layer_info[name]['module'].register_forward_hook(hook)
                self.hooks[name] = {'hook': hook, 'handle': handle}
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook_info in self.hooks.values():
            hook_info['handle'].remove()
        self.hooks.clear()
    
    def analyze_forward_pass(self, image_path: str, target_layers: List[str] = None):
        """Analyze a forward pass through the model"""
        print(f"Analyzing forward pass with image: {image_path}")
        
        # Prepare input
        image = read_image(image_path, format="BGR")
        height, width = image.shape[:2]
        
        # Apply transforms
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        input_tensor = torch.as_tensor(transformed_img.astype("float32").transpose(2, 0, 1))
        
        inputs = [{"image": input_tensor, "height": height, "width": width}]
        
        # Register hooks for target layers
        if target_layers is None:
            # Default to key architectural components
            target_layers = [
                'backbone.bottom_up.stem',
                'backbone.bottom_up.res2',
                'backbone.bottom_up.res3', 
                'backbone.bottom_up.res4',
                'backbone.bottom_up.res5',
                'backbone.fpn_lateral2',
                'backbone.fpn_lateral3',
                'backbone.fpn_lateral4',
                'backbone.fpn_lateral5',
                'proposal_generator.rpn_head',
                'roi_heads.box_head',
                'roi_heads.mask_head'
            ]
        
        self.register_hooks(target_layers)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Collect results
        results = {
            'input_shape': list(input_tensor.shape),
            'input_image_size': (height, width),
            'transformed_size': transformed_img.shape[:2],
            'layer_outputs': {},
            'model_output': self._analyze_output(outputs[0])
        }
        
        for name, hook_info in self.hooks.items():
            hook = hook_info['hook']
            results['layer_outputs'][name] = {
                'input_shapes': hook.input_shapes,
                'output_shapes': hook.output_shapes,
                'layer_type': self.layer_info[name]['type']
            }
        
        self.remove_hooks()
        return results
    
    def _analyze_output(self, output):
        """Analyze model output structure"""
        result = {}
        if 'instances' in output:
            instances = output['instances']
            result['instances'] = {
                'num_instances': len(instances),
                'fields': list(instances.get_fields().keys()),
                'image_size': instances.image_size
            }
            
            if hasattr(instances, 'pred_boxes'):
                result['instances']['pred_boxes_shape'] = list(instances.pred_boxes.tensor.shape)
            if hasattr(instances, 'pred_classes'):
                result['instances']['pred_classes_shape'] = list(instances.pred_classes.shape)
            if hasattr(instances, 'scores'):
                result['instances']['scores_shape'] = list(instances.scores.shape)
            if hasattr(instances, 'pred_masks'):
                result['instances']['pred_masks_shape'] = list(instances.pred_masks.shape)
        
        return result
    
    def print_architecture_summary(self):
        """Print a summary of the model architecture"""
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*80)
        
        # Group layers by main components
        components = defaultdict(list)
        for name, info in self.layer_info.items():
            if '.' in name:
                main_component = name.split('.')[0]
            else:
                main_component = 'root'
            components[main_component].append((name, info))
        
        total_params = 0
        for component, layers in components.items():
            print(f"\n{component.upper()}:")
            print("-" * 40)
            
            component_params = 0
            for name, info in layers:
                layer_type = info['type']
                params = info['parameters']['total']
                component_params += params
                
                if params > 0:
                    print(f"  {name:<40} {layer_type:<20} {params:>10,} params")
                else:
                    print(f"  {name:<40} {layer_type:<20} {'':>10}")
            
            print(f"  {'Component Total:':<62} {component_params:>10,} params")
            total_params += component_params
        
        print(f"\n{'TOTAL MODEL PARAMETERS:':<62} {total_params:>10,}")
        print("="*80)
    
    def print_data_flow(self, analysis_results):
        """Print the data flow through the model"""
        print("\n" + "="*80)
        print("DATA FLOW ANALYSIS")
        print("="*80)
        
        print(f"Input Image Size: {analysis_results['input_image_size']}")
        print(f"Transformed Size: {analysis_results['transformed_size']}")
        print(f"Input Tensor Shape: {analysis_results['input_shape']}")
        
        print("\nLayer-by-Layer Data Flow:")
        print("-" * 80)
        
        for layer_name, layer_data in analysis_results['layer_outputs'].items():
            print(f"\n{layer_name} ({layer_data['layer_type']}):")
            print(f"  Input shapes:  {layer_data['input_shapes']}")
            print(f"  Output shapes: {layer_data['output_shapes']}")
        
        print(f"\nModel Output Structure:")
        print("-" * 40)
        output_info = analysis_results['model_output']
        if 'instances' in output_info:
            inst = output_info['instances']
            print(f"  Number of detected instances: {inst['num_instances']}")
            print(f"  Instance fields: {inst['fields']}")
            print(f"  Image size: {inst['image_size']}")
            
            for field, shape in inst.items():
                if field.endswith('_shape'):
                    field_name = field.replace('_shape', '')
                    print(f"  {field_name} shape: {shape}")
    
    def get_layer_details(self, layer_name: str):
        """Get detailed information about a specific layer"""
        if layer_name not in self.layer_info:
            available_layers = list(self.layer_info.keys())
            print(f"Layer '{layer_name}' not found. Available layers:")
            for layer in available_layers[:10]:  # Show first 10
                print(f"  - {layer}")
            if len(available_layers) > 10:
                print(f"  ... and {len(available_layers) - 10} more")
            return None
        
        info = self.layer_info[layer_name]
        module = info['module']
        
        details = {
            'name': layer_name,
            'type': info['type'],
            'parameters': info['parameters'],
            'parent': info['parent']
        }
        
        # Add module-specific details
        if hasattr(module, 'in_features'):
            details['in_features'] = module.in_features
        if hasattr(module, 'out_features'):
            details['out_features'] = module.out_features
        if hasattr(module, 'kernel_size'):
            details['kernel_size'] = module.kernel_size
        if hasattr(module, 'stride'):
            details['stride'] = module.stride
        if hasattr(module, 'padding'):
            details['padding'] = module.padding
        
        return details
    
    def save_analysis(self, analysis_results, output_file: str):
        """Save analysis results to a JSON file"""
        # Convert tensor shapes to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Size):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            else:
                return obj
        
        json_data = convert_for_json(analysis_results)
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Analysis saved to {output_file}")

def main():
    """Example usage of the ModelAnalyzer"""
    
    # Configuration
    config_file = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
    weights_path = None  # Set to actual weights path if available
    image_path = "path/to/test/image.jpg"  # Set to actual image path
    
    # Create analyzer
    analyzer = ModelAnalyzer(config_file, weights_path)
    
    # Print architecture summary
    analyzer.print_architecture_summary()
    
    # Analyze forward pass (if image available)
    if image_path and os.path.exists(image_path):
        results = analyzer.analyze_forward_pass(image_path)
        analyzer.print_data_flow(results)
        analyzer.save_analysis(results, "model_analysis.json")
    
    # Get details of specific layers
    interesting_layers = [
        'backbone.bottom_up.stem.conv1',
        'backbone.bottom_up.res2.0.conv1',
        'backbone.fpn_lateral2',
        'proposal_generator.rpn_head.conv',
        'roi_heads.box_head.fc1',
        'roi_heads.mask_head.mask_fcn1'
    ]
    
    print("\n" + "="*80)
    print("DETAILED LAYER INFORMATION")
    print("="*80)
    
    for layer_name in interesting_layers:
        details = analyzer.get_layer_details(layer_name)
        if details:
            print(f"\n{layer_name}:")
            for key, value in details.items():
                if key != 'name':
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    import os
    main()