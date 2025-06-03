#!/usr/bin/env python
"""
Enhanced Detection-Aware Saliency Map Analysis with Multiple Attribution Methods.

This implementation supports:
1. Vanilla Gradients (your current method)
2. Input × Gradients (Shrikumar et al. 2017)
3. Integrated Gradients (Sundararajan et al. 2017)
4. SmoothGrad (Smilkov et al. 2017) - bonus method for noise reduction
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T


class EnhancedDetectionSaliencyMap:
    """
    Enhanced saliency map generator supporting multiple attribution methods.
    
    Supports:
    - vanilla_gradients: Standard gradients (your current method)
    - input_gradients: Input × Gradients (Shrikumar et al. 2017)
    - integrated_gradients: Integrated Gradients (Sundararajan et al. 2017)
    - smoothgrad: SmoothGrad with any base method
    """
    
    def __init__(self, model, cfg):
        """
        Initialize the enhanced detection saliency map generator.
        
        Args:
            model: Detectron2 model (e.g., Mask R-CNN)
            cfg: Detectron2 configuration
        """
        self.model = model
        self.cfg = cfg
        self.model.eval()
        
    def compute_instance_saliency(self, img_path, target_instance_idx, method='integrated_gradients', **kwargs):
        """
        Compute saliency map for a specific detected instance using various methods.
        
        Args:
            img_path: Path to the input image
            target_instance_idx: Index of the detection instance to analyze
            method: Attribution method to use
                   - 'vanilla_gradients': Standard gradients
                   - 'input_gradients': Input × Gradients
                   - 'integrated_gradients': Integrated Gradients
                   - 'smoothgrad_vanilla': SmoothGrad with vanilla gradients
                   - 'smoothgrad_input': SmoothGrad with input × gradients
            **kwargs: Additional arguments for specific methods
                     For integrated_gradients: steps (default=50), baseline (default='black')
                     For smoothgrad: noise_samples (default=50), noise_level (default=0.15)
            
        Returns:
            saliency_map: Pixel importance map (same size as original image)
            saliency_map_raw: Raw saliency map before resizing
            outputs: Model outputs including all detections
        """
        # Read and preprocess image
        image = read_image(img_path, format="BGR")
        height, width = image.shape[:2]
        
        # Apply standard Detectron2 transforms
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        
        # Dispatch to appropriate method
        if method == 'vanilla_gradients':
            saliency_map_raw, outputs = self._vanilla_gradients(
                transformed_img, height, width, target_instance_idx
            )
        elif method == 'input_gradients':
            saliency_map_raw, outputs = self._input_gradients(
                transformed_img, height, width, target_instance_idx
            )
        elif method == 'integrated_gradients':
            steps = kwargs.get('steps', 50)
            baseline = kwargs.get('baseline', 'black')
            saliency_map_raw, outputs = self._integrated_gradients(
                transformed_img, height, width, target_instance_idx, steps, baseline
            )
        elif method.startswith('smoothgrad'):
            base_method = method.split('_')[1] if '_' in method else 'vanilla'
            noise_samples = kwargs.get('noise_samples', 50)
            noise_level = kwargs.get('noise_level', 0.15)
            saliency_map_raw, outputs = self._smoothgrad(
                transformed_img, height, width, target_instance_idx, 
                base_method, noise_samples, noise_level
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if saliency_map_raw is None:
            return None, None, outputs
            
        # Resize saliency map to match original image dimensions
        saliency_map = cv2.resize(saliency_map_raw, (width, height))
        
        return saliency_map, saliency_map_raw, outputs

    def _vanilla_gradients(self, transformed_img, height, width, target_instance_idx):
        """Standard gradient method (your current implementation)."""
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        input_tensor.requires_grad = True
        
        outputs = self.model([{
            "image": input_tensor,
            "height": height,
            "width": width
        }])
        
        instances = outputs[0]["instances"]
        if target_instance_idx >= len(instances):
            return None, outputs[0]
            
        detection_score = instances.scores[target_instance_idx]
        
        self.model.zero_grad()
        detection_score.backward()
        
        pixel_gradients = input_tensor.grad.data.abs()
        saliency_map_raw = pixel_gradients.mean(dim=0).cpu().numpy()
        
        # Normalize
        saliency_map_raw = (saliency_map_raw - saliency_map_raw.min()) / (
            saliency_map_raw.max() - saliency_map_raw.min() + 1e-8
        )
        
        return saliency_map_raw, outputs[0]

    def _input_gradients(self, transformed_img, height, width, target_instance_idx):
        """Input × Gradients method (Shrikumar et al. 2017)."""
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        input_tensor.requires_grad = True
        
        outputs = self.model([{
            "image": input_tensor,
            "height": height,
            "width": width
        }])
        
        instances = outputs[0]["instances"]
        if target_instance_idx >= len(instances):
            return None, outputs[0]
            
        detection_score = instances.scores[target_instance_idx]
        
        self.model.zero_grad()
        detection_score.backward()
        
        # Input × Gradients: element-wise multiplication
        input_gradients_attribution = (input_tensor * input_tensor.grad).data.abs()
        saliency_map_raw = input_gradients_attribution.mean(dim=0).cpu().numpy()
        
        # Normalize
        saliency_map_raw = (saliency_map_raw - saliency_map_raw.min()) / (
            saliency_map_raw.max() - saliency_map_raw.min() + 1e-8
        )
        
        return saliency_map_raw, outputs[0]

    def _integrated_gradients(self, transformed_img, height, width, target_instance_idx, 
                            steps=20, baseline='blur'):
        """Integrated Gradients method (Sundararajan et al. 2017) - Detection-aware version."""
        print(f"    Starting Integrated Gradients with {steps} steps, baseline: {baseline}")
        
        try:
            # Create baseline - using blur by default for object detection
            if baseline == 'black':
                baseline_img = np.zeros_like(transformed_img)
            elif baseline == 'white':
                baseline_img = np.ones_like(transformed_img) * 255
            elif baseline == 'blur':
                # Heavy Gaussian blur as baseline - still recognizable but removes fine details
                baseline_img = cv2.GaussianBlur(transformed_img, (101, 101), 50)
            elif baseline == 'noise':
                # Structured noise baseline
                baseline_img = np.random.normal(128, 50, transformed_img.shape).clip(0, 255).astype(np.float32)
            elif isinstance(baseline, (int, float)):
                baseline_img = np.full_like(transformed_img, baseline)
            else:
                raise ValueError(f"Unknown baseline type: {baseline}")
            
            print(f"    Created baseline image with shape: {baseline_img.shape}")
            
            # First, get the reference detection from the original image to know what we're looking for
            input_tensor_ref = torch.as_tensor(
                transformed_img.astype("float32").transpose(2, 0, 1)
            ).to(self.cfg.MODEL.DEVICE)
            
            with torch.no_grad():  # Don't track gradients for reference
                ref_outputs = self.model([{
                    "image": input_tensor_ref,
                    "height": height,
                    "width": width
                }])
            
            ref_instances = ref_outputs[0]["instances"]
            if target_instance_idx >= len(ref_instances):
                print(f"    Error: Target instance {target_instance_idx} not found in reference (only {len(ref_instances)} instances)")
                return None, ref_outputs[0]
                
            # Get the reference bounding box and class to track across interpolations
            ref_bbox = ref_instances.pred_boxes.tensor[target_instance_idx].detach().cpu().numpy()
            ref_class = ref_instances.pred_classes[target_instance_idx].detach().cpu().item()
            print(f"    Reference detection: class {ref_class}, bbox {ref_bbox}")
            
            # Convert to tensors
            input_tensor = torch.as_tensor(
                transformed_img.astype("float32").transpose(2, 0, 1)
            ).to(self.cfg.MODEL.DEVICE)
            
            baseline_tensor = torch.as_tensor(
                baseline_img.astype("float32").transpose(2, 0, 1)
            ).to(self.cfg.MODEL.DEVICE)
            
            # Initialize variables
            integrated_grads = torch.zeros_like(input_tensor)
            valid_steps = 0
            
            # Compute integrated gradients with detection tracking
            for i in range(steps):
                if i % 5 == 0 or i == steps - 1:
                    print(f"    Progress: {i+1}/{steps} steps")
                
                try:
                    # Linear interpolation between baseline and input
                    alpha = float(i) / (steps - 1) if steps > 1 else 1.0
                    interpolated_input = baseline_tensor + alpha * (input_tensor - baseline_tensor)
                    interpolated_input.requires_grad = True
                    
                    # Forward pass
                    curr_outputs = self.model([{
                        "image": interpolated_input,
                        "height": height,
                        "width": width
                    }])
                    
                    instances = curr_outputs[0]["instances"]
                    
                    if len(instances) == 0:
                        print(f"    Step {i}: No instances detected, skipping")
                        continue
                    
                    # Find the best matching detection to our reference
                    best_match_idx = self._find_best_matching_detection(
                        instances, ref_bbox, ref_class
                    )
                    
                    if best_match_idx is None:
                        print(f"    Step {i}: No matching detection found, skipping")
                        continue
                    
                    detection_score = instances.scores[best_match_idx]
                    
                    # Backward pass
                    self.model.zero_grad()
                    detection_score.backward(retain_graph=False)
                    
                    # Check if gradients were computed
                    if interpolated_input.grad is None:
                        print(f"    Warning: No gradients computed at step {i}")
                        continue
                    
                    # Accumulate gradients
                    integrated_grads += interpolated_input.grad.data
                    valid_steps += 1
                    
                except Exception as step_error:
                    print(f"    Error at step {i}: {step_error}")
                    continue
            
            print(f"    Completed gradient accumulation from {valid_steps}/{steps} valid steps")
            
            if valid_steps == 0:
                print("    Error: No valid steps for gradient computation")
                return None, ref_outputs[0]
            
            # Average the gradients and multiply by (input - baseline)
            integrated_grads = integrated_grads / valid_steps
            integrated_grads = (input_tensor - baseline_tensor) * integrated_grads
            
            # Take absolute value and average across channels
            saliency_map_raw = integrated_grads.abs().mean(dim=0).cpu().numpy()
            
            print(f"    Saliency map shape: {saliency_map_raw.shape}")
            print(f"    Saliency range: [{saliency_map_raw.min():.6f}, {saliency_map_raw.max():.6f}]")
            
            # Normalize
            if saliency_map_raw.max() > saliency_map_raw.min():
                saliency_map_raw = (saliency_map_raw - saliency_map_raw.min()) / (
                    saliency_map_raw.max() - saliency_map_raw.min()
                )
            else:
                print("    Warning: Constant saliency map, setting to zeros")
                saliency_map_raw = np.zeros_like(saliency_map_raw)
            
            print("    Integrated Gradients completed successfully")
            return saliency_map_raw, ref_outputs[0]
            
        except Exception as e:
            print(f"    Error in Integrated Gradients: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _find_best_matching_detection(self, instances, ref_bbox, ref_class):
        """Find the detection that best matches the reference detection."""
        if len(instances) == 0:
            return None
            
        best_idx = None
        best_iou = 0.0
        
        for i, (bbox, pred_class) in enumerate(zip(
            instances.pred_boxes.tensor.detach().cpu().numpy(), 
            instances.pred_classes.detach().cpu().numpy()
        )):
            # Prefer same class predictions
            class_bonus = 0.1 if pred_class == ref_class else 0.0
            
            # Calculate IoU with reference bbox
            iou = self._calculate_iou(bbox, ref_bbox)
            combined_score = iou + class_bonus
            
            if combined_score > best_iou:
                best_iou = combined_score
                best_idx = i
        
        # Only return if we have a reasonable match (IoU > 0.3 or same class)
        if best_iou > 0.3:
            return best_idx
        else:
            return None
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes."""
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _smoothgrad(self, transformed_img, height, width, target_instance_idx, 
                   base_method='vanilla', noise_samples=50, noise_level=0.15):
        """SmoothGrad method (Smilkov et al. 2017) with different base methods."""
        
        accumulated_saliency = None
        outputs = None
        
        for i in range(noise_samples):
            # Add noise to input
            noise = np.random.normal(0, noise_level * 255, transformed_img.shape).astype(np.float32)
            noisy_img = np.clip(transformed_img + noise, 0, 255)
            
            # Compute saliency with base method
            if base_method == 'vanilla':
                saliency, curr_outputs = self._vanilla_gradients(
                    noisy_img, height, width, target_instance_idx
                )
            elif base_method == 'input':
                saliency, curr_outputs = self._input_gradients(
                    noisy_img, height, width, target_instance_idx
                )
            else:
                raise ValueError(f"Unknown base method for SmoothGrad: {base_method}")
            
            if saliency is None:
                return None, curr_outputs
                
            # Store outputs from first iteration
            if outputs is None:
                outputs = curr_outputs
                
            # Accumulate saliency maps
            if accumulated_saliency is None:
                accumulated_saliency = saliency
            else:
                accumulated_saliency += saliency
        
        # Average across all noise samples
        saliency_map_raw = accumulated_saliency / noise_samples
        
        return saliency_map_raw, outputs


class EnhancedMisclassificationSaliencyAnalyzer:
    """
    Enhanced analyzer supporting multiple attribution methods.
    """
    
    def __init__(self, config_file="configs/Cityscapes/mask_rcnn_R_50_FPN.yaml", 
                 model_weights="detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"):
        """Initialize the enhanced misclassification analyzer."""
        # Set up Detectron2 configuration (same as before)
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"
        
        cfg.freeze()
        self.cfg = cfg
        
        # Build and load model
        self.model = build_model(cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        
        # Initialize enhanced saliency map generator
        self.saliency_generator = EnhancedDetectionSaliencyMap(self.model, cfg)
        
        # Get dataset metadata
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.class_names = self.metadata.thing_classes
        
        print(f"Initialized enhanced detection saliency analyzer")
        
    def analyze_misclassified_instance_multimethod(self, img_path, pred_instance_idx, 
                                                  true_class, pred_class,
                                                  methods=['vanilla_gradients', 'input_gradients', 'integrated_gradients'],
                                                  output_dir="enhanced_detection_saliency_analysis"):
        """
        Analyze a single misclassified detection using multiple saliency methods.
        
        Args:
            img_path: Path to the image containing the misclassification
            pred_instance_idx: Index of the misclassified detection in model output
            true_class: Ground truth class name
            pred_class: Predicted (incorrect) class name
            methods: List of attribution methods to use
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing analysis results for all methods
        """
        # Create output directory for this case
        image_name = Path(img_path).stem
        case_dir = os.path.join(
            output_dir, 
            f"{image_name}_{pred_instance_idx}_{true_class}_to_{pred_class}"
        )
        os.makedirs(case_dir, exist_ok=True)
        
        results = {
            "image_path": img_path,
            "pred_instance_idx": pred_instance_idx,
            "true_class": true_class,
            "pred_class": pred_class,
            "methods": {}
        }
        
        # Read original image for visualization
        image = read_image(img_path, format="BGR")
        
        # Process each method
        for method in methods:
            print(f"  Computing {method} saliency map...")
            
            try:
                # Method-specific parameters
                method_kwargs = {}
                if method == 'integrated_gradients':
                    method_kwargs = {'steps': 15, 'baseline': 'blur'}  # Use blur baseline and fewer steps
                elif method.startswith('smoothgrad'):
                    method_kwargs = {'noise_samples': 25, 'noise_level': 0.1}  # Reduced for faster computation
                
                # Compute saliency map
                saliency_map, saliency_map_raw, output = self.saliency_generator.compute_instance_saliency(
                    img_path, pred_instance_idx, method=method, **method_kwargs
                )
                
                if saliency_map is not None and output is not None:
                    print(f"    Successfully computed {method} saliency map")
                    instance = output["instances"][pred_instance_idx]
                    
                    # Save method-specific visualizations
                    method_dir = os.path.join(case_dir, method)
                    os.makedirs(method_dir, exist_ok=True)
                    
                    self._save_saliency_visualizations(
                        image, saliency_map, instance, method_dir, 
                        true_class=true_class, method_name=method
                    )
                    
                    # Compute saliency statistics
                    bbox = instance.pred_boxes.tensor.detach().cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        bbox_saliency = saliency_map[y1:y2, x1:x2]
                        
                        results["methods"][method] = {
                            "success": True,
                            "pred_score": float(instance.scores.detach().cpu().item()),
                            "saliency_statistics": {
                                "full_image": {
                                    "mean": float(np.mean(saliency_map)),
                                    "max": float(np.max(saliency_map)),
                                    "std": float(np.std(saliency_map))
                                },
                                "within_bbox": {
                                    "mean": float(np.mean(bbox_saliency)),
                                    "max": float(np.max(bbox_saliency)),
                                    "std": float(np.std(bbox_saliency))
                                }
                            }
                        }
                        
                        # Save raw saliency map
                        np.save(os.path.join(method_dir, "saliency_map.npy"), saliency_map)
                        print(f"    Saved {method} results to {method_dir}")
                    else:
                        print(f"    Warning: Invalid bounding box for {method}")
                        results["methods"][method] = {"success": False, "error": "Invalid bounding box"}
                    
                else:
                    print(f"    Failed to compute {method} saliency map")
                    results["methods"][method] = {"success": False, "error": "Failed to compute saliency"}
                    
            except Exception as e:
                print(f"  Error with {method}: {str(e)}")
                import traceback
                traceback.print_exc()
                results["methods"][method] = {"success": False, "error": str(e)}
        
        # Create comparison visualization only for successful methods
        successful_methods = [method for method, result in results["methods"].items() 
                            if result.get("success", False)]
        print(f"  Successful methods: {successful_methods}")
        
        if len(successful_methods) >= 2:
            try:
                self._create_method_comparison(image, results, case_dir, true_class, pred_class)
                print(f"  Created method comparison")
            except Exception as e:
                print(f"  Failed to create method comparison: {e}")
        
        # Save analysis results as JSON
        with open(os.path.join(case_dir, "multimethod_analysis_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _save_saliency_visualizations(self, image, saliency_map, instance, output_dir, 
                                    true_class=None, method_name="saliency"):
        """Save visualizations with method name in title."""
        # Get bounding box coordinates
        bbox = instance.pred_boxes.tensor[0].detach().cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding around the box for better context
        height, width = image.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * 0.2)
        pad_y = int(box_height * 0.2)
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(width, x2 + pad_x)
        y2_pad = min(height, y2 + pad_y)
        
        # Extract crops
        image_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        saliency_crop = saliency_map[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Create focused visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        # Draw bounding box on cropped region
        box_in_crop = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]
        image_crop_with_box = image_crop.copy()
        cv2.rectangle(image_crop_with_box, 
                     (box_in_crop[0], box_in_crop[1]), 
                     (box_in_crop[2], box_in_crop[3]), 
                     (0, 255, 0), 2)
        plt.imshow(image_crop_with_box[:, :, ::-1])
        plt.title(f"Detection: {self.class_names[instance.pred_classes[0]]}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(saliency_crop, cmap='hot')
        plt.title(f"{method_name.replace('_', ' ').title()}")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image_crop[:, :, ::-1])
        plt.imshow(saliency_crop, cmap='hot', alpha=0.6)
        plt.title("Attribution Overlay")
        plt.axis('off')
        
        plt.suptitle(f"{method_name.replace('_', ' ').title()} Attribution - "
                    f"Predicted: {self.class_names[instance.pred_classes[0]]} "
                    f"(should be: {true_class if true_class else 'unknown'})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method_name}_attribution.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_method_comparison(self, image, results, output_dir, true_class, pred_class):
        """Create a comparison visualization of different attribution methods."""
        successful_methods = [method for method, result in results["methods"].items() 
                            if result.get("success", False)]
        
        if len(successful_methods) < 2:
            return
            
        # Load saliency maps for comparison
        saliency_maps = {}
        for method in successful_methods:
            saliency_path = os.path.join(output_dir, method, "saliency_map.npy")
            if os.path.exists(saliency_path):
                saliency_maps[method] = np.load(saliency_path)
        
        if len(saliency_maps) < 2:
            return
            
        # Create comparison plot
        n_methods = len(saliency_maps)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (method, saliency_map) in enumerate(saliency_maps.items()):
            # Top row: saliency maps
            axes[0, i].imshow(saliency_map, cmap='hot')
            axes[0, i].set_title(f"{method.replace('_', ' ').title()}")
            axes[0, i].axis('off')
            
            # Bottom row: overlays
            axes[1, i].imshow(image[:, :, ::-1])
            axes[1, i].imshow(saliency_map, cmap='hot', alpha=0.5)
            axes[1, i].set_title(f"{method.replace('_', ' ').title()} Overlay")
            axes[1, i].axis('off')
        
        plt.suptitle(f"Attribution Method Comparison\n"
                    f"Predicted: {pred_class} (should be: {true_class})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "method_comparison.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """
    Main function to run enhanced detection saliency analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced object detection saliency analysis with multiple attribution methods"
    )
    parser.add_argument("--fn_person_rider", 
                       default="output/misclassification_analysis_complete/fn_person_rider_details.csv",
                       help="CSV file with person misclassified as rider")
    parser.add_argument("--fn_rider_person", 
                       default="output/misclassification_analysis_complete/fn_rider_person_details.csv",
                       help="CSV file with rider misclassified as person")
    parser.add_argument("--output_dir", 
                       default="output/enhanced_detection_saliency_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--methods", nargs='+',
                       default=['vanilla_gradients', 'input_gradients', 'integrated_gradients'],
                       help="Attribution methods to use")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to process (for testing)")
    args = parser.parse_args()
    
    # Initialize the enhanced analyzer
    analyzer = EnhancedMisclassificationSaliencyAnalyzer()
    
    print(f"Using attribution methods: {args.methods}")
    
    # Process samples from both CSV files
    total_successful = 0
    total_failed = 0
    
    for csv_path in [args.fn_person_rider, args.fn_rider_person]:
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            continue
            
        print(f"\nProcessing: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Limit samples for testing
        df_sample = df.head(args.max_samples)
        print(f"Processing {len(df_sample)} samples (limited from {len(df)})")
        
        csv_name = Path(csv_path).stem
        output_subdir = os.path.join(args.output_dir, csv_name)
        
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Analyzing"):
            try:
                print(f"\nCase {idx+1}: {row['file_name']}")
                print(f"  Instance {row['pred_instance_idx']}: {row['gt_class']} -> {row['pred_class']}")
                
                analyzer.analyze_misclassified_instance_multimethod(
                    row['file_name'],
                    row['pred_instance_idx'],
                    row['gt_class'],
                    row['pred_class'],
                    methods=args.methods,
                    output_dir=output_subdir
                )
                total_successful += 1
                
            except Exception as e:
                print(f"  Failed: {e}")
                total_failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Enhanced Detection Saliency Analysis Complete!")
    print("="*60)
    print(f"Total cases processed: {total_successful + total_failed}")
    print(f"Successful: {total_successful}")
    if total_failed > 0:
        print(f"Failed: {total_failed}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()