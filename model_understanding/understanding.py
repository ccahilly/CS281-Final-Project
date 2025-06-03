#!/usr/bin/env python3
"""
Complete Model Understanding Script for Detectron2
Combines analysis and visualization to provide comprehensive model insights.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import the analyzer and visualizer classes
# (Assume they're saved as separate files or included in this script)
from model_analyzer import ModelAnalyzer
from model_visualizer import ModelVisualizer

# Add detectron2 to path if needed
try:
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
except ImportError:
    print("Please install detectron2 or add it to your Python path")
    sys.exit(1)

class ModelUnderstandingPipeline:
    """Complete pipeline for understanding Detectron2 models"""
    
    def __init__(self, config_file, weights_path=None, output_dir="model_analysis_output"):
        self.config_file = config_file
        self.weights_path = weights_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Initializing model understanding pipeline...")
        print(f"Config file: {config_file}")
        print(f"Weights: {weights_path if weights_path else 'Using default/random weights'}")
        print(f"Output directory: {output_dir}")
        
        # Initialize analyzer
        self.analyzer = ModelAnalyzer(config_file, weights_path)
        self.visualizer = ModelVisualizer()
        
    def run_complete_analysis(self, image_path=None):
        """Run the complete model analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPLETE MODEL ANALYSIS")
        print("="*80)
        
        # 1. Print architecture summary
        print("\n1. Analyzing model architecture...")
        self.analyzer.print_architecture_summary()
        
        # 2. Save architecture details
        self._save_architecture_details()
        
        # 3. Create visualizations
        print("\n2. Creating architecture visualizations...")
        self._create_visualizations()
        
        # 4. If image provided, analyze forward pass
        if image_path and os.path.exists(image_path):
            print(f"\n3. Analyzing forward pass with image: {image_path}")
            analysis_results = self.analyzer.analyze_forward_pass(image_path)
            self.analyzer.print_data_flow(analysis_results)
            
            # Save analysis results
            self.analyzer.save_analysis(
                analysis_results, 
                self.output_dir / "forward_pass_analysis.json"
            )
            
            # Create data flow visualization
            fig = self.visualizer.plot_data_flow(analysis_results)
            if fig:
                fig.savefig(self.output_dir / "data_flow.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            return analysis_results
        else:
            print("\n3. Skipping forward pass analysis (no valid image provided)")
            return None
    
    def _save_architecture_details(self):
        """Save detailed architecture information"""
        print("   Saving detailed layer information...")
        
        with open(self.output_dir / "layer_details.txt", 'w') as f:
            f.write("DETAILED LAYER INFORMATION\n")
            f.write("="*50 + "\n\n")
            
            for name, info in self.analyzer.layer_info.items():
                f.write(f"Layer: {name}\n")
                f.write(f"  Type: {info['type']}\n")
                f.write(f"  Parameters: {info['parameters']['total']:,} total, "
                       f"{info['parameters']['trainable']:,} trainable\n")
                f.write(f"  Parent: {info['parent']}\n")
                
                # Get additional details
                details = self.analyzer.get_layer_details(name)
                if details:
                    for key, value in details.items():
                        if key not in ['name', 'type', 'parameters', 'parent']:
                            f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    def _create_visualizations(self):
        """Create and save all visualizations"""
        print("   Creating architecture overview...")
        fig1 = self.visualizer.plot_architecture_overview()
        fig1.savefig(self.output_dir / "architecture_overview.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print("   Creating backbone details...")
        fig2 = self.visualizer.plot_backbone_details()
        fig2.savefig(self.output_dir / "backbone_details.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print("   Creating comprehensive report...")
        fig3 = self.visualizer.create_summary_report(
            self.analyzer, 
            save_path=self.output_dir / "complete_report.png"
        )
        plt.close(fig3)
    
    def analyze_specific_layers(self, layer_names):
        """Analyze specific layers in detail"""
        print(f"\nAnalyzing specific layers: {layer_names}")
        
        results = {}
        for layer_name in layer_names:
            details = self.analyzer.get_layer_details(layer_name)
            if details:
                results[layer_name] = details
                print(f"\n{layer_name}:")
                for key, value in details.items():
                    if key != 'name':
                        print(f"  {key}: {value}")
            else:
                print(f"Layer '{layer_name}' not found")
        
        return results
    
    def understand_model_components(self):
        """Provide detailed explanations of model components"""
        print("\n" + "="*80)
        print("MODEL COMPONENTS EXPLANATION")
        print("="*80)
        
        explanations = {
            'backbone': """
BACKBONE (ResNet-50 + FPN):
- ResNet-50: Deep convolutional network for feature extraction
- Stages: stem (7x7 conv) → res2-5 (residual blocks)
- FPN: Feature Pyramid Network for multi-scale features
- Output: 5 feature maps at different scales (P2-P6)""",
            
            'proposal_generator': """
REGION PROPOSAL NETWORK (RPN):
- Generates object proposals from FPN features
- Uses 3x3 conv + objectness classifier + box regression
- Operates on all FPN levels (P2-P6)
- Output: ~1000 object proposals per image""",
            
            'roi_heads': """
ROI HEADS:
- Takes proposals from RPN + FPN features  
- ROI Align: Extract fixed-size features from variable proposals
- Two branches: Box Head + Mask Head
- Box Head: Classification + bounding box regression
- Mask Head: Instance segmentation masks""",
            
            'data_flow': """
DATA FLOW:
1. Image → Backbone → Multi-scale features (P2-P6)
2. Features → RPN → Object proposals
3. Proposals + Features → ROI Heads → Final predictions
4. Box Head: Object classes + refined boxes
5. Mask Head: Pixel-level segmentation masks"""
        }
        
        for component, explanation in explanations.items():
            print(f"\n{component.upper()}:")
            print(explanation)
    
    def get_model_summary(self):
        """Get a concise model summary"""
        total_params = sum(info['parameters']['total'] 
                          for info in self.analyzer.layer_info.values())
        
        trainable_params = sum(info['parameters']['trainable'] 
                              for info in self.analyzer.layer_info.values())
        
        layer_count = len(self.analyzer.layer_info)
        
        summary = {
            'architecture': 'Mask R-CNN with ResNet-50 + FPN backbone',
            'total_parameters': f"{total_params:,}",
            'trainable_parameters': f"{trainable_params:,}",
            'total_layers': layer_count,
            'input_size': 'Variable (resized to min 800px)',
            'output': 'Bounding boxes, classes, and segmentation masks',
            'framework': 'Detectron2'
        }
        
        return summary

def main():
    """Main function demonstrating the complete pipeline"""
    
    # Configuration - Update these paths for your setup
    config_file = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
    weights_path = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"  # Set to your model weights path
    image_path = "datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png"    # Set to a test image path
    output_dir = "output/model_understanding_output"
    
    try:
        # Initialize pipeline
        pipeline = ModelUnderstandingPipeline(config_file, weights_path, output_dir)
        
        # Run complete analysis
        analysis_results = pipeline.run_complete_analysis(image_path)
        
        # Get model summary
        summary = pipeline.get_model_summary()
        print(f"\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Explain model components
        pipeline.understand_model_components()
        
        # Analyze specific interesting layers
        interesting_layers = [
            'backbone.bottom_up.stem.conv1',
            'backbone.fpn_lateral2',
            'proposal_generator.rpn_head.conv',
            'roi_heads.box_head.fc1',
            'roi_heads.mask_head.mask_fcn1'
        ]
        
        pipeline.analyze_specific_layers(interesting_layers)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"All outputs saved to: {output_dir}/")
        print("Files created:")
        print("- architecture_overview.png: High-level architecture diagram")
        print("- backbone_details.png: Detailed backbone structure")
        print("- complete_report.png: Comprehensive visual report")
        print("- layer_details.txt: Detailed layer information")
        if analysis_results:
            print("- forward_pass_analysis.json: Forward pass analysis")
            print("- data_flow.png: Data flow visualization")
        
        # Show key insights
        print(f"\nKEY INSIGHTS:")
        print(f"- Total parameters: {summary['total_parameters']}")
        print(f"- Architecture: {summary['architecture']}")
        print(f"- Main components: Backbone → FPN → RPN → ROI Heads")
        print(f"- Output: Boxes, classes, and segmentation masks")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Make sure detectron2 is installed and config file exists")

if __name__ == "__main__":
    main()