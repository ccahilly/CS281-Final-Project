#!/usr/bin/env python
"""
Working GradCAM implementation for Detectron2 Mask R-CNN.
This version avoids the autograd issues by using input gradients instead of layer gradients.
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


class InputGradientCAM:
    """
    A simpler approach that uses input gradients to create attention maps.
    This avoids the complex autograd issues with layer hooks.
    """
    
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.model.eval()
        
    def compute(self, img_path, target_instance):
        """Compute gradient-based attention map."""
        # Read and preprocess image
        image = read_image(img_path, format="BGR")
        height, width = image.shape[:2]
        
        # Apply transforms
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        
        # Create tensor with gradient tracking
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        input_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model([{
            "image": input_tensor,
            "height": height,
            "width": width
        }])
        
        instances = outputs[0]["instances"]
        if target_instance >= len(instances):
            return None, None, outputs[0]
            
        # Get score and compute gradients
        score = instances.scores[target_instance]
        self.model.zero_grad()
        score.backward()
        
        # Get input gradients
        gradients = input_tensor.grad.data.abs()
        
        # Average across channels to get spatial attention
        attention = gradients.mean(dim=0).cpu().numpy()
        
        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Resize to original size
        attention_resized = cv2.resize(attention, (width, height))
        
        return attention_resized, attention, outputs[0]


class FeatureMapVisualizer:
    """
    Visualize feature maps from different layers without computing gradients.
    This gives insights into what the model sees at different stages.
    """
    
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.model.eval()
        self.features = {}
        
    def get_layer_features(self, img_path, layer_names):
        """Extract features from specified layers."""
        # Read and preprocess image
        image = read_image(img_path, format="BGR")
        height, width = image.shape[:2]
        
        # Apply transforms
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        
        # Register hooks to capture features
        handles = []
        self.features = {}
        
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(get_features(name))
                handles.append(handle)
                
        # Forward pass
        with torch.no_grad():
            outputs = self.model([{
                "image": input_tensor,
                "height": height,
                "width": width
            }])
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return self.features, outputs[0], (height, width)


class CityscapesAttentionAnalysis:
    """
    Analyze person/rider misclassifications using various attention techniques.
    """
    
    def __init__(self, config_file="configs/Cityscapes/mask_rcnn_R_50_FPN.yaml", 
                 model_weights="detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"):
        """Initialize the analyzer."""
        # Set up config
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
        
        # Build model
        self.model = build_model(cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        
        # Initialize analyzers
        self.input_grad_cam = InputGradientCAM(self.model, cfg)
        self.feature_viz = FeatureMapVisualizer(self.model, cfg)
        
        # Get metadata
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.class_names = self.metadata.thing_classes
        
        print(f"Initialized attention analysis")
        
    def analyze_misclassification(self, img_path, pred_instance_idx, true_class, pred_class,
                                  output_dir="attention_analysis"):
        """Analyze a misclassification case."""
        # Create output directory
        image_name = Path(img_path).stem
        case_dir = os.path.join(output_dir, f"{image_name}_{pred_instance_idx}_{true_class}_to_{pred_class}")
        os.makedirs(case_dir, exist_ok=True)
        
        results = {
            "image_path": img_path,
            "pred_instance_idx": pred_instance_idx,
            "true_class": true_class,
            "pred_class": pred_class
        }
        
        try:
            # 1. Input gradient attention
            print(f"  Computing input gradient attention...")
            attention, attention_orig, output = self.input_grad_cam.compute(
                img_path, pred_instance_idx
            )
            
            if attention is not None:
                # Read original image for visualization
                image = read_image(img_path, format="BGR")
                instance = output["instances"][pred_instance_idx]
                
                # Save visualization
                self._save_attention_viz(
                    image, attention, instance, "input_gradients", case_dir
                )
                
                results["input_gradient_analysis"] = {
                    "pred_score": float(instance.scores.detach().cpu().item()),
                    "pred_box": instance.pred_boxes.tensor.detach().cpu().numpy()[0].tolist(),
                    "attention_stats": {
                        "mean": float(np.mean(attention)),
                        "max": float(np.max(attention)),
                        "std": float(np.std(attention))
                    }
                }
                
            # 2. Feature map visualization
            print(f"  Extracting feature maps...")
            layer_names = [
                "backbone.bottom_up.res3.3.conv3",
                "backbone.bottom_up.res4.5.conv3", 
                "backbone.bottom_up.res5.2.conv3"
            ]
            
            features, output, (h, w) = self.feature_viz.get_layer_features(
                img_path, layer_names
            )
            
            # Visualize top activations from each layer
            for layer_name, feat in features.items():
                if feat is not None:
                    # Average across channels
                    feat_map = feat[0].mean(dim=0).cpu().numpy()
                    
                    # Normalize and resize
                    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                    feat_map = cv2.resize(feat_map, (w, h))
                    
                    # Save
                    layer_safe = layer_name.replace('.', '_')
                    np.save(os.path.join(case_dir, f"features_{layer_safe}.npy"), feat_map)
                    
                    # Create heatmap visualization
                    plt.figure(figsize=(8, 6))
                    plt.imshow(feat_map, cmap='hot')
                    plt.colorbar()
                    plt.title(f"Feature map: {layer_name}")
                    plt.savefig(os.path.join(case_dir, f"features_{layer_safe}.png"))
                    plt.close()
                    
        except Exception as e:
            print(f"  Error: {str(e)}")
            results["error"] = str(e)
            
        # Save results
        with open(os.path.join(case_dir, "analysis_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _save_attention_viz(self, image, attention, instance, method_name, output_dir):
        """Save attention visualization."""
        # Get bounding box
        bbox = instance.pred_boxes.tensor[0].detach().cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding around the box (20% of box dimensions)
        height, width = image.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * 0.2)
        pad_y = int(box_height * 0.2)
        
        # Calculate padded box with bounds checking
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(width, x2 + pad_x)
        y2_pad = min(height, y2 + pad_y)
        
        # Crop images and attention map
        image_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        attention_crop = attention[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Create figure with both full and cropped views
        fig = plt.figure(figsize=(20, 10))
        
        # Full view
        plt.subplot(2, 3, 1)
        v = Visualizer(image, self.metadata, scale=1.0)
        vis_output = v.draw_instance_predictions(instance.to("cpu"))
        plt.imshow(vis_output.get_image()[:, :, ::-1])
        plt.title("Full Image - Prediction")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(attention, cmap='hot')
        plt.title(f"Full Image - {method_name} attention")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(image[:, :, ::-1])
        plt.imshow(attention, cmap='hot', alpha=0.5)
        plt.title("Full Image - Overlay")
        plt.axis('off')
        
        # Cropped view (zoomed to bounding box)
        plt.subplot(2, 3, 4)
        # Draw box on cropped region
        box_in_crop = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]
        image_crop_with_box = image_crop.copy()
        cv2.rectangle(image_crop_with_box, 
                     (box_in_crop[0], box_in_crop[1]), 
                     (box_in_crop[2], box_in_crop[3]), 
                     (0, 255, 0), 2)
        plt.imshow(image_crop_with_box[:, :, ::-1])
        plt.title("Zoomed - Prediction Box")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(attention_crop, cmap='hot')
        plt.title(f"Zoomed - {method_name} attention")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(image_crop[:, :, ::-1])
        plt.imshow(attention_crop, cmap='hot', alpha=0.5)
        plt.title("Zoomed - Overlay")
        plt.axis('off')
        
        plt.suptitle(f"{method_name.replace('_', ' ').title()} Analysis - {instance.pred_classes[0].item()} (score: {instance.scores[0].item():.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method_name}_visualization.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save just the zoomed view separately
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_crop_with_box[:, :, ::-1])
        plt.title("Prediction Box")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(attention_crop, cmap='hot')
        plt.title(f"{method_name} attention")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image_crop[:, :, ::-1])
        plt.imshow(attention_crop, cmap='hot', alpha=0.5)
        plt.title("Attention Overlay")
        plt.axis('off')
        
        plt.suptitle(f"Zoomed View - {instance.pred_classes[0].item()} (score: {instance.scores[0].item():.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method_name}_zoomed.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_person_rider", 
                       default="output/misclassification_analysis_complete/fn_person_rider_details.csv")
    parser.add_argument("--fn_rider_person", 
                       default="output/misclassification_analysis_complete/fn_rider_person_details.csv")
    parser.add_argument("--output_dir", default="attention_analysis")
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CityscapesAttentionAnalysis()
    
    # Process misclassifications
    for csv_path in [args.fn_person_rider, args.fn_rider_person]:
        if os.path.exists(csv_path):
            print(f"\nProcessing: {csv_path}")
            df = pd.read_csv(csv_path)
            
            csv_name = Path(csv_path).stem
            output_dir = os.path.join(args.output_dir, csv_name)
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                print(f"\nCase {idx+1}/{len(df)}: {row['file_name']}")
                
                try:
                    analyzer.analyze_misclassification(
                        row['file_name'],
                        row['pred_instance_idx'],
                        row['gt_class'],
                        row['pred_class'],
                        output_dir=output_dir
                    )
                except Exception as e:
                    print(f"  Failed: {e}")
                    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()