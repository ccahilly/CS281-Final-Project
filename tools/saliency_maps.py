#!/usr/bin/env python
"""
Detection-Aware Saliency Map Analysis for Detectron2 Mask R-CNN.

This implementation computes saliency maps specifically for object detection tasks,
showing which input pixels contribute to a specific detection instance's classification score.
Unlike traditional saliency maps for image classification, this analyzes individual
detected objects within an image.
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


class DetectionSaliencyMap:
    """
    Computes saliency maps for object detection models.
    
    This class generates pixel-level attribution maps showing which input pixels
    influence a specific detection instance's classification score. Unlike traditional
    saliency maps that analyze whole-image classification, this method is instance-specific
    for object detection tasks.
    """
    
    def __init__(self, model, cfg):
        """
        Initialize the detection saliency map generator.
        
        Args:
            model: Detectron2 model (e.g., Mask R-CNN)
            cfg: Detectron2 configuration
        """
        self.model = model
        self.cfg = cfg
        self.model.eval()
        
    def compute_instance_saliency(self, img_path, target_instance_idx):
        """
        Compute saliency map for a specific detected instance.
        
        This method calculates which input pixels most influence the model's
        confidence score for a particular detection (bounding box + class prediction).
        
        Args:
            img_path: Path to the input image
            target_instance_idx: Index of the detection instance to analyze
            
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
        
        # Create tensor with gradient tracking enabled
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        input_tensor.requires_grad = True
        
        # Forward pass through detection model
        outputs = self.model([{
            "image": input_tensor,
            "height": height,
            "width": width
        }])
        
        instances = outputs[0]["instances"]
        
        # Check if target instance exists
        if target_instance_idx >= len(instances):
            return None, None, outputs[0]
            
        # Get the classification score for the target detection
        # This is the confidence that this bounding box contains the predicted class
        detection_score = instances.scores[target_instance_idx]
        
        # Compute gradients of the detection score with respect to input pixels
        self.model.zero_grad()
        detection_score.backward()
        
        # Extract gradients - these show pixel sensitivity
        # Absolute value shows importance regardless of direction
        pixel_gradients = input_tensor.grad.data.abs()
        
        # Average across color channels to get spatial saliency map
        # Shape: [C, H, W] -> [H, W]
        saliency_map_raw = pixel_gradients.mean(dim=0).cpu().numpy()
        
        # Normalize to [0, 1] range for visualization
        saliency_map_raw = (saliency_map_raw - saliency_map_raw.min()) / (
            saliency_map_raw.max() - saliency_map_raw.min() + 1e-8
        )
        
        # Resize saliency map to match original image dimensions
        saliency_map = cv2.resize(saliency_map_raw, (width, height))
        
        return saliency_map, saliency_map_raw, outputs[0]


class MisclassificationSaliencyAnalyzer:
    """
    Analyzes object detection misclassifications using detection-aware saliency maps.
    
    This class specifically focuses on understanding why certain objects are
    misclassified (e.g., person vs. rider confusion in Cityscapes dataset)
    by visualizing which pixels contribute to incorrect predictions.
    """
    
    def __init__(self, config_file="configs/Cityscapes/mask_rcnn_R_50_FPN.yaml", 
                 model_weights="detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"):
        """Initialize the misclassification analyzer."""
        # Set up Detectron2 configuration
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
        
        # Initialize saliency map generator
        self.saliency_generator = DetectionSaliencyMap(self.model, cfg)
        
        # Get dataset metadata
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.class_names = self.metadata.thing_classes
        
        print(f"Initialized detection saliency analyzer for misclassification analysis")
        
    def analyze_misclassified_instance(self, img_path, pred_instance_idx, true_class, pred_class,
                                      output_dir="detection_saliency_analysis"):
        """
        Analyze a single misclassified detection using saliency maps.
        
        Args:
            img_path: Path to the image containing the misclassification
            pred_instance_idx: Index of the misclassified detection in model output
            true_class: Ground truth class name
            pred_class: Predicted (incorrect) class name
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing analysis results
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
            "pred_class": pred_class
        }
        
        try:
            # Compute detection-specific saliency map
            print(f"  Computing detection saliency map for instance {pred_instance_idx}...")
            saliency_map, saliency_map_raw, output = self.saliency_generator.compute_instance_saliency(
                img_path, pred_instance_idx
            )
            
            if saliency_map is not None:
                # Read original image for visualization
                image = read_image(img_path, format="BGR")
                instance = output["instances"][pred_instance_idx]
                
                # Save visualizations
                self._save_saliency_visualizations(
                    image, saliency_map, instance, case_dir, true_class=true_class
                )
                
                # Extract detection information
                bbox = instance.pred_boxes.tensor.detach().cpu().numpy()[0]
                
                # Compute saliency statistics within the bounding box
                x1, y1, x2, y2 = map(int, bbox)
                bbox_saliency = saliency_map[y1:y2, x1:x2]
                
                results["detection_analysis"] = {
                    "pred_score": float(instance.scores.detach().cpu().item()),
                    "pred_class_id": int(instance.pred_classes.detach().cpu().item()),
                    "pred_box": bbox.tolist(),
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
                
                # Save raw saliency map for further analysis
                np.save(os.path.join(case_dir, "saliency_map.npy"), saliency_map)
                
        except Exception as e:
            print(f"  Error during analysis: {str(e)}")
            results["error"] = str(e)
            
        # Save analysis results as JSON
        with open(os.path.join(case_dir, "analysis_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _save_saliency_visualizations(self, image, saliency_map, instance, output_dir, true_class=None):
        """
        Create and save visualizations of the detection saliency analysis.
        
        Creates both full image and zoomed views showing:
        1. The detection with bounding box
        2. The saliency map heatmap
        3. Overlay of saliency on the original image
        
        Args:
            image: Original image array
            saliency_map: Computed saliency map
            instance: Detection instance from model output
            output_dir: Directory to save visualizations
            true_class: Ground truth class name (optional, for misclassification context)
        """
        # Get bounding box coordinates
        bbox = instance.pred_boxes.tensor[0].detach().cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding around the box for better context (20% of box dimensions)
        height, width = image.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * 0.2)
        pad_y = int(box_height * 0.2)
        
        # Calculate padded region with bounds checking
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(width, x2 + pad_x)
        y2_pad = min(height, y2 + pad_y)
        
        # Extract crops
        image_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        saliency_crop = saliency_map[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 10))
        
        # Full image views
        plt.subplot(2, 3, 1)
        v = Visualizer(image, self.metadata, scale=1.0)
        vis_output = v.draw_instance_predictions(instance.to("cpu"))
        plt.imshow(vis_output.get_image()[:, :, ::-1])
        plt.title(f"Detection: {self.class_names[instance.pred_classes[0]]} "
                 f"(score: {instance.scores[0]:.3f})")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(saliency_map, cmap='hot')
        plt.title("Detection Saliency Map")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(image[:, :, ::-1])
        plt.imshow(saliency_map, cmap='hot', alpha=0.5)
        plt.title("Saliency Overlay")
        plt.axis('off')
        
        # Zoomed views focusing on the detection
        plt.subplot(2, 3, 4)
        # Draw bounding box on cropped region
        box_in_crop = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]
        image_crop_with_box = image_crop.copy()
        cv2.rectangle(image_crop_with_box, 
                     (box_in_crop[0], box_in_crop[1]), 
                     (box_in_crop[2], box_in_crop[3]), 
                     (0, 255, 0), 2)
        plt.imshow(image_crop_with_box[:, :, ::-1])
        plt.title("Zoomed: Detection Box")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(saliency_crop, cmap='hot')
        plt.title("Zoomed: Saliency Map")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(image_crop[:, :, ::-1])
        plt.imshow(saliency_crop, cmap='hot', alpha=0.5)
        plt.title("Zoomed: Saliency Overlay")
        plt.axis('off')
        
        plt.suptitle(f"Detection Saliency Analysis - Predicted: {self.class_names[instance.pred_classes[0]]} "
                    f"(should be: {true_class if true_class else 'unknown'})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "detection_saliency_analysis.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save a separate focused view
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_crop_with_box[:, :, ::-1])
        plt.title(f"Detection: {self.class_names[instance.pred_classes[0]]}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(saliency_crop, cmap='hot')
        plt.title("Pixel Importance for Detection")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image_crop[:, :, ::-1])
        plt.imshow(saliency_crop, cmap='hot', alpha=0.5)
        plt.title("Important Pixels Highlighted")
        plt.axis('off')
        
        plt.suptitle(f"Detection Saliency - Score: {instance.scores[0]:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "detection_saliency_focused.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def analyze_misclassification_dataset(csv_path, analyzer, output_base_dir):
    """
    Process a CSV file containing misclassification cases.
    
    Args:
        csv_path: Path to CSV with columns: file_name, pred_instance_idx, gt_class, pred_class
        analyzer: MisclassificationSaliencyAnalyzer instance
        output_base_dir: Base directory for outputs
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return 0, 0
        
    print(f"\nProcessing misclassifications from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    csv_name = Path(csv_path).stem
    output_dir = os.path.join(output_base_dir, csv_name)
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing detections"):
        try:
            print(f"\nAnalyzing case {idx+1}/{len(df)}: {row['file_name']}")
            print(f"  Instance {row['pred_instance_idx']}: {row['gt_class']} -> {row['pred_class']}")
            
            analyzer.analyze_misclassified_instance(
                row['file_name'],
                row['pred_instance_idx'],
                row['gt_class'],
                row['pred_class'],
                output_dir=output_dir
            )
            successful += 1
            
        except Exception as e:
            print(f"  Failed to analyze: {e}")
            failed += 1
            
    return successful, failed


def main():
    """
    Main function to run detection saliency analysis on misclassification cases.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze object detection misclassifications using detection-aware saliency maps"
    )
    parser.add_argument("--fn_person_rider", 
                       default="output/misclassification_analysis_complete/fn_person_rider_details.csv",
                       help="CSV file with person misclassified as rider")
    parser.add_argument("--fn_rider_person", 
                       default="output/misclassification_analysis_complete/fn_rider_person_details.csv",
                       help="CSV file with rider misclassified as person")
    parser.add_argument("--output_dir", 
                       default="output/detection_saliency_analysis",
                       help="Output directory for analysis results")
    args = parser.parse_args()
    
    # Initialize the analyzer
    analyzer = MisclassificationSaliencyAnalyzer()
    
    # Process both misclassification types
    total_successful = 0
    total_failed = 0
    
    for csv_path in [args.fn_person_rider, args.fn_rider_person]:
        successful, failed = analyze_misclassification_dataset(
            csv_path, analyzer, args.output_dir
        )
        total_successful += successful
        total_failed += failed
    
    # Print summary
    print("\n" + "="*60)
    print("Detection Saliency Analysis Complete!")
    print("="*60)
    print(f"Total cases analyzed: {total_successful + total_failed}")
    print(f"Successful: {total_successful}")
    if total_failed > 0:
        print(f"Failed: {total_failed}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()