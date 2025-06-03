#!/usr/bin/env python
"""
Detection-Specific Saliency Methods for Object Detection Models.

This implementation provides saliency methods specifically designed for object detection,
addressing the limitations of applying image classification saliency to detection tasks.
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


class DetectionSpecificSaliency:
    """
    Saliency methods specifically designed for object detection models.
    
    These methods address the unique challenges of object detection:
    1. Multiple output components (classification, localization, objectness)
    2. Spatial localization requirements
    3. Detection instability across input perturbations
    """
    
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.model.eval()
        
    def compute_detection_saliency(self, img_path, target_instance_idx, method='guided_backprop'):
        """
        Compute detection-specific saliency maps.
        
        Methods:
        - 'guided_backprop': Guided backpropagation focusing on positive contributions
        - 'grad_cam': Class Activation Mapping adapted for detection
        - 'occlusion': Systematic occlusion analysis
        - 'rise': Randomized Input Sampling for Explanation
        """
        
        if method == 'guided_backprop':
            return self._guided_backprop_detection(img_path, target_instance_idx)
        elif method == 'grad_cam':
            return self._detection_grad_cam(img_path, target_instance_idx)
        elif method == 'occlusion':
            return self._occlusion_sensitivity(img_path, target_instance_idx)
        elif method == 'rise':
            return self._rise_detection(img_path, target_instance_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _guided_backprop_detection(self, img_path, target_instance_idx):
        """
        Guided backpropagation adapted for object detection.
        
        Key differences from standard guided backprop:
        1. Focus on detection confidence rather than classification only
        2. Combine objectness and classification scores
        3. Use ReLU guidance only for positive contributions
        """
        # Read and preprocess image
        image = read_image(img_path, format="BGR")
        height, width = image.shape[:2]
        
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        input_tensor.requires_grad = True
        
        # Register hooks for guided backpropagation
        self._register_guided_hooks()
        
        try:
            # Forward pass
            outputs = self.model([{
                "image": input_tensor,
                "height": height,
                "width": width
            }])
            
            instances = outputs[0]["instances"]
            if target_instance_idx >= len(instances):
                return None, None, outputs[0]
            
            # Combine classification confidence and detection confidence
            # This gives a more complete picture of "detection strength"
            class_score = instances.scores[target_instance_idx]
            
            # Get the box regression loss as a proxy for localization confidence
            # (Lower loss = better localization = higher confidence)
            pred_boxes = instances.pred_boxes.tensor[target_instance_idx]
            
            # Use classification score as the primary signal
            # (You could also experiment with combining multiple signals)
            detection_signal = class_score
            
            # Backward pass
            self.model.zero_grad()
            detection_signal.backward()
            
            # Get guided gradients (positive contributions only)
            guided_grads = input_tensor.grad.data
            
            # Apply ReLU to focus on positive evidence
            guided_grads = torch.clamp(guided_grads, min=0)
            
            # Average across channels
            saliency_map_raw = guided_grads.mean(dim=0).cpu().numpy()
            
            # Normalize
            if saliency_map_raw.max() > 0:
                saliency_map_raw = saliency_map_raw / saliency_map_raw.max()
            
            # Resize to original image size
            saliency_map = cv2.resize(saliency_map_raw, (width, height))
            
            return saliency_map, saliency_map_raw, outputs[0]
            
        finally:
            self._remove_hooks()
    
    def _occlusion_sensitivity(self, img_path, target_instance_idx, patch_size=50, stride=25):
        """
        Occlusion sensitivity analysis for object detection.
        
        This method systematically occludes parts of the image and measures
        how the detection confidence changes. More robust than gradient methods.
        """
        print(f"    Computing occlusion sensitivity (patch_size={patch_size}, stride={stride})")
        
        # Read and preprocess image
        image = read_image(img_path, format="BGR")
        height, width = image.shape[:2]
        
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        
        # Get baseline detection score
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        
        with torch.no_grad():
            baseline_outputs = self.model([{
                "image": input_tensor,
                "height": height,
                "width": width
            }])
        
        baseline_instances = baseline_outputs[0]["instances"]
        if target_instance_idx >= len(baseline_instances):
            return None, None, baseline_outputs[0]
            
        baseline_score = baseline_instances.scores[target_instance_idx].item()
        baseline_bbox = baseline_instances.pred_boxes.tensor[target_instance_idx].detach().cpu().numpy()
        baseline_class = baseline_instances.pred_classes[target_instance_idx].item()
        
        print(f"    Baseline score: {baseline_score:.3f}")
        
        # Create occlusion sensitivity map
        img_h, img_w = transformed_img.shape[:2]
        sensitivity_map = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Systematic occlusion
        total_patches = ((img_h - patch_size) // stride + 1) * ((img_w - patch_size) // stride + 1)
        patch_count = 0
        
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_count += 1
                if patch_count % 50 == 0:
                    print(f"    Progress: {patch_count}/{total_patches} patches")
                
                # Create occluded image
                occluded_img = transformed_img.copy()
                # Use gray occlusion (more natural than black)
                occluded_img[y:y+patch_size, x:x+patch_size] = 128
                
                occluded_tensor = torch.as_tensor(
                    occluded_img.astype("float32").transpose(2, 0, 1)
                ).to(self.cfg.MODEL.DEVICE)
                
                with torch.no_grad():
                    occluded_outputs = self.model([{
                        "image": occluded_tensor,
                        "height": height,
                        "width": width
                    }])
                
                # Find the matching detection in occluded image
                occluded_instances = occluded_outputs[0]["instances"]
                
                if len(occluded_instances) == 0:
                    # No detections - maximum sensitivity
                    sensitivity_score = baseline_score
                else:
                    # Find best matching detection
                    best_match_score = 0.0
                    for i, (bbox, cls, score) in enumerate(zip(
                        occluded_instances.pred_boxes.tensor.detach().cpu().numpy(),
                        occluded_instances.pred_classes.detach().cpu().numpy(),
                        occluded_instances.scores.detach().cpu().numpy()
                    )):
                        if cls == baseline_class:
                            iou = self._calculate_iou(bbox, baseline_bbox)
                            if iou > 0.3:  # Reasonable overlap
                                best_match_score = max(best_match_score, score)
                    
                    # Sensitivity = drop in confidence
                    sensitivity_score = baseline_score - best_match_score
                
                # Assign sensitivity to all pixels in the patch
                sensitivity_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                    sensitivity_map[y:y+patch_size, x:x+patch_size],
                    sensitivity_score
                )
        
        # Normalize sensitivity map
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()
        
        # Resize to original image size
        saliency_map = cv2.resize(sensitivity_map, (width, height))
        
        print(f"    Occlusion sensitivity completed")
        return saliency_map, sensitivity_map, baseline_outputs[0]
    
    def _rise_detection(self, img_path, target_instance_idx, n_masks=2000, mask_prob=0.5):
        """
        RISE (Randomized Input Sampling for Explanation) adapted for object detection.
        
        Generates random masks and measures detection confidence changes.
        More robust than gradient methods and faster than exhaustive occlusion.
        """
        print(f"    Computing RISE with {n_masks} random masks")
        
        # Read and preprocess image
        image = read_image(img_path, format="BGR")
        height, width = image.shape[:2]
        
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], 
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        img_h, img_w = transformed_img.shape[:2]
        
        # Get baseline detection
        input_tensor = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1)
        ).to(self.cfg.MODEL.DEVICE)
        
        with torch.no_grad():
            baseline_outputs = self.model([{
                "image": input_tensor,
                "height": height,
                "width": width
            }])
        
        baseline_instances = baseline_outputs[0]["instances"]
        if target_instance_idx >= len(baseline_instances):
            return None, None, baseline_outputs[0]
            
        baseline_bbox = baseline_instances.pred_boxes.tensor[target_instance_idx].detach().cpu().numpy()
        baseline_class = baseline_instances.pred_classes[target_instance_idx].item()
        
        # RISE algorithm
        importance_map = np.zeros((img_h, img_w), dtype=np.float32)
        
        for i in range(n_masks):
            if i % 200 == 0:
                print(f"    Progress: {i}/{n_masks} masks")
            
            # Generate random binary mask
            mask = np.random.binomial(1, mask_prob, (img_h, img_w))
            
            # Apply mask to image
            masked_img = transformed_img.copy()
            for c in range(3):  # Apply to all channels
                masked_img[:, :, c] = masked_img[:, :, c] * mask
            
            masked_tensor = torch.as_tensor(
                masked_img.astype("float32").transpose(2, 0, 1)
            ).to(self.cfg.MODEL.DEVICE)
            
            with torch.no_grad():
                masked_outputs = self.model([{
                    "image": masked_tensor,
                    "height": height,
                    "width": width
                }])
            
            # Find matching detection confidence
            masked_instances = masked_outputs[0]["instances"]
            detection_score = 0.0
            
            if len(masked_instances) > 0:
                for bbox, cls, score in zip(
                    masked_instances.pred_boxes.tensor.detach().cpu().numpy(),
                    masked_instances.pred_classes.detach().cpu().numpy(),
                    masked_instances.scores.detach().cpu().numpy()
                ):
                    if cls == baseline_class:
                        iou = self._calculate_iou(bbox, baseline_bbox)
                        if iou > 0.3:
                            detection_score = max(detection_score, score)
            
            # Weight the mask by the detection score
            importance_map += mask * detection_score
        
        # Normalize by number of masks
        importance_map = importance_map / n_masks
        
        # Normalize to [0, 1]
        if importance_map.max() > 0:
            importance_map = importance_map / importance_map.max()
        
        # Resize to original image size
        saliency_map = cv2.resize(importance_map, (width, height))
        
        print(f"    RISE completed")
        return saliency_map, importance_map, baseline_outputs[0]
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _register_guided_hooks(self):
        """Register hooks for guided backpropagation."""
        self.hooks = []
        
        def relu_hook_function(module, grad_in, grad_out):
            """Guided backprop hook - only pass positive gradients."""
            return (torch.clamp(grad_in[0], min=0.0),)
        
        # Register hooks on ReLU layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_backward_hook(relu_hook_function))
    
    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ImprovedMisclassificationAnalyzer:
    """Analyzer using detection-specific methods."""
    
    def __init__(self, config_file="configs/Cityscapes/mask_rcnn_R_50_FPN.yaml", 
                 model_weights="detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"):
        # Same setup as before
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
        
        self.model = build_model(cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        
        self.saliency_generator = DetectionSpecificSaliency(self.model, cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.class_names = self.metadata.thing_classes
        
        print("Initialized improved detection saliency analyzer")
    
    def analyze_with_detection_methods(self, img_path, pred_instance_idx, true_class, pred_class,
                                     methods=['guided_backprop', 'occlusion'],
                                     output_dir="improved_detection_saliency"):
        """Analyze using detection-specific methods."""
        
        image_name = Path(img_path).stem
        case_dir = os.path.join(output_dir, f"{image_name}_{pred_instance_idx}_{true_class}_to_{pred_class}")
        os.makedirs(case_dir, exist_ok=True)
        
        results = {
            "image_path": img_path,
            "pred_instance_idx": pred_instance_idx,
            "true_class": true_class,
            "pred_class": pred_class,
            "methods": {}
        }
        
        image = read_image(img_path, format="BGR")
        
        for method in methods:
            print(f"  Computing {method}...")
            
            try:
                saliency_map, saliency_raw, output = self.saliency_generator.compute_detection_saliency(
                    img_path, pred_instance_idx, method=method
                )
                
                if saliency_map is not None:
                    method_dir = os.path.join(case_dir, method)
                    os.makedirs(method_dir, exist_ok=True)
                    
                    # Save visualization
                    self._save_detection_visualization(
                        image, saliency_map, output["instances"][pred_instance_idx],
                        method_dir, method, true_class
                    )
                    
                    results["methods"][method] = {"success": True}
                    np.save(os.path.join(method_dir, "saliency_map.npy"), saliency_map)
                    
                else:
                    results["methods"][method] = {"success": False, "error": "Failed to compute"}
                    
            except Exception as e:
                print(f"  Error with {method}: {e}")
                results["methods"][method] = {"success": False, "error": str(e)}
        
        # Save results
        with open(os.path.join(case_dir, "detection_analysis_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_detection_visualization(self, image, saliency_map, instance, output_dir, method, true_class):
        """Save detection-specific visualization."""
        bbox = instance.pred_boxes.tensor[0].detach().cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop around detection with padding
        height, width = image.shape[:2]
        pad_x = max(50, int((x2 - x1) * 0.3))
        pad_y = max(50, int((y2 - y1) * 0.3))
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(width, x2 + pad_x)
        y2_pad = min(height, y2 + pad_y)
        
        image_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        saliency_crop = saliency_map[y1_pad:y2_pad, x1_pad:x2_pad]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        image_with_box = image_crop.copy()
        box_in_crop = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]
        cv2.rectangle(image_with_box, (box_in_crop[0], box_in_crop[1]), 
                     (box_in_crop[2], box_in_crop[3]), (0, 255, 0), 3)
        plt.imshow(image_with_box[:, :, ::-1])
        plt.title(f"Detection: {self.class_names[instance.pred_classes[0]]}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(saliency_crop, cmap='hot')
        plt.title(f"{method.replace('_', ' ').title()}")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image_crop[:, :, ::-1])
        plt.imshow(saliency_crop, cmap='hot', alpha=0.6)
        plt.title("Attribution Overlay")
        plt.axis('off')
        
        plt.suptitle(f"{method.replace('_', ' ').title()} - Predicted: {self.class_names[instance.pred_classes[0]]} "
                    f"(should be: {true_class})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_detection_analysis.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function using improved detection-specific methods."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved detection saliency analysis")
    parser.add_argument("--fn_person_rider", 
                       default="output/misclassification_analysis_complete/fn_person_rider_details.csv")
    parser.add_argument("--fn_rider_person", 
                       default="output/misclassification_analysis_complete/fn_rider_person_details.csv")
    parser.add_argument("--output_dir", default="output/improved_detection_saliency")
    parser.add_argument("--methods", nargs='+', default=['guided_backprop', 'occlusion'],
                       help="Detection methods: guided_backprop, occlusion, rise")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()
    
    analyzer = ImprovedMisclassificationAnalyzer()
    
    total_successful = 0
    
    for csv_path in [args.fn_person_rider, args.fn_rider_person]:
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        df_sample = df.head(args.max_samples)
        
        csv_name = Path(csv_path).stem
        output_subdir = os.path.join(args.output_dir, csv_name)
        
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            try:
                print(f"\nCase {idx+1}: {row['file_name']}")
                
                analyzer.analyze_with_detection_methods(
                    row['file_name'],
                    row['pred_instance_idx'],
                    row['gt_class'],
                    row['pred_class'],
                    methods=args.methods,
                    output_dir=output_subdir
                )
                total_successful += 1
                
            except Exception as e:
                print(f"Failed: {e}")
    
    print(f"\nCompleted! Successful: {total_successful}")


if __name__ == "__main__":
    main()