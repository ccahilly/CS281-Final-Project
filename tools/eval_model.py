#!/usr/bin/env python

import logging
import os
import numpy as np
from collections import OrderedDict
import cv2

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import CityscapesInstanceEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

class DetailedCityscapesEvaluator(CityscapesInstanceEvaluator):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        
        self.classes = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        self.num_classes = len(self.classes)
        
        # Map Cityscapes training IDs to our class indices
        self.trainId_to_class = {
            24: 0,  # person
            25: 1,  # rider
            26: 2,  # car
            27: 3,  # truck
            28: 4,  # bus
            31: 5,  # train
            32: 6,  # motorcycle
            33: 7,  # bicycle
        }
        
        # Initialize metrics for each class
        self.metrics = {cls: {
            'TP': 0,  # True Positives: correct class and IoU > threshold
            'FP': 0,  # False Positives: predicted this class but wrong
            'FN': 0,  # False Negatives: missed instances of this class
            'total_gt': 0,  # Total ground truth instances
            'total_pred': 0  # Total predictions
        } for cls in self.classes}
        
        self.iou_threshold = 0.5
        self._logger = logging.getLogger(__name__)
        self._predictions = []
        
        # Create output directory for CSV
        self._output_dir = os.path.join(os.path.dirname(dataset_name), "evaluation_output")
        os.makedirs(self._output_dir, exist_ok=True)

    def process(self, inputs, outputs):
        """Process predictions from model."""
        for input, output in zip(inputs, outputs):
            pred_instances = output["instances"].to(self._cpu_device)
            
            # Get ground truth instances from the instance segmentation file
            img_path = input["file_name"]
            if "leftImg8bit" not in img_path:
                self._logger.warning(f"Unexpected image path format: {img_path}")
                continue
                
            # Convert path from leftImg8bit to gtFine and change extension
            gt_path = img_path.replace("/leftImg8bit/", "/gtFine/")
            gt_path = gt_path.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
            
            if not os.path.isfile(gt_path):
                self._logger.warning(f"Instance segmentation file not found: {gt_path}")
                continue
                
            # Load instance segmentation
            gt_seg = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_seg is None:
                self._logger.warning(f"Could not read instance segmentation file: {gt_path}")
                continue
                
            # Extract instances
            gt_instances = []
            gt_classes = []
            gt_boxes = []
            
            # Process each instance ID
            instance_ids = np.unique(gt_seg)[1:]  # Skip background (0)
            for instance_id in instance_ids:
                # Get instance mask and class ID
                instance_mask = (gt_seg == instance_id)
                class_id = instance_id // 1000
                
                if class_id not in self.trainId_to_class:
                    continue
                    
                # Get bounding box
                y_indices, x_indices = np.where(instance_mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                gt_instances.append(instance_mask)
                gt_classes.append(self.trainId_to_class[class_id])
                gt_boxes.append([x_min, y_min, x_max, y_max])
            
            if not gt_instances:
                self._logger.warning(f"No valid ground truth instances found in {gt_path}")
                continue
            
            # Store for evaluation
            self._predictions.append({
                "file_name": input["file_name"],
                "instances": pred_instances,
                "gt_boxes": np.array(gt_boxes),
                "gt_classes": np.array(gt_classes)
            })

    def evaluate(self):
        """Evaluate predictions and compute metrics."""
        if not self._predictions:
            self._logger.warning("No predictions to evaluate!")
            return None
            
        # Reset metrics
        for metrics in self.metrics.values():
            metrics.update({'TP': 0, 'FP': 0, 'FN': 0, 'total_gt': 0, 'total_pred': 0})
        
        # Set ground truth totals from dataset statistics
        gt_counts = {
            'person': 14,
            'rider': 1,
            'car': 47,
            'truck': 1,
            'bus': 1,
            'train': 0,
            'motorcycle': 0,
            'bicycle': 2
        }
        for class_name, count in gt_counts.items():
            self.metrics[class_name]['total_gt'] = count
        
        # Process each prediction
        for pred in self._predictions:
            pred_instances = pred["instances"]
            gt_boxes = pred["gt_boxes"]
            gt_classes = pred["gt_classes"]
            
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            
            # Update prediction counts
            for pred_class in pred_classes:
                self.metrics[self.classes[pred_class]]['total_pred'] += 1
            
            # Match predictions to ground truth
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ious = self._compute_iou_matrix(gt_boxes, pred_boxes)
                gt_matched = np.zeros(len(gt_classes), dtype=bool)
                pred_matched = np.zeros(len(pred_classes), dtype=bool)
                
                # For each ground truth instance, find best matching prediction
                for gt_idx, gt_class in enumerate(gt_classes):
                    gt_class_name = self.classes[gt_class]
                    best_iou = 0
                    best_pred_idx = -1
                    
                    for pred_idx, pred_class in enumerate(pred_classes):
                        if pred_matched[pred_idx]:
                            continue
                        if ious[gt_idx, pred_idx] > best_iou:
                            best_iou = ious[gt_idx, pred_idx]
                            best_pred_idx = pred_idx
                    
                    if best_iou > self.iou_threshold and best_pred_idx >= 0:
                        pred_class = pred_classes[best_pred_idx]
                        gt_matched[gt_idx] = True
                        pred_matched[best_pred_idx] = True
                        
                        if pred_class == gt_class:
                            self.metrics[gt_class_name]['TP'] += 1
                        else:
                            self.metrics[self.classes[pred_class]]['FP'] += 1
                
                # Unmatched predictions are False Positives
                for pred_idx, pred_class in enumerate(pred_classes):
                    if not pred_matched[pred_idx]:
                        self.metrics[self.classes[pred_class]]['FP'] += 1
        
        # Compute final metrics
        results = OrderedDict()
        
        # Save detailed metrics as CSV
        metrics_file = os.path.join(self._output_dir, "detailed_metrics.csv")
        with open(metrics_file, "w") as f:
            f.write("class,total_gt,total_pred,TP,FP,FN,precision,recall,f1\n")
            
            for class_name, metrics in self.metrics.items():
                tp = metrics['TP']
                fp = metrics['FP']
                total_gt = metrics['total_gt']
                total_pred = metrics['total_pred']
                
                # False Negatives = total ground truth - true positives
                fn = total_gt - tp
                
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / total_gt if total_gt > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                # Write to CSV
                f.write(f"{class_name},{total_gt},{total_pred},{tp},{fp},{fn},{precision:.4f},{recall:.4f},{f1:.4f}\n")
                
                # Store in results
                results[f"{class_name}/total_gt"] = total_gt
                results[f"{class_name}/total_pred"] = total_pred
                results[f"{class_name}/TP"] = tp
                results[f"{class_name}/FP"] = fp
                results[f"{class_name}/FN"] = fn
                results[f"{class_name}/precision"] = precision * 100
                results[f"{class_name}/recall"] = recall * 100
                results[f"{class_name}/f1"] = f1 * 100
        
        self._logger.info(f"Detailed metrics saved to: {metrics_file}")
        return results

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between all pairs of boxes between boxes1 and boxes2.
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = np.clip(rb - lt, 0, None)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        return iou

def setup_cfg(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Model Evaluation with Detailed Analysis")
    parser.add_argument(
        "--config-file",
        default="configs/Cityscapes/mask_rcnn_R_50_FPN.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="perform evaluation only",
    )
    parser.add_argument(
        "--dataset",
        help="name of the dataset to evaluate on",
        default="cityscapes_fine_instance_seg_val",
    )
    parser.add_argument(
        "--weights",
        default="detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl",
        help="path to the model weights",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def main(args):
    cfg = setup_cfg(args)
    
    # Set up logger
    logger = setup_logger()
    logger.info("Command Line Args: %s", args)
    
    # Create model
    model = DefaultPredictor(cfg)
    
    # Create data loader
    data_loader = build_detection_test_loader(cfg, args.dataset)
    
    # Create evaluator
    evaluator = DetailedCityscapesEvaluator(dataset_name=args.dataset)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = inference_on_dataset(model.model, data_loader, evaluator)
    
    # Print results
    logger.info("Evaluation Results:")
    for k, v in results.items():
        logger.info(f"{k}: {v}")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args) 