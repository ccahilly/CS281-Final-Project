#!/usr/bin/env python

import logging
import os
import numpy as np
from collections import defaultdict, OrderedDict
import torch
import json
from typing import Dict, List, Optional, Tuple
import cv2

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
)
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetMapper

class Instance:
    def __init__(self, imgNp, instID):
        if instID == -1:
            return
        self.instID = int(instID)
        self.labelID = self.getLabelID(instID)
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))
        self.mask = (imgNp == instID)
        
        # Get bounding box
        y_indices, x_indices = np.where(self.mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            self.bbox = [
                float(np.min(x_indices)),
                float(np.min(y_indices)),
                float(np.max(x_indices)),
                float(np.max(y_indices))
            ]
        else:
            self.bbox = None

    def getLabelID(self, instID):
        if instID < 1000:
            return instID
        else:
            return int(instID / 1000)

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

class DetailedCityscapesEvaluator(CityscapesInstanceEvaluator):
    """
    Evaluator that tracks detailed misclassification statistics for Cityscapes instance segmentation.
    """
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
        
        # IoU threshold for matching
        self.iou_threshold = 0.5
        
        self._logger = logging.getLogger(__name__)
        self._predictions = []
        
        # Create output directory
        self._output_dir = os.path.join(os.path.dirname(dataset_name), "evaluation_output")
        os.makedirs(self._output_dir, exist_ok=True)

    def process(self, inputs, outputs):
        """
        Process one batch of inputs and outputs.
        Args:
            inputs: List[dict], each dict has "image_id", "file_name", "height", "width"
            outputs: List[dict], each dict has "instances" field
        """
        for input, output in zip(inputs, outputs):
            pred_instances = output["instances"].to(self._cpu_device)
            
            # Get ground truth instances from the instance segmentation file
            img_path = input["file_name"]
            if "leftImg8bit" not in img_path:
                self._logger.warning(f"Unexpected image path format: {img_path}")
                continue
                
            # Example input path: datasets/cityscapes/leftImg8bit/val/munster/munster_000000_000019_leftImg8bit.png
            # Example gt path: datasets/cityscapes/gtFine/val/munster/munster_000000_000019_gtFine_instanceIds.png
            gt_path = img_path.replace("/leftImg8bit/", "/gtFine/")
            gt_path = gt_path.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
            
            self._logger.info(f"Looking for ground truth at: {gt_path}")
            
            if not os.path.isfile(gt_path):
                self._logger.warning(f"Instance segmentation file not found: {gt_path}")
                # Try to list the directory contents to debug
                gt_dir = os.path.dirname(gt_path)
                if os.path.exists(gt_dir):
                    self._logger.info(f"Contents of {gt_dir}:")
                    for f in os.listdir(gt_dir):
                        self._logger.info(f"  {f}")
                else:
                    self._logger.warning(f"Directory does not exist: {gt_dir}")
                continue
                
            # Load instance segmentation
            gt_seg = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_seg is None:
                self._logger.warning(f"Could not read instance segmentation file: {gt_path}")
                continue
                
            # Extract instances using Cityscapes instance handling
            gt_instances = []
            instance_ids = np.unique(gt_seg)[1:]  # Skip background (0)
            
            for instance_id in instance_ids:
                instance = Instance(gt_seg, instance_id)
                if instance.labelID in self.trainId_to_class and instance.bbox is not None:
                    gt_instances.append(instance)
            
            if not gt_instances:
                self._logger.warning(f"No valid ground truth instances found in {gt_path}")
                continue
            
            # Convert instances to format needed for evaluation
            gt_boxes = np.array([inst.bbox for inst in gt_instances])
            gt_classes = np.array([self.trainId_to_class[inst.labelID] for inst in gt_instances])
            gt_masks = [inst.mask for inst in gt_instances]
            
            self._logger.info(f"Found {len(gt_instances)} ground truth instances:")
            for i, inst in enumerate(gt_instances):
                self._logger.info(f"  Instance {i}: class={self.classes[self.trainId_to_class[inst.labelID]]}, pixels={inst.pixelCount}")
            
            # Store for later analysis
            self._predictions.append({
                "image_id": input["image_id"],
                "file_name": input["file_name"],
                "instances": pred_instances,
                "gt_boxes": gt_boxes,
                "gt_classes": gt_classes,
                "gt_masks": gt_masks
            })
            
            self._logger.info(
                f"Processed {input['file_name']}: "
                f"{len(pred_instances)} predictions, {len(gt_instances)} ground truth instances"
            )

    def evaluate(self):
        """
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self._predictions:
            self._logger.warning("No predictions to evaluate!")
            return None
            
        # Reset metrics
        for metrics in self.metrics.values():
            metrics.update({'TP': 0, 'FP': 0, 'FN': 0, 'total_gt': 0, 'total_pred': 0})
        
        # Set the ground truth totals from the dataset statistics
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
        
        # Analyze each prediction
        for pred in self._predictions:
            pred_instances = pred["instances"]
            gt_boxes = pred["gt_boxes"]
            gt_classes = pred["gt_classes"]
            
            # Get prediction details
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            pred_scores = pred_instances.scores.numpy()
            
            self._logger.info(f"\nProcessing {pred['file_name']}:")
            self._logger.info(f"  Ground truth: {len(gt_classes)} instances")
            self._logger.info(f"  Predictions: {len(pred_instances)} instances")
            
            # Update total prediction counts
            for pred_class in pred_classes:
                self.metrics[self.classes[pred_class]]['total_pred'] += 1
            
            # Track which GT instances have been matched
            gt_matched = np.zeros(len(gt_classes), dtype=bool)
            pred_matched = np.zeros(len(pred_classes), dtype=bool)
            
            # Compute IoU between all pairs of boxes
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ious = self._compute_iou_matrix(gt_boxes, pred_boxes)
            else:
                ious = np.zeros((len(gt_boxes), len(pred_boxes)))
            
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
                    pred_class_name = self.classes[pred_class]
                    pred_score = pred_scores[best_pred_idx]
                    
                    gt_matched[gt_idx] = True
                    pred_matched[best_pred_idx] = True
                    
                    if pred_class == gt_class:
                        # True Positive
                        self.metrics[gt_class_name]['TP'] += 1
                        self._logger.info(f"  TP: {gt_class_name} correctly detected with IoU={best_iou:.2f}, score={pred_score:.2f}")
                    else:
                        # False Positive for predicted class (wrong class)
                        self.metrics[pred_class_name]['FP'] += 1
                        # False Negative for ground truth class (misclassified)
                        self.metrics[gt_class_name]['FN'] += 1
                        self._logger.info(
                            f"  Misclassification: GT {gt_class_name} detected as {pred_class_name} "
                            f"with IoU={best_iou:.2f}, score={pred_score:.2f}"
                        )
            
            # Unmatched predictions are False Positives
            for pred_idx, pred_class in enumerate(pred_classes):
                if not pred_matched[pred_idx]:
                    pred_class_name = self.classes[pred_class]
                    self.metrics[pred_class_name]['FP'] += 1
                    self._logger.info(f"  FP: Extra {pred_class_name} prediction with score={pred_scores[pred_idx]:.2f}")
        
        # Count unmatched GT instances as False Negatives
        # We do this at the end to ensure we don't double count FNs
        for class_name, metrics in self.metrics.items():
            total_gt = metrics['total_gt']
            tp = metrics['TP']
            # FN is any ground truth instance that wasn't a true positive
            metrics['FN'] = total_gt - tp
        
        # Print detailed metrics for each class
        self._logger.info("\nDetailed Metrics per Class:")
        header = f"{'Class':12} | {'Total GT':8} | {'Total Pred':10} | {'TP':6} | {'FP':6} | {'FN':6} | {'Precision':9} | {'Recall':9}"
        separator = "-" * len(header)
        self._logger.info(separator)
        self._logger.info(header)
        self._logger.info(separator)
        
        results = OrderedDict()
        for class_name, metrics in self.metrics.items():
            tp, fp, fn = metrics['TP'], metrics['FP'], metrics['FN']
            total_gt = metrics['total_gt']
            total_pred = metrics['total_pred']
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / total_gt if total_gt > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            
            self._logger.info(
                f"{class_name:12} | {total_gt:8d} | {total_pred:10d} | {tp:6d} | {fp:6d} | {fn:6d} | "
                f"{precision:8.2%} | {recall:8.2%}"
            )
            
            # Store metrics in results
            results[f"{class_name}/total_gt"] = total_gt
            results[f"{class_name}/total_pred"] = total_pred
            results[f"{class_name}/TP"] = tp
            results[f"{class_name}/FP"] = fp
            results[f"{class_name}/FN"] = fn
            results[f"{class_name}/precision"] = precision * 100
            results[f"{class_name}/recall"] = recall * 100
            results[f"{class_name}/f1"] = f1 * 100
        
        self._logger.info(separator)
        
        # Save detailed metrics as CSV
        metrics_file = os.path.join(self._output_dir, "detailed_metrics.csv")
        with open(metrics_file, "w") as f:
            f.write("class,total_gt,total_pred,TP,FP,FN,precision,recall,f1\n")
            for class_name, metrics in self.metrics.items():
                tp, fp, fn = metrics['TP'], metrics['FP'], metrics['FN']
                total_gt = metrics['total_gt']
                total_pred = metrics['total_pred']
                
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / total_gt if total_gt > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                f.write(f"{class_name},{total_gt},{total_pred},{tp},{fp},{fn},{precision:.4f},{recall:.4f},{f1:.4f}\n")
        
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
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Set score threshold for visualizing predictions
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    
    # Set evaluation mode
    cfg.DATASETS.TEST = (args.dataset, )
    
    # Set model weights
    cfg.MODEL.WEIGHTS = args.weights
        
    # Force CPU
    cfg.MODEL.DEVICE = "cpu"
    
    # Make sure we're doing instance segmentation
    cfg.MODEL.MASK_ON = True
    
    # Set NMS threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # Ensure we're using the right number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # person, rider, car, truck, bus, train, motorcycle, bicycle
    
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
        choices=["cityscapes_fine_instance_seg_val", "cityscapes_fine_instance_seg_train", "cityscapes_fine_instance_seg_test"]
    )
    parser.add_argument(
        "--weights",
        default="detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl",
        help="path to the model weights",
    )
    parser.add_argument(
        "--output",
        help="output directory for visualization and analysis files",
        default="evaluation_output"
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
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(output=args.output)
    logger.info("Command Line Args: %s", args)
    
    # Create model
    model = DefaultPredictor(cfg)
    
    # Create data loader
    data_loader = build_detection_test_loader(cfg, args.dataset)
    
    # Create evaluator
    evaluator = DetailedCityscapesEvaluator(
        dataset_name=args.dataset
    )
    
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