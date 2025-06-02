#!/usr/bin/env python

import logging
import os
import numpy as np
from collections import OrderedDict
import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import CityscapesInstanceEvaluator, verify_results
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

# Constants for evaluation configuration
CONFIG_FILE = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
CONFIDENCE_THRESHOLD = 0.5

# Classes of interest for misclassification analysis
CLASSES_OF_INTEREST = ["person", "rider"]
POSSIBLE_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

class MisclassificationAnalyzer(CityscapesInstanceEvaluator):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.metadata = MetadataCatalog.get(dataset_name)
        self.classes = self.metadata.thing_classes
        self.num_classes = len(self.classes)
        
        # Map Cityscapes training IDs to class indices
        from cityscapesscripts.helpers.labels import labels
        self.trainId_to_class = {}
        thing_count = 0
        for label in labels:
            if label.hasInstances and not label.ignoreInEval:
                self.trainId_to_class[label.id] = thing_count
                if hasattr(label, 'trainId') and label.trainId >= 0:
                    self.trainId_to_class[label.trainId] = thing_count
                thing_count += 1
        
        # Initialize misclassification matrices
        self.fn_matrix = {
            cls: {pred_cls: 0 for pred_cls in POSSIBLE_CLASSES + ["none"]} 
            for cls in CLASSES_OF_INTEREST
        }
        self.fp_matrix = {
            cls: {gt_cls: 0 for gt_cls in POSSIBLE_CLASSES + ["none"]} 
            for cls in CLASSES_OF_INTEREST
        }
        # Initialize true positive counts
        self.tp_counts = {cls: 0 for cls in CLASSES_OF_INTEREST}
        
        self.iou_threshold = 0.5
        self._logger = logging.getLogger(__name__)
        self._predictions = []  # Initialize empty predictions list
        
        # Create output directory
        self._output_dir = os.path.join(os.path.dirname(dataset_name), "misclassification_analysis")
        os.makedirs(self._output_dir, exist_ok=True)

    def process(self, inputs, outputs):
        # Store predictions for later evaluation
        self._predictions.extend(outputs)
        
        for input, output in zip(inputs, outputs):
            pred_instances = output["instances"].to(self._cpu_device)
            
            # Get ground truth path
            img_path = input["file_name"]
            gt_path = img_path.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
            
            if not os.path.isfile(gt_path):
                self._logger.warning(f"Instance segmentation file not found: {gt_path}")
                continue
                
            # Load ground truth image
            gt_seg = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_seg is None:
                self._logger.warning(f"Could not read ground truth file: {gt_path}")
                continue
            
            # Extract ground truth instances
            gt_instances = []
            gt_classes = []
            gt_boxes = []
            
            for instance_id in np.unique(gt_seg)[1:]:  # Skip background (0)
                instance_mask = (gt_seg == instance_id)
                class_id = instance_id // 1000
                
                if class_id not in self.trainId_to_class:
                    continue
                    
                y_indices, x_indices = np.where(instance_mask)
                if len(y_indices) == 0:
                    continue
                    
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                gt_instances.append(instance_mask)
                gt_classes.append(self.trainId_to_class[class_id])
                gt_boxes.append([x_min, y_min, x_max, y_max])
            
            if not gt_instances:
                continue
            
            # Get predictions
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            pred_scores = pred_instances.scores.numpy()
            
            # Compute IoU matrix between ground truth and predictions
            ious = self._compute_iou_matrix(np.array(gt_boxes), pred_boxes)
            
            # Track which predictions have been matched to a ground truth
            matched_preds = set()
            
            # First pass: For each ground truth, find its best matching prediction with correct class
            for gt_idx, gt_class in enumerate(gt_classes):
                gt_class_name = self.classes[gt_class]
                if gt_class_name not in CLASSES_OF_INTEREST:
                    continue
                
                # Find best matching prediction with correct class
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, pred_class in enumerate(pred_classes):
                    if pred_idx in matched_preds:  # Skip already matched predictions
                        continue
                    
                    # Only consider predictions with matching class
                    if pred_class != gt_class:
                        continue
                        
                    iou = ious[gt_idx, pred_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                # Store the best IoU from TP phase for debugging
                best_iou_tp = best_iou
                
                if best_iou >= self.iou_threshold:
                    # Found a match - it's a true positive
                    matched_preds.add(best_pred_idx)
                    self.tp_counts[gt_class_name] += 1
                else:
                    # GT doesn't match with correct class above threshold - false negative
                    # Find the highest IoU prediction (if any) to record what it was misclassified as
                    best_iou = 0
                    best_pred_class = "none"
                    
                    for pred_idx, pred_class in enumerate(pred_classes):
                        # Skip predictions that were already matched as true positives
                        if pred_idx in matched_preds:
                            continue
                            
                        iou = ious[gt_idx, pred_idx]
                        if iou > best_iou and iou >= self.iou_threshold:
                            best_iou = iou
                            best_pred_class = self.classes[pred_class]
                            if gt_class == pred_class:
                                self._logger.warning(f"ALERT: Same class prediction found in misclassification phase! GT class {gt_class_name} Pred class {best_pred_class} IoU {iou:.3f}")
                                self._logger.warning(f"Previous best IoU for this GT in TP phase was: {best_iou_tp:.3f}")
                                # Include the file name and the image in the warning
                                self._logger.warning(f"Image: {img_path}")
                    
                    # Record the false negative
                    # If no prediction has IoU >= threshold, count it as "none" regardless of class
                    self.fn_matrix[gt_class_name][best_pred_class] += 1
                    if gt_class_name == best_pred_class:
                        self._logger.warning(f"Recording fn_{gt_class_name}_{best_pred_class} - this should never happen!")
            
            # Second pass: Any unmatched predictions are false positives
            for pred_idx, pred_class in enumerate(pred_classes):
                if pred_idx in matched_preds:
                    continue
                    
                pred_class_name = self.classes[pred_class]
                if pred_class_name not in CLASSES_OF_INTEREST:
                    continue
                
                # Find if this prediction overlaps with any ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_class in enumerate(gt_classes):
                    iou = ious[gt_idx, pred_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.iou_threshold:
                    # False positive that overlaps with a ground truth
                    gt_class_name = self.classes[gt_classes[best_gt_idx]]
                    self.fp_matrix[pred_class_name][gt_class_name] += 1
                else:
                    # False positive with no significant overlap
                    self.fp_matrix[pred_class_name]["none"] += 1

    def evaluate(self):
        if not self._predictions:
            self._logger.warning("No predictions to evaluate!")
            return None
            
        results = OrderedDict()
        
        # Save misclassification analysis to CSV files
        fn_file = os.path.join(self._output_dir, "false_negatives.csv")
        fp_file = os.path.join(self._output_dir, "false_positives.csv")
        tp_file = os.path.join(self._output_dir, "true_positives.csv")
        
        # Write false negatives analysis
        with open(fn_file, "w") as f:
            # Write header
            f.write("gt_class,pred_class,count\n")
            
            # Write data
            for gt_class in CLASSES_OF_INTEREST:
                for pred_class, count in self.fn_matrix[gt_class].items():
                    f.write(f"{gt_class},{pred_class},{count}\n")
                    results[f"fn_{gt_class}_{pred_class}"] = count
        
        # Write false positives analysis
        with open(fp_file, "w") as f:
            # Write header
            f.write("pred_class,gt_class,count\n")
            
            # Write data
            for pred_class in CLASSES_OF_INTEREST:
                for gt_class, count in self.fp_matrix[pred_class].items():
                    f.write(f"{pred_class},{gt_class},{count}\n")
                    results[f"fp_{pred_class}_{gt_class}"] = count
        
        # Write true positives analysis
        with open(tp_file, "w") as f:
            # Write header
            f.write("class,count\n")
            
            # Write data
            for class_name, count in self.tp_counts.items():
                f.write(f"{class_name},{count}\n")
                results[f"tp_{class_name}"] = count
        
        self._logger.info(f"Misclassification analysis saved to: {self._output_dir}")
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

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
    cfg.freeze()
    return cfg

def main():
    logger = setup_logger()
    cfg = setup_cfg()
    
    # Build model and load weights
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    
    # Evaluate each dataset
    results = {}
    for dataset_name in cfg.DATASETS.TEST:
        logger.info(f"\nAnalyzing misclassifications for dataset: {dataset_name}")
        evaluator = MisclassificationAnalyzer(dataset_name=dataset_name)
        results[dataset_name] = DefaultTrainer.test(cfg, model, evaluators=[evaluator])
        
        logger.info(f"Results for {dataset_name}:")
        for k, v in results[dataset_name].items():
            logger.info(f"  {k}: {v}")
    
    if comm.is_main_process():
        verify_results(cfg, results)
    
    return results

if __name__ == "__main__":
    main() 