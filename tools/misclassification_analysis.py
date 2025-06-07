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
        
        # Initialize detailed misclassification info
        self.fn_person_rider_details = []  # person classified as rider
        self.fn_rider_person_details = []  # rider classified as person
        
        self.iou_threshold = 0.5
        self._logger = logging.getLogger(__name__)
        self._predictions = []  # Initialize empty predictions list
        
        # Create output directory
        self._output_dir = os.path.join(os.path.dirname(dataset_name), "output/misclassification_analysis")
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
            gt_instance_ids = []  # Store instance IDs
            
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
                gt_instance_ids.append(instance_id)
            
            if not gt_instances:
                continue
            
            # Get predictions
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            
            # Compute IoU matrix between ground truth and predictions
            ious = self._compute_iou_matrix(np.array(gt_boxes), pred_boxes)
            
            # Track which predictions have been matched to a ground truth
            matched_preds = set()
            
            # First pass: For each ground truth, find its best matching prediction with correct class
            for gt_idx, (gt_class, gt_instance_id) in enumerate(zip(gt_classes, gt_instance_ids)):
                gt_class_name = self.classes[gt_class]
                if gt_class_name not in CLASSES_OF_INTEREST:
                    continue
                
                # Get ground truth bounding box
                gt_box = gt_boxes[gt_idx]
                
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
                
                if best_iou >= self.iou_threshold:
                    # Found a match - it's a true positive
                    matched_preds.add(best_pred_idx)
                    self.tp_counts[gt_class_name] += 1
                else:
                    # GT doesn't match with correct class above threshold - false negative
                    # Find the highest IoU prediction (if any) to record what it was misclassified as
                    best_iou = 0
                    best_pred_class = "none"
                    best_pred_idx = -1
                    best_pred_box = None
                    
                    for pred_idx, pred_class in enumerate(pred_classes):
                        # Skip predictions that were already matched as true positives
                        if pred_idx in matched_preds:
                            continue
                            
                        iou = ious[gt_idx, pred_idx]
                        if iou > best_iou and iou >= self.iou_threshold:
                            best_iou = iou
                            best_pred_class = self.classes[pred_class]
                            best_pred_idx = pred_idx
                            best_pred_box = pred_boxes[pred_idx] if pred_idx >= 0 else None
                    
                    # Record the false negative
                    self.fn_matrix[gt_class_name][best_pred_class] += 1
                    
                    # Store detailed information for person-rider misclassifications
                    if gt_class_name == "person" and best_pred_class == "rider":
                        detail = {
                            'file_name': img_path,
                            'gt_instance_id': gt_instance_id,
                            'gt_class': gt_class_name,
                            'gt_box': gt_box,  # Add ground truth box
                            'pred_instance_idx': best_pred_idx,
                            'pred_class': best_pred_class,
                            'pred_box': best_pred_box.tolist() if best_pred_box is not None else None,  # Add prediction box
                            'iou': best_iou
                        }
                        self.fn_person_rider_details.append(detail)
                    elif gt_class_name == "rider" and best_pred_class == "person":
                        detail = {
                            'file_name': img_path,
                            'gt_instance_id': gt_instance_id,
                            'gt_class': gt_class_name,
                            'gt_box': gt_box,  # Add ground truth box
                            'pred_instance_idx': best_pred_idx,
                            'pred_class': best_pred_class,
                            'pred_box': best_pred_box.tolist() if best_pred_box is not None else None,  # Add prediction box
                            'iou': best_iou
                        }
                        self.fn_rider_person_details.append(detail)
            
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
        
        # Save detailed person-rider misclassification files
        person_rider_details_file = os.path.join(self._output_dir, "fn_person_rider_details.csv")
        rider_person_details_file = os.path.join(self._output_dir, "fn_rider_person_details.csv")
        
        # Write detailed person-rider misclassification files with bounding box information
        with open(person_rider_details_file, "w") as f:
            f.write("file_name,gt_instance_id,gt_class,gt_box_x_min,gt_box_y_min,gt_box_x_max,gt_box_y_max,pred_instance_idx,pred_class,pred_box_x_min,pred_box_y_min,pred_box_x_max,pred_box_y_max,iou\n")
            for detail in self.fn_person_rider_details:
                gt_box = detail['gt_box']
                pred_box = detail['pred_box']
                
                # Format ground truth box
                gt_box_str = f"{gt_box[0]},{gt_box[1]},{gt_box[2]},{gt_box[3]}"
                
                # Format prediction box, handle None case
                if pred_box is not None:
                    pred_box_str = f"{pred_box[0]},{pred_box[1]},{pred_box[2]},{pred_box[3]}"
                else:
                    pred_box_str = ",,,"  # Empty values for missing box
                
                f.write(f"{detail['file_name']},{detail['gt_instance_id']},{detail['gt_class']},{gt_box_str},{detail['pred_instance_idx']},{detail['pred_class']},{pred_box_str},{detail['iou']:.3f}\n")
                
        with open(rider_person_details_file, "w") as f:
            f.write("file_name,gt_instance_id,gt_class,gt_box_x_min,gt_box_y_min,gt_box_x_max,gt_box_y_max,pred_instance_idx,pred_class,pred_box_x_min,pred_box_y_min,pred_box_x_max,pred_box_y_max,iou\n")
            for detail in self.fn_rider_person_details:
                gt_box = detail['gt_box']
                pred_box = detail['pred_box']
                
                # Format ground truth box
                gt_box_str = f"{gt_box[0]},{gt_box[1]},{gt_box[2]},{gt_box[3]}"
                
                # Format prediction box, handle None case
                if pred_box is not None:
                    pred_box_str = f"{pred_box[0]},{pred_box[1]},{pred_box[2]},{pred_box[3]}"
                else:
                    pred_box_str = ",,,"  # Empty values for missing box
                
                f.write(f"{detail['file_name']},{detail['gt_instance_id']},{detail['gt_class']},{gt_box_str},{detail['pred_instance_idx']},{detail['pred_class']},{pred_box_str},{detail['iou']:.3f}\n")
        
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