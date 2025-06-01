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
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as comm

# Constants for evaluation configuration
CONFIG_FILE = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
CONFIDENCE_THRESHOLD = 0.5  # Standard confidence threshold for Cityscapes

# Colors for visualization (in range [0, 1])
TP_COLOR = (0.0, 1.0, 0.0)  # Green
FP_COLOR = (1.0, 1.0, 0.0)  # Yellow
FN_COLOR = (1.0, 0.0, 0.0)  # Red

class DetailedCityscapesEvaluator(CityscapesInstanceEvaluator):
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
                thing_count += 1
        
        # Initialize metrics
        self.metrics = {cls: {
            'TP': 0, 'FP': 0, 'FN': 0,
            'total_gt': 0, 'total_pred': 0,
            'scores': [],  # Store prediction scores for AP calculation
            'matched': []  # Store whether each prediction matched a GT (True/False)
        } for cls in self.classes}
        
        self.iou_threshold = 0.5
        self._logger = logging.getLogger(__name__)
        self._predictions = []
        
        # Create output directories
        self._output_dir = os.path.join(os.path.dirname(dataset_name), "evaluation_output")
        self._images_dir = os.path.join(self._output_dir, "images")
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._images_dir, exist_ok=True)

    def _visualize_instance(self, image, box, mask, label, output_path, color=None):
        """Helper function to visualize a single instance."""
        v = Visualizer(image.copy(), metadata=self.metadata)
        if color is not None:
            v.draw_box(box, edge_color=color)
            if mask is not None:
                v.draw_binary_mask(mask, color=color, alpha=0.3)
            v.draw_text(label, (box[0], box[1]), color=color)
        else:
            v = v.overlay_instances(
                boxes=np.array([box]) if box is not None else None,
                masks=[mask] if mask is not None else None,
                labels=[label] if label is not None else None
            )
        instance_vis = v.get_output().get_image()
        cv2.imwrite(output_path, instance_vis[:, :, ::-1])

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred_instances = output["instances"].to(self._cpu_device)
            
            # Get ground truth path
            img_path = input["file_name"]
            gt_path = img_path.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
            
            if not os.path.isfile(gt_path):
                self._logger.warning(f"Instance segmentation file not found: {gt_path}")
                continue
                
            # Load images
            gt_seg = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            original_img = cv2.imread(img_path)
            if gt_seg is None or original_img is None:
                self._logger.warning(f"Could not read image files: {gt_path} or {img_path}")
                continue
            original_img = original_img[:, :, ::-1]  # BGR to RGB
            
            # Create output directories for this image
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_output_dir = os.path.join(self._images_dir, img_name)
            gt_instances_dir = os.path.join(img_output_dir, "gt")
            pred_instances_dir = os.path.join(img_output_dir, "pred")
            os.makedirs(img_output_dir, exist_ok=True)
            os.makedirs(gt_instances_dir, exist_ok=True)
            os.makedirs(pred_instances_dir, exist_ok=True)
            
            # Extract ground truth instances
            gt_instances, gt_classes, gt_boxes = [], [], []
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
            pred_masks = pred_instances.pred_masks.numpy() if pred_instances.has("pred_masks") else None
            
            # Evaluate instances
            eval_info = {
                'gt_classes': gt_classes,
                'gt_matched': np.zeros(len(gt_classes), dtype=bool),
                'pred_classes': pred_classes,
                'pred_matched': np.zeros(len(pred_classes), dtype=bool),
                'is_tp': np.zeros(len(pred_classes), dtype=bool),
                'pred_scores': pred_scores,
                'ious': self._compute_iou_matrix(np.array(gt_boxes), pred_boxes)
            }
            
            # Match instances
            for gt_idx, gt_class in enumerate(gt_classes):
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, pred_class in enumerate(pred_classes):
                    if pred_class == gt_class and not eval_info['pred_matched'][pred_idx]:
                        iou = eval_info['ious'][gt_idx, pred_idx]
                        if iou > self.iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_idx
                
                if best_pred_idx >= 0:
                    eval_info['gt_matched'][gt_idx] = True
                    eval_info['pred_matched'][best_pred_idx] = True
                    eval_info['is_tp'][best_pred_idx] = True
            
            # Store prediction scores and match status for AP calculation
            for pred_idx, (pred_class, is_tp, score) in enumerate(zip(pred_classes, eval_info['is_tp'], pred_scores)):
                class_name = self.classes[pred_class]
                self.metrics[class_name]['scores'].append(score)
                self.metrics[class_name]['matched'].append(is_tp)

            # Visualize ground truth instances
            for gt_idx, (mask, box, class_idx) in enumerate(zip(gt_instances, gt_boxes, gt_classes)):
                color = TP_COLOR if eval_info['gt_matched'][gt_idx] else FN_COLOR
                self._visualize_instance(
                    original_img, box, mask,
                    self.classes[class_idx],
                    os.path.join(gt_instances_dir, f"{self.classes[class_idx]}_{gt_idx + 1}.png"),
                    color=color
                )
            
            # Visualize predicted instances
            for pred_idx, (box, class_idx) in enumerate(zip(pred_boxes, pred_classes)):
                color = TP_COLOR if eval_info['is_tp'][pred_idx] else FP_COLOR
                mask = pred_masks[pred_idx] if pred_masks is not None else None
                best_iou = max(eval_info['ious'][:, pred_idx])
                
                self._visualize_instance(
                    original_img, box, mask,
                    f"{self.classes[class_idx]} ({best_iou:.2f})",
                    os.path.join(pred_instances_dir, f"{self.classes[class_idx]}_{pred_idx + 1}.png"),
                    color=color
                )
            
            self._predictions.append({"evaluation": eval_info})

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

    def _compute_ap(self, scores, matched):
        """Compute Average Precision at the confidence threshold."""
        if not scores:
            return 0.0
            
        # Sort by score in descending order
        scores = np.array(scores)
        matched = np.array(matched)
        
        # Only consider predictions above confidence threshold
        mask = scores >= CONFIDENCE_THRESHOLD
        matched = matched[mask]
        
        if len(matched) == 0:
            return 0.0
        
        # Compute precision
        true_positives = np.sum(matched)
        total_predictions = len(matched)
        
        precision = true_positives / total_predictions if total_predictions > 0 else 0.0
        return float(precision)

    def evaluate(self):
        if not self._predictions:
            self._logger.warning("No predictions to evaluate!")
            return None
            
        # Reset metrics
        for metrics in self.metrics.values():
            metrics.update({'TP': 0, 'FP': 0, 'FN': 0, 'total_gt': 0, 'total_pred': 0})
        
        # Accumulate metrics across all images
        for pred in self._predictions:
            eval_info = pred["evaluation"]
            
            # Count instances
            for gt_class in eval_info['gt_classes']:
                self.metrics[self.classes[gt_class]]['total_gt'] += 1
            for pred_class in eval_info['pred_classes']:
                self.metrics[self.classes[pred_class]]['total_pred'] += 1
            
            # Count matches
            for gt_idx, (gt_class, is_matched) in enumerate(zip(eval_info['gt_classes'], eval_info['gt_matched'])):
                gt_class_name = self.classes[gt_class]
                if is_matched:
                    self.metrics[gt_class_name]['TP'] += 1
                else:
                    self.metrics[gt_class_name]['FN'] += 1
            
            for pred_idx, (pred_class, is_tp) in enumerate(zip(eval_info['pred_classes'], eval_info['is_tp'])):
                if not is_tp:
                    self.metrics[self.classes[pred_class]]['FP'] += 1
        
        # Compute final metrics
        results = OrderedDict()
        metrics_file = os.path.join(self._output_dir, "detailed_metrics.csv")
        
        # Calculate mean AP
        mean_ap = 0.0
        valid_classes = 0
        
        with open(metrics_file, "w") as f:
            f.write("class,total_gt,total_pred,TP,FP,FN,precision,recall,f1,AP\n")
            
            for class_name, metrics in self.metrics.items():
                tp, fp = metrics['TP'], metrics['FP']
                fn = metrics['FN']
                total_gt = metrics['total_gt']
                total_pred = metrics['total_pred']
                
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / total_gt if total_gt > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                # Compute AP at confidence threshold
                ap = self._compute_ap(metrics['scores'], metrics['matched'])
                
                if total_gt > 0:
                    mean_ap += ap
                    valid_classes += 1
                
                f.write(f"{class_name},{total_gt},{total_pred},{tp},{fp},{fn},{precision:.4f},{recall:.4f},{f1:.4f},{ap:.4f}\n")
                
                # Store metrics
                results[f"{class_name}/total_gt"] = total_gt
                results[f"{class_name}/total_pred"] = total_pred
                results[f"{class_name}/TP"] = tp
                results[f"{class_name}/FP"] = fp
                results[f"{class_name}/FN"] = fn
                results[f"{class_name}/precision"] = precision * 100
                results[f"{class_name}/recall"] = recall * 100
                results[f"{class_name}/f1"] = f1 * 100
                results[f"{class_name}/AP"] = ap * 100
        
        # Add mean AP
        if valid_classes > 0:
            results["mean/AP"] = (mean_ap / valid_classes) * 100
        
        self._logger.info(f"Detailed metrics saved to: {metrics_file}")
        return results

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
        logger.info(f"\nEvaluating dataset: {dataset_name}")
        evaluator = DetailedCityscapesEvaluator(dataset_name=dataset_name)
        results[dataset_name] = DefaultTrainer.test(cfg, model, evaluators=[evaluator])
        logger.info(f"Results for {dataset_name}:")
        for k, v in results[dataset_name].items():
            logger.info(f"  {k}: {v}")
    
    if comm.is_main_process():
        verify_results(cfg, results)
    
    return results

if __name__ == "__main__":
    main() 