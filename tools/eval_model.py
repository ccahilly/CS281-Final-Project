#!/usr/bin/env python

import logging
import os
import numpy as np
from collections import OrderedDict, defaultdict
import cv2

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import CityscapesInstanceEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.utils.comm as comm

# Constants for evaluation configuration
CONFIG_FILE = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
CONFIDENCE_THRESHOLD = 0.5

class DetailedCityscapesEvaluator(CityscapesInstanceEvaluator):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        
        # Get Cityscapes metadata
        self.metadata = MetadataCatalog.get(dataset_name)
        self.classes = self.metadata.thing_classes
        self.num_classes = len(self.classes)
        
        # Map Cityscapes training IDs to our class indices
        from cityscapesscripts.helpers.labels import labels
        self.trainId_to_class = {}
        thing_count = 0
        for label in labels:
            if label.hasInstances and not label.ignoreInEval:
                # Map from instance ID to contiguous thing class id
                self.trainId_to_class[label.id] = thing_count
                thing_count += 1
        
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
        
        # Create output directories for visualizations
        self._output_dir = os.path.join(os.path.dirname(dataset_name), "evaluation_output")
        self._vis_dir = os.path.join(self._output_dir, "visualizations")
        self._gt_vis_dir = os.path.join(self._vis_dir, "ground_truth")
        self._pred_vis_dir = os.path.join(self._vis_dir, "predictions")
        
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._gt_vis_dir, exist_ok=True)
        os.makedirs(self._pred_vis_dir, exist_ok=True)

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
                
            # Load instance segmentation and original image
            gt_seg = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            original_img = cv2.imread(img_path)
            if gt_seg is None or original_img is None:
                self._logger.warning(f"Could not read image files: {gt_path} or {img_path}")
                continue
                
            # Convert BGR to RGB for visualization
            original_img = original_img[:, :, ::-1]
                
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
            
            # Visualize ground truth
            v_gt = Visualizer(original_img, metadata=self.metadata)
            v_gt = v_gt.overlay_instances(
                boxes=np.array(gt_boxes),
                labels=[self.classes[i] for i in gt_classes],
                masks=gt_instances
            )
            gt_vis = v_gt.get_image()
            
            # Visualize predictions
            v_pred = Visualizer(original_img, metadata=self.metadata)
            v_pred = v_pred.draw_instance_predictions(pred_instances)
            pred_vis = v_pred.get_image()
            
            # Save visualizations
            img_name = os.path.basename(img_path)
            cv2.imwrite(
                os.path.join(self._gt_vis_dir, f"gt_{img_name}"), 
                gt_vis[:, :, ::-1]  # Convert back to BGR for cv2
            )
            cv2.imwrite(
                os.path.join(self._pred_vis_dir, f"pred_{img_name}"), 
                pred_vis[:, :, ::-1]  # Convert back to BGR for cv2
            )
            
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
        
        # Calculate ground truth totals from the dataset
        for pred in self._predictions:
            gt_classes = pred["gt_classes"]
            for class_idx in gt_classes:
                self.metrics[self.classes[class_idx]]['total_gt'] += 1
        
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

def setup_cfg():
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
    cfg.freeze()
    return cfg

def main():
    # Set up logger
    logger = setup_logger()
    
    # Create model configuration
    cfg = setup_cfg()
    logger.info("Using configuration:")
    logger.info(f"  Config file: {CONFIG_FILE}")
    logger.info(f"  Weights: {WEIGHTS}")
    logger.info(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"  Datasets to evaluate: {cfg.DATASETS.TEST}")
    
    # Build model and load weights properly
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name in cfg.DATASETS.TEST:
        logger.info(f"\nEvaluating dataset: {dataset_name}")
        
        # Create evaluator for this dataset
        evaluator = DetailedCityscapesEvaluator(dataset_name=dataset_name)
        
        # Run evaluation using proper testing pipeline
        results = DefaultTrainer.test(cfg, model, evaluators=[evaluator])
        
        # Store results
        all_results[dataset_name] = results
        
        # Print results for this dataset
        logger.info(f"Results for {dataset_name}:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")
    
    # Run test-time augmentation if enabled
    if cfg.TEST.AUG.ENABLED:
        logger.info("\nRunning evaluation with test-time augmentation...")
        from detectron2.modeling import GeneralizedRCNNWithTTA
        model = GeneralizedRCNNWithTTA(cfg, model)
        for dataset_name in cfg.DATASETS.TEST:
            evaluator = DetailedCityscapesEvaluator(dataset_name=dataset_name)
            results = DefaultTrainer.test(cfg, model, evaluators=[evaluator])
            all_results[f"{dataset_name}_TTA"] = results
    
    if comm.is_main_process():
        # Verify results if we're the main process
        from detectron2.evaluation import verify_results
        verify_results(cfg, all_results)
    
    return all_results

if __name__ == "__main__":
    main() 