#!/usr/bin/env python

import os
import logging
import numpy as np
import torch
import json
from pathlib import Path
import cv2
import pandas as pd
import sys

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

# Constants for evaluation configuration
CONFIG_FILE = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
CONFIDENCE_THRESHOLD = 0.5

def setup_cfg():
    """Set up detectron2 config."""
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for easier debugging
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
    # Make sure RPN proposals are saved
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000  # Number of proposals to keep
    cfg.MODEL.RPN.RETURN_PROPOSALS = True  # Make sure proposals are returned
    return cfg

class ProposalExtractor:
    def __init__(self, cfg):
        # Modify config to return proposals
        cfg = cfg.clone()
        cfg.MODEL.RPN.RETURN_PROPOSALS = True
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self._logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = "output/proposals"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_proposals_for_prediction(self, image_path, pred_instance_idx, pred_class):
        """
        Extract RPN proposals that led to a specific prediction.
        
        Args:
            image_path: Path to the input image
            pred_instance_idx: Index of the prediction in the model's output
            pred_class: Class name of the prediction
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            self._logger.error(f"Could not read image: {image_path}")
            return
            
        # Get model predictions with RPN proposals
        with torch.no_grad():
            # Preprocess image
            transformed_image = self.predictor.aug.get_transform(image).apply_image(image)
            # Convert to tensor and add batch dimension
            image_tensor = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
            
            # Format input as expected by the model
            inputs = [{"image": image_tensor, "height": image.shape[0], "width": image.shape[1]}]
            
            # Run the full model to get both proposals and predictions
            outputs = self.predictor.model(inputs)
            
            # Get final predictions first
            instances = outputs[0]["instances"].to("cpu")
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_classes = instances.pred_classes.numpy()
            pred_scores = instances.scores.numpy()
            
            if pred_instance_idx >= len(pred_boxes):
                self._logger.error(f"Prediction index {pred_instance_idx} out of range")
                return
            
            # Now get proposals by running the model's RPN
            images = self.predictor.model.preprocess_image(inputs)
            features = self.predictor.model.backbone(images.tensor)
            
            # Get the feature names that RPN expects
            rpn_in_features = self.predictor.model.proposal_generator.in_features
            self._logger.info(f"RPN expects features: {rpn_in_features}")
            self._logger.info(f"Available features: {list(features.keys())}")
            
            # Run RPN with the features
            proposals = self.predictor.model.proposal_generator(images, features)[0]
            proposals = proposals[0]  # Get proposals for the first (and only) image
            
            # Extract proposal boxes and scores
            proposal_boxes = proposals.proposal_boxes.tensor.cpu().numpy()
            proposal_scores = proposals.objectness_logits.cpu().numpy()
            
            # Get the specific prediction we're interested in
            target_box = pred_boxes[pred_instance_idx]
            target_class = pred_classes[pred_instance_idx]
            target_score = pred_scores[pred_instance_idx]
            
            # Verify the class matches
            if self.metadata.thing_classes[target_class] != pred_class:
                self._logger.error(f"Prediction class mismatch: expected {pred_class}, got {self.metadata.thing_classes[target_class]}")
                return
            
            # Calculate IoU between the prediction and all proposals
            ious = self._compute_ious(target_box, proposal_boxes)
            
            # Get proposals with significant overlap (IoU > 0.5)
            relevant_mask = ious > 0.5
            relevant_proposals = proposal_boxes[relevant_mask]
            relevant_scores = proposal_scores[relevant_mask]
            
            # Create output directory for this image and prediction
            image_name = Path(image_path).stem
            output_subdir = os.path.join(self.output_dir, image_name, f"{pred_instance_idx}_{pred_class}")
            os.makedirs(output_subdir, exist_ok=True)
            
            # Create crops directory
            crops_dir = os.path.join(output_subdir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            
            # Save crop of the final prediction
            final_pred_crop = self._get_crop(image, target_box)
            cv2.imwrite(os.path.join(crops_dir, f"final_prediction_score_{target_score:.3f}.jpg"), final_pred_crop)
            
            # Save crops of relevant proposals
            for i, (proposal, score) in enumerate(zip(relevant_proposals, relevant_scores)):
                proposal_crop = self._get_crop(image, proposal)
                cv2.imwrite(os.path.join(crops_dir, f"proposal_{i:03d}_score_{score:.3f}.jpg"), proposal_crop)
            
            # Save the original image with boxes drawn
            vis_image = image.copy()
            
            # Draw the final prediction in green
            self._draw_box(vis_image, target_box, (0, 255, 0), 2)  # Green for final prediction
            
            # Draw relevant proposals in blue with varying transparency based on score
            for proposal, score in zip(relevant_proposals, relevant_scores):
                alpha = float(score)  # Use objectness score for transparency
                color = (255, 0, 0)  # Blue for proposals
                self._draw_box(vis_image, proposal, color, 1)
            
            # Save visualization
            cv2.imwrite(os.path.join(output_subdir, "visualization.jpg"), vis_image)
            
            # Save proposal data
            proposal_data = {
                "image_path": image_path,
                "prediction": {
                    "box": target_box.tolist(),
                    "class": pred_class,
                    "score": float(target_score),
                    "instance_idx": pred_instance_idx
                },
                "relevant_proposals": [
                    {
                        "box": box.tolist(),
                        "score": float(score),
                        "iou": float(iou)
                    }
                    for box, score, iou in zip(relevant_proposals, relevant_scores, ious[relevant_mask])
                ]
            }
            
            with open(os.path.join(output_subdir, "proposal_data.json"), "w") as f:
                json.dump(proposal_data, f, indent=2)
            
            self._logger.info(f"Saved proposal analysis to {output_subdir}")
            return proposal_data
    
    def _compute_ious(self, box, boxes):
        """Compute IoU between a box and an array of boxes."""
        # Convert boxes to format [x1, y1, x2, y2]
        box = box.reshape(1, 4)
        
        # Calculate intersection
        x1 = np.maximum(box[0, 0], boxes[:, 0])
        y1 = np.maximum(box[0, 1], boxes[:, 1])
        x2 = np.minimum(box[0, 2], boxes[:, 2])
        y2 = np.minimum(box[0, 3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas
        box_area = (box[0, 2] - box[0, 0]) * (box[0, 3] - box[0, 1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Calculate IoU
        union = box_area + boxes_area - intersection
        iou = intersection / union
        
        return iou
    
    def _draw_box(self, image, box, color, thickness):
        """Draw a bounding box on an image."""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
    def _get_crop(self, image, box):
        """Extract a crop from an image given a bounding box."""
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        return image[y1:y2, x1:x2]

def setup_logger():
    """Set up the logger for real-time output."""
    logger = logging.getLogger("detectron2")
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def process_misclassification_csv(csv_path, extractor):
    """
    Process a CSV file containing misclassification details.
    
    Args:
        csv_path: Path to the CSV file with misclassification details
        extractor: Instance of ProposalExtractor
    """
    logger = logging.getLogger(__name__)
    print(f"\nProcessing misclassification file: {csv_path}", flush=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    total_instances = len(df)
    print(f"Found {total_instances} instances to process", flush=True)
    
    # Process each misclassification case
    successful = 0
    failed = 0
    
    for idx, row in df.iterrows():
        image_path = row['file_name']
        pred_instance_idx = row['pred_instance_idx']
        pred_class = row['pred_class']
        
        # Calculate progress percentage
        progress = (idx + 1) / total_instances * 100
        print(f"Processing instance {idx + 1}/{total_instances} ({progress:.1f}%) - Image: {image_path}, Prediction: {pred_instance_idx} (class: {pred_class})", flush=True)
        
        try:
            proposal_data = extractor.extract_proposals_for_prediction(
                image_path,
                pred_instance_idx,
                pred_class
            )
            if proposal_data is None:
                print(f"Failed to extract proposals for {image_path}", flush=True)
                failed += 1
            else:
                successful += 1
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}", flush=True)
            failed += 1
    
    # Print final statistics
    print(f"\nFinished processing {csv_path}", flush=True)
    print(f"Successfully processed: {successful}/{total_instances} instances", flush=True)
    if failed > 0:
        print(f"Failed to process: {failed}/{total_instances} instances", flush=True)
    
    return successful, failed

def main():
    """Main function to extract proposals for misclassification cases."""
    logger = setup_logger()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract RPN proposals for misclassification cases")
    parser.add_argument("--fn_person_rider", default="output/misclassification_analysis_complete/fn_person_rider_details.csv",
                      help="Path to CSV file containing person->rider misclassifications")
    parser.add_argument("--fn_rider_person", default="output/misclassification_analysis_complete/fn_rider_person_details.csv",
                      help="Path to CSV file containing rider->person misclassifications")
    args = parser.parse_args()
    
    # Set up config and model
    cfg = setup_cfg()
    extractor = ProposalExtractor(cfg)
    
    total_successful = 0
    total_failed = 0
    
    # Process both misclassification files
    for csv_path in [args.fn_person_rider, args.fn_rider_person]:
        if os.path.exists(csv_path):
            successful, failed = process_misclassification_csv(csv_path, extractor)
            total_successful += successful
            total_failed += failed
        else:
            print(f"Misclassification file not found: {csv_path}", flush=True)
    
    # Print overall statistics
    total_instances = total_successful + total_failed
    if total_instances > 0:
        print("\n=== Overall Statistics ===", flush=True)
        print(f"Total instances processed: {total_instances}", flush=True)
        print(f"Total successful: {total_successful} ({total_successful/total_instances*100:.1f}%)", flush=True)
        if total_failed > 0:
            print(f"Total failed: {total_failed} ({total_failed/total_instances*100:.1f}%)", flush=True)
    
    return 0

if __name__ == "__main__":
    exit(main()) 