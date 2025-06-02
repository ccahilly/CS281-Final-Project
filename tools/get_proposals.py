#!/usr/bin/env python

import os
import logging
import numpy as np
import torch
import json
from pathlib import Path
import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
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

def main():
    """Main function to extract proposals for a specific prediction."""
    logger = setup_logger()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract RPN proposals for a specific prediction")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("pred_instance_idx", type=int, help="Index of the prediction")
    parser.add_argument("pred_class", help="Class name of the prediction")
    args = parser.parse_args()
    
    # Set up config and model
    cfg = setup_cfg()
    extractor = ProposalExtractor(cfg)
    
    # Extract proposals
    proposal_data = extractor.extract_proposals_for_prediction(
        args.image_path,
        args.pred_instance_idx,
        args.pred_class
    )
    
    if proposal_data is None:
        logger.error("Failed to extract proposals")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 