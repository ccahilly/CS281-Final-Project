#!/usr/bin/env python

import os
import logging
import numpy as np
import torch
import json
from pathlib import Path
import cv2
import pandas as pd
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes

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
    return cfg

class ModelWrapper(torch.nn.Module):
    def __init__(self, box_head, box_predictor, person_idx, rider_idx):
        super().__init__()
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.person_idx = person_idx
        self.rider_idx = rider_idx
        
    def forward(self, pooled_features):
        # pooled_features should be the output of ROI pooling
        with torch.set_grad_enabled(True):  # Ensure gradients are computed
            box_features = self.box_head(pooled_features)
            scores = self.box_predictor.cls_score(box_features)
            
            # Extract only person and rider scores
            person_rider_scores = scores[:, [self.person_idx, self.rider_idx]]
            return person_rider_scores

    def analyze_proposal(self, image_path, proposal_data_path):
        """
        Analyze a proposal using SHAP, focusing on person vs. rider classification.
        """
        print(f"\nAnalyzing proposals for image: {image_path}")
        
        # Read image and proposal data
        image = cv2.imread(image_path)  # OpenCV reads as BGR uint8
        if image is None:
            self._logger.error(f"Could not read image: {image_path}")
            return
            
        with open(proposal_data_path, 'r') as f:
            proposal_data = json.load(f)
        
        print(f"Found {len(proposal_data['relevant_proposals'])} proposals to analyze")
        
        # Get the model's box head and predictor
        box_head = self.predictor.model.roi_heads.box_head
        box_predictor = self.predictor.model.roi_heads.box_predictor
        
        # Extract features from the full image once
        print("Extracting features from full image...")
        with torch.no_grad():
            # Convert BGR uint8 to RGB float32 tensor with batch dimension
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)) / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # Get features from backbone
            features_dict = self.predictor.model.backbone(image_tensor)
            # Get FPN feature levels
            feature_names = ["p2", "p3", "p4", "p5"]
            features = [features_dict[f] for f in feature_names if f in features_dict]
        
        # Create model wrapper that works with pooled features
        model = ModelWrapper(
            box_head,
            box_predictor,
            self.person_idx,
            self.rider_idx
        )
        
        # Process each proposal
        results = []
        for i, proposal in enumerate(proposal_data['relevant_proposals']):
            print(f"\nProcessing proposal {i+1}/{len(proposal_data['relevant_proposals'])}")
            try:
                # Get proposal box
                box = torch.tensor(proposal['box']).unsqueeze(0)
                x1, y1, x2, y2 = map(int, box[0])
                print(f"Proposal box coordinates: ({x1}, {y1}, {x2}, {y2})")
                
                # Create Boxes object for the proposal
                boxes = [Boxes(box.to(image_tensor.device))]
                
                # Get pooled features for this proposal
                with torch.no_grad():
                    pooled_features = self.predictor.model.roi_heads.box_pooler(features, boxes)
                
                # Create background distribution for SHAP (zero tensor of same size)
                background = torch.zeros_like(pooled_features)
                
                print("Creating SHAP explainer...")
                # Create explainer using GradientExplainer
                explainer = shap.GradientExplainer(model, background)
                
                print("Calculating SHAP values...")
                # Calculate SHAP values
                shap_values = explainer.shap_values(pooled_features)
                
                print("Getting model predictions...")
                # Get prediction
                with torch.no_grad():
                    pred = model(pooled_features)
                    probs = torch.nn.functional.softmax(pred, dim=1)
                    pred_class_idx = probs[0].argmax().item()  # 0 for person, 1 for rider
                    pred_score = probs[0][pred_class_idx].item()
                    class_names = ['person', 'rider']
                    pred_class = class_names[pred_class_idx]
                
                print(f"Prediction: {pred_class} with score {pred_score:.3f}")
                
                # Extract the proposal region from the image for visualization
                proposal_image = image[y1:y2, x1:x2]
                
                # Save results
                result = {
                    'proposal_idx': i,
                    'box': proposal['box'],
                    'score': proposal['score'],
                    'iou': proposal['iou'],
                    'predicted_class': pred_class,
                    'person_prob': float(probs[0][0]),  # probability of person
                    'rider_prob': float(probs[0][1]),   # probability of rider
                }
                results.append(result)
                
                print("Generating visualization...")
                # Save SHAP visualization
                shap_img_path = os.path.join(
                    self.output_dir,
                    f"{Path(image_path).stem}_proposal_{i}_shap.png"
                )
                
                # Create figure with three subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
                
                # Plot original image (convert BGR to RGB for matplotlib)
                ax1.imshow(cv2.cvtColor(proposal_image, cv2.COLOR_BGR2RGB))
                ax1.set_title(f"Original Proposal\nPrediction: {pred_class} ({pred_score:.3f})")
                ax1.axis('off')
                
                # Process SHAP values for person class
                shap_person = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                person_map = shap_person.mean(axis=1)  # Average over channels
                # Resize person_map to match proposal image size
                person_map = cv2.resize(person_map, (proposal_image.shape[1], proposal_image.shape[0]))
                
                # Process SHAP values for rider class
                shap_rider = shap_values[1] if isinstance(shap_values, list) else shap_values[1]
                rider_map = shap_rider.mean(axis=1)  # Average over channels
                # Resize rider_map to match proposal image size
                rider_map = cv2.resize(rider_map, (proposal_image.shape[1], proposal_image.shape[0]))
                
                # Normalize SHAP values for visualization
                abs_max = max(np.abs(person_map).max(), np.abs(rider_map).max())
                person_map = person_map / abs_max
                rider_map = rider_map / abs_max
                
                # Create overlays for person class
                proposal_rgb = cv2.cvtColor(proposal_image, cv2.COLOR_BGR2RGB)
                person_heatmap = plt.cm.RdBu_r(0.5 + person_map/2)[:, :, :3]  # Convert to RGB, exclude alpha
                person_overlay = (proposal_rgb / 255.0 * 0.7 + person_heatmap * 0.3)  # Blend with original image
                person_overlay = np.clip(person_overlay, 0, 1)
                
                # Create overlays for rider class
                rider_heatmap = plt.cm.RdBu_r(0.5 + rider_map/2)[:, :, :3]  # Convert to RGB, exclude alpha
                rider_overlay = (proposal_rgb / 255.0 * 0.7 + rider_heatmap * 0.3)  # Blend with original image
                rider_overlay = np.clip(rider_overlay, 0, 1)
                
                # Plot overlaid SHAP values
                ax2.imshow(person_overlay)
                ax2.set_title(f"SHAP values for 'person' (overlaid)\nProbability: {probs[0][0]:.3f}")
                ax2.axis('off')
                
                ax3.imshow(rider_overlay)
                ax3.set_title(f"SHAP values for 'rider' (overlaid)\nProbability: {probs[0][1]:.3f}")
                ax3.axis('off')
                
                plt.tight_layout()
                plt.savefig(shap_img_path)
                plt.close()
                
                print(f"Saved visualization to {shap_img_path}")
                
            except Exception as e:
                print(f"Error processing proposal {i}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        return results

class ShapAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self._logger = logging.getLogger(__name__)
        
        # Get indices for person and rider classes
        self.person_idx = self.metadata.thing_classes.index('person')
        self.rider_idx = self.metadata.thing_classes.index('rider')
        
        # Create output directory
        self.output_dir = "output/shap_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_proposal(self, image_path, proposal_data_path):
        """
        Analyze a proposal using SHAP, focusing on person vs. rider classification.
        """
        print(f"\nAnalyzing proposals for image: {image_path}")
        
        # Read image and proposal data
        image = cv2.imread(image_path)  # OpenCV reads as BGR uint8
        if image is None:
            self._logger.error(f"Could not read image: {image_path}")
            return
            
        with open(proposal_data_path, 'r') as f:
            proposal_data = json.load(f)
        
        print(f"Found {len(proposal_data['relevant_proposals'])} proposals to analyze")
        
        # Get the model's box head and predictor
        box_head = self.predictor.model.roi_heads.box_head
        box_predictor = self.predictor.model.roi_heads.box_predictor
        
        # Extract features from the full image once
        print("Extracting features from full image...")
        with torch.no_grad():
            # Convert BGR uint8 to RGB float32 tensor with batch dimension
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)) / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # Get features from backbone
            features_dict = self.predictor.model.backbone(image_tensor)
            # Get FPN feature levels
            feature_names = ["p2", "p3", "p4", "p5"]
            features = [features_dict[f] for f in feature_names if f in features_dict]
        
        # Create model wrapper that works with pooled features
        model = ModelWrapper(
            box_head,
            box_predictor,
            self.person_idx,
            self.rider_idx
        )
        
        # Process each proposal
        results = []
        for i, proposal in enumerate(proposal_data['relevant_proposals']):
            print(f"\nProcessing proposal {i+1}/{len(proposal_data['relevant_proposals'])}")
            try:
                # Get proposal box
                box = torch.tensor(proposal['box']).unsqueeze(0)
                x1, y1, x2, y2 = map(int, box[0])
                print(f"Proposal box coordinates: ({x1}, {y1}, {x2}, {y2})")
                
                # Create Boxes object for the proposal
                boxes = [Boxes(box.to(image_tensor.device))]
                
                # Get pooled features for this proposal
                with torch.no_grad():
                    pooled_features = self.predictor.model.roi_heads.box_pooler(features, boxes)
                
                # Create background distribution for SHAP (zero tensor of same size)
                background = torch.zeros_like(pooled_features)
                
                print("Creating SHAP explainer...")
                # Create explainer using GradientExplainer
                explainer = shap.GradientExplainer(model, background)
                
                print("Calculating SHAP values...")
                # Calculate SHAP values
                shap_values = explainer.shap_values(pooled_features)
                
                print("Getting model predictions...")
                # Get prediction
                with torch.no_grad():
                    pred = model(pooled_features)
                    probs = torch.nn.functional.softmax(pred, dim=1)
                    pred_class_idx = probs[0].argmax().item()  # 0 for person, 1 for rider
                    pred_score = probs[0][pred_class_idx].item()
                    class_names = ['person', 'rider']
                    pred_class = class_names[pred_class_idx]
                
                print(f"Prediction: {pred_class} with score {pred_score:.3f}")
                
                # Extract the proposal region from the image for visualization
                proposal_image = image[y1:y2, x1:x2]
                
                # Save results
                result = {
                    'proposal_idx': i,
                    'box': proposal['box'],
                    'score': proposal['score'],
                    'iou': proposal['iou'],
                    'predicted_class': pred_class,
                    'person_prob': float(probs[0][0]),  # probability of person
                    'rider_prob': float(probs[0][1]),   # probability of rider
                }
                results.append(result)
                
                print("Generating visualization...")
                # Save SHAP visualization
                shap_img_path = os.path.join(
                    self.output_dir,
                    f"{Path(image_path).stem}_proposal_{i}_shap.png"
                )
                
                # Create figure with three subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
                
                # Plot original image (convert BGR to RGB for matplotlib)
                ax1.imshow(cv2.cvtColor(proposal_image, cv2.COLOR_BGR2RGB))
                ax1.set_title(f"Original Proposal\nPrediction: {pred_class} ({pred_score:.3f})")
                ax1.axis('off')
                
                # Process SHAP values for person class
                shap_person = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                person_map = shap_person.mean(axis=1)  # Average over channels
                # Resize person_map to match proposal image size
                person_map = cv2.resize(person_map, (proposal_image.shape[1], proposal_image.shape[0]))
                
                # Process SHAP values for rider class
                shap_rider = shap_values[1] if isinstance(shap_values, list) else shap_values[1]
                rider_map = shap_rider.mean(axis=1)  # Average over channels
                # Resize rider_map to match proposal image size
                rider_map = cv2.resize(rider_map, (proposal_image.shape[1], proposal_image.shape[0]))
                
                # Normalize SHAP values for visualization
                abs_max = max(np.abs(person_map).max(), np.abs(rider_map).max())
                person_map = person_map / abs_max
                rider_map = rider_map / abs_max
                
                # Create overlays for person class
                proposal_rgb = cv2.cvtColor(proposal_image, cv2.COLOR_BGR2RGB)
                person_heatmap = plt.cm.RdBu_r(0.5 + person_map/2)[:, :, :3]  # Convert to RGB, exclude alpha
                person_overlay = (proposal_rgb / 255.0 * 0.7 + person_heatmap * 0.3)  # Blend with original image
                person_overlay = np.clip(person_overlay, 0, 1)
                
                # Create overlays for rider class
                rider_heatmap = plt.cm.RdBu_r(0.5 + rider_map/2)[:, :, :3]  # Convert to RGB, exclude alpha
                rider_overlay = (proposal_rgb / 255.0 * 0.7 + rider_heatmap * 0.3)  # Blend with original image
                rider_overlay = np.clip(rider_overlay, 0, 1)
                
                # Plot overlaid SHAP values
                ax2.imshow(person_overlay)
                ax2.set_title(f"SHAP values for 'person' (overlaid)\nProbability: {probs[0][0]:.3f}")
                ax2.axis('off')
                
                ax3.imshow(rider_overlay)
                ax3.set_title(f"SHAP values for 'rider' (overlaid)\nProbability: {probs[0][1]:.3f}")
                ax3.axis('off')
                
                plt.tight_layout()
                plt.savefig(shap_img_path)
                plt.close()
                
                print(f"Saved visualization to {shap_img_path}")
                
            except Exception as e:
                print(f"Error processing proposal {i}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        return results

def process_all_proposals(analyzer):
    """Process all proposals in the output directory."""
    proposals_dir = "output/proposals"
    
    # Find all proposal data files
    proposal_files = []
    for root, _, files in os.walk(proposals_dir):
        for file in files:
            if file == "proposal_data.json":
                proposal_files.append(os.path.join(root, file))
    
    print(f"Found {len(proposal_files)} proposal files to analyze")
    
    # Process each file
    all_results = []
    for proposal_file in tqdm(proposal_files):
        try:
            # Get image path from proposal data
            with open(proposal_file, 'r') as f:
                proposal_data = json.load(f)
                image_path = proposal_data['image_path']
            
            # Analyze proposals
            results = analyzer.analyze_proposal(image_path, proposal_file)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"Error processing {proposal_file}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Save all results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(analyzer.output_dir, "all_proposals_shap.csv"), index=False)

def main():
    """Main function to analyze proposals using SHAP."""
    logger = setup_logger()
    
    # Set up config and model
    cfg = setup_cfg()
    analyzer = ShapAnalyzer(cfg)
    
    # Process all proposals
    process_all_proposals(analyzer)
    
    return 0

if __name__ == "__main__":
    exit(main()) 