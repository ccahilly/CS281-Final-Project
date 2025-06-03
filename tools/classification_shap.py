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
    def __init__(self, backbone, box_pooler, box_head, box_predictor, person_idx, rider_idx):
        super().__init__()
        self.backbone = backbone
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.person_idx = person_idx
        self.rider_idx = rider_idx
        # The feature names are typically p2, p3, p4, p5 for FPN
        self.feature_names = ["p2", "p3", "p4", "p5"]
        
    def forward(self, x):
        # x should be [N, C, H, W]
        with torch.set_grad_enabled(True):  # Ensure gradients are computed
            features = self.backbone(x)
            # Get only the FPN feature levels
            features = [features[f] for f in self.feature_names if f in features]
            
            # Create boxes for each image in the batch
            boxes = []
            for i in range(x.shape[0]):
                box = torch.tensor([[0, 0, x.shape[3], x.shape[2]]]).float()
                boxes.append(Boxes(box.to(x.device)))
            
            box_features = self.box_pooler(features, boxes)
            box_features = self.box_head(box_features)
            scores = self.box_predictor.cls_score(box_features)
            
            # Extract only person and rider scores
            person_rider_scores = scores[:, [self.person_idx, self.rider_idx]]
            return person_rider_scores

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
        image = cv2.imread(image_path)
        if image is None:
            self._logger.error(f"Could not read image: {image_path}")
            return
            
        with open(proposal_data_path, 'r') as f:
            proposal_data = json.load(f)
        
        print(f"Found {len(proposal_data['relevant_proposals'])} proposals to analyze")
        
        # Get the model's box head and predictor
        box_head = self.predictor.model.roi_heads.box_head
        box_predictor = self.predictor.model.roi_heads.box_predictor
        
        # Process each proposal
        results = []
        for i, proposal in enumerate(proposal_data['relevant_proposals']):
            print(f"\nProcessing proposal {i+1}/{len(proposal_data['relevant_proposals'])}")
            try:
                # Get proposal box
                box = torch.tensor(proposal['box']).unsqueeze(0)
                x1, y1, x2, y2 = map(int, box[0])
                print(f"Proposal box coordinates: ({x1}, {y1}, {x2}, {y2})")
                
                # Extract the proposal region from the image
                proposal_image = image[y1:y2, x1:x2]
                if proposal_image.size == 0:
                    print(f"Skipping proposal {i} - empty region")
                    continue
                    
                print("Resizing proposal image...")
                # Resize to a standard size for analysis
                proposal_image = cv2.resize(proposal_image, (224, 224))
                
                print("Creating model wrapper...")
                # Create model wrapper
                model = ModelWrapper(
                    self.predictor.model.backbone,
                    self.predictor.model.roi_heads.box_pooler,
                    box_head,
                    box_predictor,
                    self.person_idx,
                    self.rider_idx
                )
                
                print("Preprocessing image...")
                # Preprocess image
                image_tensor = torch.from_numpy(
                    proposal_image.transpose(2, 0, 1).astype(np.float32)
                ) / 255.0
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
                # Create background distribution
                background = torch.zeros_like(image_tensor)
                
                print("Creating SHAP explainer...")
                # Create explainer using GradientExplainer
                explainer = shap.GradientExplainer(model, background)
                
                print("Calculating SHAP values...")
                # Calculate SHAP values
                shap_values = explainer.shap_values(image_tensor)
                
                print("Getting model predictions...")
                # Get prediction
                with torch.no_grad():
                    pred = model(image_tensor)
                    probs = torch.nn.functional.softmax(pred, dim=1)
                    pred_class_idx = probs[0].argmax().item()  # 0 for person, 1 for rider
                    pred_score = probs[0][pred_class_idx].item()
                    class_names = ['person', 'rider']
                    pred_class = class_names[pred_class_idx]
                
                print(f"Prediction: {pred_class} with score {pred_score:.3f}")
                
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
                
                # Plot original image
                ax1.imshow(cv2.cvtColor(proposal_image, cv2.COLOR_BGR2RGB))
                ax1.set_title(f"Original Proposal\nPrediction: {pred_class} ({pred_score:.3f})")
                ax1.axis('off')
                
                # Plot SHAP values for person class
                shap_person = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]
                shap_person = np.transpose(shap_person, (1, 2, 0))
                person_map = shap_person.sum(axis=2)
                abs_max = np.abs(person_map).max()
                im2 = ax2.imshow(person_map, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
                ax2.set_title(f"SHAP values for 'person'\nProbability: {probs[0][0]:.3f}")
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2)
                
                # Plot SHAP values for rider class
                shap_rider = shap_values[1][0] if isinstance(shap_values, list) else shap_values[1]
                shap_rider = np.transpose(shap_rider, (1, 2, 0))
                rider_map = shap_rider.sum(axis=2)
                im3 = ax3.imshow(rider_map, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
                ax3.set_title(f"SHAP values for 'rider'\nProbability: {probs[0][1]:.3f}")
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3)
                
                plt.tight_layout()
                plt.savefig(shap_img_path)
                plt.close()
                
                print(f"Saved visualization to {shap_img_path}")
                
            except Exception as e:
                print(f"Error processing proposal {i}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Save all results to JSON
        output_path = os.path.join(
            self.output_dir,
            f"{Path(image_path).stem}_shap_analysis.json"
        )
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
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