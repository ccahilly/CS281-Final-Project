#!/usr/bin/env python

import os
import sys
import json
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Constants
CONFIG_FILE = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
CLASSES_OF_INTEREST = ["person", "rider"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_cfg():
    """Set up detectron2 config."""
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Cityscapes has 8 classes
    return cfg

class ProposalClassificationExplainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Build model
        self.model = build_model(cfg)
        self.model.eval()
        
        # Load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        
        # Get metadata
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.class_names = self.metadata.thing_classes
        
        # Get class indices for our classes of interest
        self.class_indices = {
            class_name: self.class_names.index(class_name) 
            for class_name in CLASSES_OF_INTEREST 
            if class_name in self.class_names
        }
        
        # Setup logger
        self._logger = logging.getLogger(__name__)
        
    def extract_proposal_features(self, image, proposal_box):
        """
        Extract features for a single proposal box from the image.
        
        Args:
            image: Original image (HxWxC numpy array)
            proposal_box: Box coordinates [x1, y1, x2, y2]
            
        Returns:
            pooled_features: Features after ROI pooling
        """
        # Preprocess image
        height, width = image.shape[:2]
        
        # Create model inputs
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        
        with torch.no_grad():
            # Preprocess image
            images = self.model.preprocess_image(inputs)
            
            # Extract backbone features
            features = self.model.backbone(images.tensor)
            
            # Create proposal boxes
            proposal_boxes = Boxes(torch.tensor([proposal_box], dtype=torch.float32).to(self.device))
            
            # Pool features for this proposal
            box_features = self.model.roi_heads.box_pooler(
                [features[f] for f in self.model.roi_heads.box_in_features], 
                [proposal_boxes]
            )
            
            return box_features
    
    def classify_proposal(self, box_features):
        """
        Run classification on pooled box features.
        
        Args:
            box_features: Pooled features from ROI pooler
            
        Returns:
            scores: Classification scores for all classes
        """
        with torch.no_grad():
            # Pass through box head
            box_features = self.model.roi_heads.box_head(box_features)
            
            # Get classification scores
            predictions = self.model.roi_heads.box_predictor(box_features)
            scores = predictions[0]  # cls_score
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(scores, dim=-1)
            
            return probs
    
    def compute_shap_values(self, image, proposal_box, proposal_crop, class_name, num_samples=50):
        """
        Compute actual SHAP values for a specific class prediction.
        
        Args:
            image: Original image
            proposal_box: Box coordinates [x1, y1, x2, y2]
            proposal_crop: The cropped proposal region
            class_name: Class to explain ('person' or 'rider')
            num_samples: Number of samples for SHAP computation
            
        Returns:
            shap_values: SHAP values for the specified class
        """
        x1, y1, x2, y2 = map(int, proposal_box)
        class_idx = self.class_indices[class_name] + 1  # +1 for background class
        
        def predict_fn(masked_images):
            """Prediction function for SHAP that takes masked images and returns class probabilities."""
            # masked_images shape: (n_samples, height, width, channels)
            n_samples = masked_images.shape[0]
            predictions = []
            
            for i in range(n_samples):
                # Create a copy of the original image
                modified_image = image.copy()
                
                # Replace the proposal region with the masked version
                modified_image[y1:y2, x1:x2] = masked_images[i]
                
                # Extract features and classify
                box_features = self.extract_proposal_features(modified_image, proposal_box)
                probs = self.classify_proposal(box_features)
                probs_np = probs.detach().cpu().numpy()[0]
                
                predictions.append(probs_np[class_idx])
            
            return np.array(predictions)
        
        # Create a masker that uses the mean pixel value of the proposal
        masker = shap.maskers.Image("blur(128,128)", proposal_crop.shape)
        
        # Create the explainer
        explainer = shap.Explainer(predict_fn, masker)
        
        # Compute SHAP values
        # We use a smaller number of samples for computational efficiency
        shap_values = explainer(proposal_crop[np.newaxis, ...], max_evals=num_samples, batch_size=50)
        
        return shap_values
    
    def explain_proposal_classification(self, image, proposal_box, proposal_crop_path, output_dir):
        """
        Run SHAP analysis on a single proposal.
        
        Args:
            image: Original image
            proposal_box: Box coordinates [x1, y1, x2, y2]
            proposal_crop_path: Path to the proposal crop image
            output_dir: Directory to save SHAP visualizations
            
        Returns:
            dict: Results including class scores and SHAP values
        """
        # Extract the proposal crop for visualization
        x1, y1, x2, y2 = map(int, proposal_box)
        proposal_crop = image[y1:y2, x1:x2]
        
        # Get classification scores for the original proposal
        box_features = self.extract_proposal_features(image, proposal_box)
        probs = self.classify_proposal(box_features)
        probs_np = probs.detach().cpu().numpy()[0]
        
        # Get scores for classes of interest only
        scores_of_interest = {}
        for class_name, class_idx in self.class_indices.items():
            # Adjust for background class at index 0
            adjusted_idx = class_idx + 1
            if adjusted_idx < len(probs_np):
                scores_of_interest[class_name] = float(probs_np[adjusted_idx])
        
        results = {
            "proposal_box": proposal_box.tolist() if isinstance(proposal_box, np.ndarray) else proposal_box,
            "person_score": scores_of_interest.get("person", 0.0),
            "rider_score": scores_of_interest.get("rider", 0.0)
        }
        
        # Compute and save SHAP values for each class of interest
        shap_results = {}
        
        try:
            for class_name in CLASSES_OF_INTEREST:
                if class_name not in self.class_indices:
                    continue
                    
                self._logger.info(f"Computing SHAP values for {class_name} class...")
                
                # Compute SHAP values
                shap_values = self.compute_shap_values(image, proposal_box, proposal_crop, class_name)
                
                # Extract the SHAP values array
                shap_array = shap_values.values[0]  # Shape: (height, width, channels)
                
                # Save raw SHAP values
                proposal_name = Path(proposal_crop_path).stem
                shap_save_path = os.path.join(output_dir, f"{proposal_name}_shap_{class_name}.npy")
                np.save(shap_save_path, shap_array)
                
                # Compute aggregate statistics
                shap_mean = np.mean(shap_array)
                shap_abs_mean = np.mean(np.abs(shap_array))
                shap_max = np.max(shap_array)
                shap_min = np.min(shap_array)
                
                # Store SHAP statistics
                shap_results[f"{class_name}_shap_mean"] = float(shap_mean)
                shap_results[f"{class_name}_shap_abs_mean"] = float(shap_abs_mean)
                shap_results[f"{class_name}_shap_max"] = float(shap_max)
                shap_results[f"{class_name}_shap_min"] = float(shap_min)
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                # Original crop
                plt.subplot(1, 3, 1)
                plt.imshow(proposal_crop)
                plt.title(f"Original Proposal\n{class_name} score: {scores_of_interest.get(class_name, 0):.3f}")
                plt.axis('off')
                
                # SHAP values (averaged across channels)
                plt.subplot(1, 3, 2)
                shap_2d = np.mean(shap_array, axis=-1)  # Average across channels
                vmax = np.abs(shap_2d).max()
                plt.imshow(shap_2d, cmap='RdBu', vmin=-vmax, vmax=vmax)
                plt.colorbar(label='SHAP value')
                plt.title(f"SHAP values for '{class_name}'\n(red=positive, blue=negative)")
                plt.axis('off')
                
                # Absolute SHAP values (importance)
                plt.subplot(1, 3, 3)
                shap_abs = np.abs(shap_2d)
                plt.imshow(shap_abs, cmap='hot')
                plt.colorbar(label='|SHAP value|')
                plt.title(f"Absolute importance for '{class_name}'")
                plt.axis('off')
                
                # Save figure
                fig_save_path = os.path.join(output_dir, f"{proposal_name}_shap_{class_name}.png")
                plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self._logger.error(f"Error computing SHAP values: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Add placeholder values if SHAP computation failed
            for class_name in CLASSES_OF_INTEREST:
                if class_name in self.class_indices:
                    shap_results[f"{class_name}_shap_mean"] = None
                    shap_results[f"{class_name}_shap_abs_mean"] = None
                    shap_results[f"{class_name}_shap_max"] = None
                    shap_results[f"{class_name}_shap_min"] = None
        
        # Combine classification scores and SHAP results
        results.update(shap_results)
        
        return results

def process_proposal_directory(proposal_dir, explainer):
    """
    Process all proposals in a directory.
    
    Args:
        proposal_dir: Directory containing proposal data
        explainer: ProposalClassificationExplainer instance
        
    Returns:
        DataFrame with classification results
    """
    logger = logging.getLogger(__name__)
    
    # Load proposal data
    proposal_data_path = os.path.join(proposal_dir, "proposal_data.json")
    with open(proposal_data_path, 'r') as f:
        proposal_data = json.load(f)
    
    # Get image path
    image_path = proposal_data["image_path"]
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return pd.DataFrame()
    
    # Create output directory for SHAP visualizations
    shap_output_dir = os.path.join(proposal_dir, "shap_analysis")
    os.makedirs(shap_output_dir, exist_ok=True)
    
    # Process each proposal crop
    crops_dir = os.path.join(proposal_dir, "crops")
    results_list = []
    
    # Get all proposal crops
    proposal_crops = sorted([f for f in os.listdir(crops_dir) if f.startswith("proposal_")])
    
    logger.info(f"Processing {len(proposal_crops)} proposals from {proposal_dir}")
    logger.info(f"Number of relevant proposals in JSON: {len(proposal_data.get('relevant_proposals', []))}")
    
    # Debug: print first few filenames
    if proposal_crops:
        logger.info(f"Sample proposal filenames: {proposal_crops[:3]}")
    
    # Limit to first few proposals for testing (remove this limit for full run)
    # proposal_crops = proposal_crops[:3]  # Comment this out for full processing
    
    for i, crop_filename in enumerate(tqdm(proposal_crops, desc="Processing proposals")):
        crop_path = os.path.join(crops_dir, crop_filename)
        
        try:
            # Extract proposal index from filename
            # Handle format: proposal_XXX_score_YYY.jpg
            parts = crop_filename.replace('.jpg', '').split('_')
            if len(parts) >= 2 and parts[0] == 'proposal':
                proposal_idx = int(parts[1])
            else:
                logger.warning(f"Could not parse proposal index from {crop_filename}")
                continue
            
            # Get proposal box
            if proposal_idx < len(proposal_data["relevant_proposals"]):
                proposal_info = proposal_data["relevant_proposals"][proposal_idx]
                proposal_box = proposal_info["box"]
                
                # Run SHAP analysis
                results = explainer.explain_proposal_classification(
                    image, proposal_box, crop_path, shap_output_dir
                )
                
                # Add additional info
                results["proposal_idx"] = proposal_idx
                results["proposal_filename"] = crop_filename
                results["proposal_score"] = proposal_info["score"]
                results["proposal_iou"] = proposal_info["iou"]
                results["image_path"] = image_path
                results["prediction_class"] = proposal_data["prediction"]["class"]
                results["prediction_score"] = proposal_data["prediction"]["score"]
                
                results_list.append(results)
            else:
                logger.warning(f"Proposal index {proposal_idx} out of range for {crop_filename}")
                
        except Exception as e:
            logger.error(f"Error processing {crop_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if results_list:
        # Create DataFrame
        df = pd.DataFrame(results_list)
        
        # Save results
        csv_path = os.path.join(shap_output_dir, "classification_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Also save detailed results as JSON
        json_path = os.path.join(shap_output_dir, "classification_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        return df
    
    return pd.DataFrame()

def main():
    """Main function to run SHAP analysis on proposals."""
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run SHAP analysis on RPN proposals")
    parser.add_argument("--proposals_dir", default="output/proposals",
                      help="Directory containing proposal outputs from get_proposals.py")
    parser.add_argument("--output_dir", default="output/shap_analysis",
                      help="Directory to save SHAP analysis results")
    parser.add_argument("--num_shap_samples", type=int, default=50,
                      help="Number of samples for SHAP computation (more = slower but more accurate)")
    args = parser.parse_args()
    
    # Setup model
    cfg = setup_cfg()
    explainer = ProposalClassificationExplainer(cfg)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all proposal directories
    proposal_dirs = []
    for image_dir in os.listdir(args.proposals_dir):
        image_path = os.path.join(args.proposals_dir, image_dir)
        if os.path.isdir(image_path):
            for instance_dir in os.listdir(image_path):
                instance_path = os.path.join(image_path, instance_dir)
                if os.path.isdir(instance_path) and os.path.exists(os.path.join(instance_path, "proposal_data.json")):
                    proposal_dirs.append(instance_path)
    
    logger.info(f"Found {len(proposal_dirs)} proposal directories to process")
    
    # Process each proposal directory
    all_results = []
    for proposal_dir in tqdm(proposal_dirs, desc="Processing proposal directories"):
        try:
            df = process_proposal_directory(proposal_dir, explainer)
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            logger.error(f"Error processing {proposal_dir}: {str(e)}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_csv_path = os.path.join(args.output_dir, "all_classification_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        
        # Print summary statistics
        logger.info(f"\n=== Classification Summary ===")
        logger.info(f"Total proposals analyzed: {len(combined_df)}")
        
        for class_name in CLASSES_OF_INTEREST:
            logger.info(f"\n{class_name} class:")
            
            # Classification scores
            if f'{class_name}_score' in combined_df.columns:
                class_scores = combined_df[f'{class_name}_score']
                logger.info(f"  Classification scores:")
                logger.info(f"    Average: {class_scores.mean():.3f}")
                logger.info(f"    Max: {class_scores.max():.3f}")
                logger.info(f"    Min: {class_scores.min():.3f}")
                logger.info(f"    Proposals with score > 0.5: {(class_scores > 0.5).sum()}")
                logger.info(f"    Proposals with score > 0.8: {(class_scores > 0.8).sum()}")
            
            # SHAP values
            shap_mean_col = f'{class_name}_shap_mean'
            shap_abs_mean_col = f'{class_name}_shap_abs_mean'
            
            if shap_mean_col in combined_df.columns:
                shap_means = combined_df[shap_mean_col].dropna()
                shap_abs_means = combined_df[shap_abs_mean_col].dropna()
                
                if len(shap_means) > 0:
                    logger.info(f"  SHAP values:")
                    logger.info(f"    Average SHAP mean: {shap_means.mean():.6f}")
                    logger.info(f"    Average |SHAP| mean: {shap_abs_means.mean():.6f}")
                    logger.info(f"    Proposals with positive SHAP mean: {(shap_means > 0).sum()}")
                    logger.info(f"    Proposals with negative SHAP mean: {(shap_means < 0).sum()}")
    else:
        logger.warning("No results to combine")
    
    logger.info(f"\nAnalysis complete. Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())