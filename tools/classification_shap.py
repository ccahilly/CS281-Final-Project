#!/usr/bin/env python

import logging
import os
import torch
import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer

# Constants for evaluation configuration
CONFIG_FILE = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
CONFIDENCE_THRESHOLD = 0.5

# Custom dataset paths
CUSTOM_DATASET_ROOT = "/Users/carolinecahilly/Documents/cs281/CS281-Final-Project/datasets/cityscapes"
CUSTOM_IMAGE_DIR = os.path.join(CUSTOM_DATASET_ROOT, "leftImg8bit", "shap")
CUSTOM_GT_DIR = os.path.join(CUSTOM_DATASET_ROOT, "gtFine", "shap")

# Colors for visualization
PERSON_RIDER_COLOR = (0, 255, 0)  # Green for person/rider
OTHER_COLOR = (255, 0, 0)  # Red for other classes

def get_cityscapes_custom_dicts():
    """
    Register a custom subset of Cityscapes for SHAP analysis.
    Similar to the original Cityscapes dataset registration but with custom paths.
    """
    from cityscapesscripts.helpers.labels import labels
    
    # Get class information
    thing_classes = [l.name for l in labels if l.hasInstances and not l.ignoreInEval]
    thing_dataset_id_to_contiguous_id = {
        l.id: idx for idx, l in enumerate(labels) if l.hasInstances and not l.ignoreInEval
    }
    
    # Register metadata
    meta = MetadataCatalog.get("cityscapes_shap")
    meta.set(
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        evaluator_type="cityscapes_instance",
        image_root=CUSTOM_IMAGE_DIR,
        gt_dir=CUSTOM_GT_DIR,
    )
    
    # Get image files from all city subdirectories
    dataset_dicts = []
    for city in os.listdir(CUSTOM_IMAGE_DIR):
        city_img_dir = os.path.join(CUSTOM_IMAGE_DIR, city)
        city_gt_dir = os.path.join(CUSTOM_GT_DIR, city)
        
        if not os.path.isdir(city_img_dir):
            continue
            
        for image_file in os.listdir(city_img_dir):
            if not image_file.endswith("_leftImg8bit.png"):
                continue
                
            image_path = os.path.join(city_img_dir, image_file)
            
            # Get corresponding ground truth file
            gt_file = image_file.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
            gt_path = os.path.join(city_gt_dir, gt_file)
            
            if not os.path.exists(gt_path):
                continue
            
            record = {}
            record["file_name"] = image_path
            record["image_id"] = os.path.basename(image_file)
            record["sem_seg_file_name"] = gt_path
            dataset_dicts.append(record)
    
    return dataset_dicts

def register_custom_dataset():
    """Register the custom dataset and its metadata."""
    # First, remove if already registered
    if "cityscapes_shap" in DatasetCatalog:
        DatasetCatalog.remove("cityscapes_shap")
    if "cityscapes_shap" in MetadataCatalog:
        MetadataCatalog.remove("cityscapes_shap")
    
    # Register the dataset
    DatasetCatalog.register("cityscapes_shap", get_cityscapes_custom_dicts)
    
    # Get the dataset to trigger metadata registration
    get_cityscapes_custom_dicts()

class DetectionVisualizer:
    def __init__(self, dataset_name):
        self.metadata = MetadataCatalog.get(dataset_name)
        self.model = self._build_model()
        self._logger = logging.getLogger(__name__)
        
        # Create output directory
        self._output_dir = os.path.join("detection_visualization")
        os.makedirs(self._output_dir, exist_ok=True)
        
        # Setup image transform
        self.transform = T.ResizeShortestEdge(
            [800, 800], 1333
        )
        
        # Define classes of interest
        self.classes_of_interest = ["person", "rider"]
    
    def _build_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(CONFIG_FILE)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
        cfg.MODEL.WEIGHTS = WEIGHTS
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.metadata.thing_classes)
        cfg.freeze()
        
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()
        return model
    
    def visualize_detections(self, image_path):
        """Visualize all detections in an image."""
        # Read image
        original_image = cv2.imread(image_path)
        if original_image is None:
            self._logger.warning(f"Could not read image: {image_path}")
            return
        
        self._logger.info(f"Processing image: {image_path}")
        self._logger.info(f"Image shape: {original_image.shape}")
        
        # Convert to RGB for model
        image_rgb = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        
        # Create visualization image (copy of original)
        vis_image = original_image.copy()
        
        # Transform image for model
        image = self.transform.get_transform(image_rgb).apply_image(image_rgb)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        
        # Prepare inputs
        inputs = {"image": image, "height": height, "width": width}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model([inputs])[0]
        
        # Get instances above confidence threshold
        instances = outputs["instances"]
        
        # Log basic detection info
        self._logger.info(f"Found {len(instances)} instances")
        
        # Create Visualizer instance
        v = Visualizer(image_rgb, self.metadata)
        
        # Draw all detections
        detection_count = 0
        person_rider_count = 0
        
        # Save crops directory
        crops_dir = os.path.join(self._output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        
        for idx, (box, class_idx, score) in enumerate(zip(
            instances.pred_boxes,
            instances.pred_classes,
            instances.scores
        )):
            if score > CONFIDENCE_THRESHOLD:
                detection_count += 1
                class_name = self.metadata.thing_classes[class_idx]
                box = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Only process person/rider classes
                is_person_rider = class_name in self.classes_of_interest
                if is_person_rider:
                    person_rider_count += 1
                    
                    # Ensure box coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Only save crop if the box has valid dimensions
                    if x2 > x1 and y2 > y1:
                        # Extract and save crop
                        crop = original_image[y1:y2, x1:x2]
                        
                        # Save both the original crop and a resized version for visualization
                        crop_path = os.path.join(
                            crops_dir,
                            f"crop_instance_{idx}_class_{class_name}.png"
                        )
                        cv2.imwrite(crop_path, crop)
                        
                        # Draw box and label on visualization image
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), PERSON_RIDER_COLOR, 2)
                        
                        # Add label with class name and score
                        label = f"{class_name}: {score:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw label background
                        cv2.rectangle(vis_image, (x1, y1 - label_height - 5), (x1 + label_width, y1), PERSON_RIDER_COLOR, -1)
                        
                        # Draw label text
                        cv2.putText(vis_image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
                        
                        self._logger.info(f"Detection {idx}: {class_name} (score: {score:.3f}) at box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                else:
                    # For non-person/rider classes, just draw in red
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), OTHER_COLOR, 2)
        
        self._logger.info(f"Total detections above threshold: {detection_count}")
        self._logger.info(f"Person/rider detections: {person_rider_count}")
        
        # Save visualization
        output_path = os.path.join(
            self._output_dir,
            f"detections_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, vis_image)
        self._logger.info(f"Saved annotated image to: {output_path}")
        
        return outputs

def main():
    logger = setup_logger()
    
    # Register custom dataset and metadata
    register_custom_dataset()
    
    # Initialize visualizer
    visualizer = DetectionVisualizer("cityscapes_shap")
    
    # Process all images in the custom dataset
    dataset_dicts = DatasetCatalog.get("cityscapes_shap")
    for d in dataset_dicts:
        logger.info(f"Processing {d['file_name']}")
        visualizer.visualize_detections(d["file_name"])
    
    # Unregister the dataset
    DatasetCatalog.remove("cityscapes_shap")
    MetadataCatalog.remove("cityscapes_shap")

if __name__ == "__main__":
    main() 