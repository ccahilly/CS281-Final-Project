#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2
import logging
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from cityscapesscripts.helpers.labels import labels

# Constants for visualization
INSTANCE_COLOR = (0.0, 1.0, 0.0)  # Green for visualization
TP_COLOR = (0.0, 1.0, 0.0)  # Green
FP_COLOR = (1.0, 1.0, 0.0)  # Yellow
FN_COLOR = (1.0, 0.0, 0.0)  # Red

def setup_metadata(dataset_name="cityscapes_fine_instance_seg_val"):
    """Set up the metadata and class mapping for Cityscapes dataset."""
    metadata = MetadataCatalog.get(dataset_name)
    classes = metadata.thing_classes
    
    # Map Cityscapes training IDs to class indices
    trainId_to_class = {}
    thing_count = 0
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            # Map both label.id and trainId to handle both formats
            trainId_to_class[label.id] = thing_count
            if hasattr(label, 'trainId') and label.trainId >= 0:
                trainId_to_class[label.trainId] = thing_count
            thing_count += 1
            
    return metadata, classes, trainId_to_class

def visualize_instance(image, box, mask, label, output_path, color=INSTANCE_COLOR, metadata=None):
    """Helper function to visualize a single instance."""
    v = Visualizer(image.copy(), metadata=metadata)
    
    v.draw_box(box, edge_color=color)
    if mask is not None:
        v.draw_binary_mask(mask, color=color, alpha=0.3)
    v.draw_text(label, (box[0], box[1]), color=color)
    
    instance_vis = v.get_output().get_image()
    cv2.imwrite(output_path, instance_vis[:, :, ::-1])  # RGB to BGR for OpenCV
    return instance_vis

def visualize_gt_instances(img_path, output_dir, confidence_threshold=0.5):
    """Visualize ground truth instances in a single image."""
    logger = logging.getLogger(__name__)
    
    # Set up metadata
    metadata, classes, trainId_to_class = setup_metadata()
    
    # Convert image path to GT path
    gt_path = img_path.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
    
    if not os.path.isfile(gt_path):
        logger.error(f"Instance segmentation file not found: {gt_path}")
        return
    
    if not os.path.isfile(img_path):
        logger.error(f"Image file not found: {img_path}")
        return
        
    # Load images
    gt_seg = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    original_img = cv2.imread(img_path)
    if gt_seg is None or original_img is None:
        logger.error(f"Could not read image files: {gt_path} or {img_path}")
        return
    
    original_img_rgb = original_img[:, :, ::-1]  # BGR to RGB
    
    # Log instance counts for debugging
    unique_instances = np.unique(gt_seg)[1:]  # Skip background
    logger.info(f"Found {len(unique_instances)} raw instances in {os.path.basename(gt_path)}")
    
    # Create output directories
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_output_dir = os.path.join(output_dir, img_name)
    gt_instances_dir = os.path.join(img_output_dir, "gt_instances")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(gt_instances_dir, exist_ok=True)
    
    # Extract ground truth instances following the same approach as in eval_model.py
    gt_instances = []
    gt_classes = []
    gt_boxes = []
    all_masks = []
    all_boxes = []
    all_class_labels = []
    skipped_instances = 0
    
    # Create a composite image
    composite_img = original_img_rgb.copy()
    v_composite = Visualizer(composite_img, metadata=metadata)
    
    # Process each instance ID in the ground truth segmentation
    for instance_id in unique_instances:
        # Extract instance mask
        instance_mask = (gt_seg == instance_id)
        
        # In Cityscapes, class ID is instance_id // 1000
        class_id = instance_id // 1000
        
        # Skip if not a valid class
        if class_id not in trainId_to_class:
            skipped_instances += 1
            logger.debug(f"Skipping instance with unknown class_id {class_id} (instance_id: {instance_id})")
            continue
            
        # Extract instance coordinates
        y_indices, x_indices = np.where(instance_mask)
        if len(y_indices) == 0:
            skipped_instances += 1
            logger.debug(f"Skipping empty instance mask for class_id {class_id}")
            continue
                
        # Get bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        box = [x_min, y_min, x_max, y_max]
        
        # Map class_id to detectron2 class index and get class name
        class_idx = trainId_to_class[class_id]
        class_name = classes[class_idx]
        
        # Store instance data
        gt_instances.append(instance_mask)
        gt_classes.append(class_idx)
        gt_boxes.append(box)
        
        # Add to visualization collections
        all_masks.append(instance_mask)
        all_boxes.append(box)
        all_class_labels.append(class_name)
        
        # Save individual instance visualization (optional)
        instance_output_path = os.path.join(gt_instances_dir, f"{class_name}_{len(gt_instances)}.png")
        visualize_instance(original_img_rgb, box, instance_mask, class_name, instance_output_path, metadata=metadata)
    
    if skipped_instances > 0:
        logger.info(f"Skipped {skipped_instances} instances in {os.path.basename(gt_path)}")
    
    if not gt_instances:
        logger.warning(f"No valid instances found in {os.path.basename(gt_path)}")
        return
    
    # Create and save composite visualization with all instances
    v_composite = v_composite.overlay_instances(
        masks=all_masks,
        boxes=np.array(all_boxes) if all_boxes else None,
        labels=all_class_labels if all_class_labels else None
    )
    composite_output = v_composite.get_image()
    composite_output_path = os.path.join(img_output_dir, "all_gt_instances.png")
    cv2.imwrite(composite_output_path, composite_output[:, :, ::-1])  # RGB to BGR for OpenCV
    
    logger.info(f"Ground truth instance visualization saved to: {img_output_dir}")
    logger.info(f"Composite visualization: {composite_output_path}")
    logger.info(f"Individual instance visualizations: {gt_instances_dir}")
    
    return composite_output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize ground truth instances for a Cityscapes image")
    parser.add_argument(
        "--image-path", 
        required=True,
        help="Path to a Cityscapes validation image (leftImg8bit)"
    )
    parser.add_argument(
        "--output-dir", 
        default="output/gt_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold for visualizations"
    )
    
    args = parser.parse_args()
    logger = setup_logger()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = visualize_gt_instances(
        img_path=args.image_path,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold
    )
    
    if output_path:
        logger.info(f"Visualization complete. Output saved to: {output_path}")
    else:
        logger.error("Visualization failed.")

if __name__ == "__main__":
    main()
