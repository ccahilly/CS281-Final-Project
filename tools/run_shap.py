import os
import cv2
import torch
import shap
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.structures import pairwise_iou, Boxes

import torch.nn.functional as F


# --- Load config and model ---
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("configs/Cityscapes/shap_mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.ROI_HEADS.NAME = "ROIHeadsWithAllScores"
    cfg.MODEL.WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = cfg.TEST.DETECTIONS_PER_IMAGE
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

class WrappedModel(torch.nn.Module):
    def __init__(self, model, cfg, target_box_tensor):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.target_box = target_box_tensor
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes

    def _find_matching_instance(self, instances, iou_threshold=0.5):
        ious = pairwise_iou(instances.pred_boxes, Boxes(self.target_box))
        max_iou, idx = ious.max(dim=0)
        if max_iou > iou_threshold:
            print(f"Matched IoU: {max_iou.item():.3f}")
            return int(idx.item())
        else:
            raise ValueError(f"No matching instance found with IoU > {iou_threshold:.2f}")

    def forward(self, x):
        input_dicts = [{
            "image": x[i],
            "height": x.shape[2],
            "width": x.shape[3],
            "file_name": f"shap_{i}"
        } for i in range(x.shape[0])]

        with torch.enable_grad():
            images = self.model.preprocess_image(input_dicts)
            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features, None)
            instances, _ = self.model.roi_heads(images, features, proposals)

        instances = instances[0]
        print(f"Instances found: {len(instances)}")

        matched_idx = self._find_matching_instance(instances)
        print(f"Box of matched instance {matched_idx}: {instances[matched_idx].pred_boxes.tensor.detach().cpu().numpy()}")

        logits = instances.all_logits[matched_idx][:-1]

        for i in range(len(logits)):
            print(f"Class {self.class_names[i]}: {logits[i].item():.3f}")
        print("requires_grad?", logits.requires_grad)
        print("Logits shape:", logits.shape)
        return logits.unsqueeze(0)
        # print(logits[self.class_names.index("person")].view(1))
        # return logits[self.class_names.index("person")].view(1, 1)ok
        

# --- Main SHAP logic ---
def compute_shap_for_row(cfg, model, row, use_masked_baseline=False):
    file_name = row["file_name"]
    gt_class = row["gt_class"]
    pred_class = row["pred_class"]
    pred_instance_idx = int(row["pred_instance_idx"])

    assert gt_class in ["rider", "person"], f"Invalid gt_class: {gt_class}"
    assert pred_class in ["rider", "person"], f"Invalid pred_class: {pred_class}"

    class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
    class_idx = class_names.index(pred_class)

    print(f"\U0001F4C2 Processing: {file_name} | Instance: {pred_instance_idx} | Class: {pred_class}")

    image = cv2.imread(file_name)[:, :, ::-1]
    transform_gen = T.ResizeShortestEdge([
        cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST
    ], cfg.INPUT.MAX_SIZE_TEST)
    transform = transform_gen.get_transform(image)
    image_transformed = transform.apply_image(image)
    height, width = image_transformed.shape[:2]

    image_tensor = torch.as_tensor(image_transformed.copy().transpose(2, 0, 1)).float().to(cfg.MODEL.DEVICE)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor.requires_grad_()

    if use_masked_baseline:
        print("Using masked baseline over union of GT and predicted boxes")
        blurred = F.avg_pool2d(image_tensor, kernel_size=15, stride=1, padding=7)
        x1 = int(min(row["gt_box_x_min"], row["pred_box_x_min"]))
        y1 = int(min(row["gt_box_y_min"], row["pred_box_y_min"]))
        x2 = int(max(row["gt_box_x_max"], row["pred_box_x_max"]))
        y2 = int(max(row["gt_box_y_max"], row["pred_box_y_max"]))
        baseline = image_tensor.clone()
        baseline[:, :, y1:y2, x1:x2] = blurred[:, :, y1:y2, x1:x2]
        baseline = blurred
        plt.imshow(baseline.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
        plt.title("Baseline")
        plt.axis("off")
        plt.show()
    else:
        baseline = image_tensor

    pred_box = torch.tensor([[row["pred_box_x_min"], row["pred_box_y_min"], row["pred_box_x_max"], row["pred_box_y_max"]]], device=cfg.MODEL.DEVICE).float()

    wrapped_model = WrappedModel(model, cfg, pred_box)
    background = [baseline.clone().detach()]
    explainer = shap.GradientExplainer(wrapped_model, background)

    shap_vals = explainer([image_tensor], nsamples=1)[class_idx]

    # shap_img = shap_vals.values[0].sum(0)
    # shap_img = shap_img.transpose(1, 2, 0)
    # shap_img = np.abs(shap_img).mean(axis=-1)

    # plt.imshow(image_transformed)
    # plt.imshow(shap_img, cmap='jet', alpha=0.5)
    # plt.title(f"SHAP overlay for {pred_class}")
    # plt.axis("off")
    # plt.show()

    shap_img = shap_vals.values[0, 0]  # [H, W]
    plt.imshow(image_transformed)
    plt.imshow(np.abs(shap_img), cmap='jet', alpha=0.5)
    plt.title(f"SHAP overlay for {pred_class}")
    plt.axis("off")
    plt.show()

    img_id = os.path.splitext(os.path.basename(file_name))[0]
    save_name = f"shap_values_{pred_class}_{img_id}_inst{pred_instance_idx}.pkl"
    save_path = os.path.join("output", save_name)
    joblib.dump(shap_vals, save_path)

    del shap_vals
    del wrapped_model
    torch.cuda.empty_cache()
    gc.collect()


# --- CLI + main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked_baseline", action="store_true", help="Use a masked baseline over the union of GT and predicted box")
    args = parser.parse_args()

    cfg = setup_cfg()
    setup_logger()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    df = pd.read_csv("output/misclassification_analysis_mini/fn_rider_person_details.csv")
    os.makedirs("output", exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            compute_shap_for_row(cfg, model, row, use_masked_baseline=args.masked_baseline)
        except Exception as e:
            print(f"❌ Error on row {row['file_name']}: {e}")
