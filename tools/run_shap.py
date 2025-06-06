import os
import cv2
import torch
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T


# --- Load config and model ---
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("configs/Cityscapes/shap_mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.ROI_HEADS.NAME = "ROIHeadsWithAllScores"
    cfg.MODEL.WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


class WrappedModel(torch.nn.Module):
    def __init__(self, model, cfg, instance_idx):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.instance_idx = instance_idx
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes

    def forward(self, x):
        input_dicts = [{
            "image": x[i],
            "height": x.shape[2],
            "width": x.shape[3],
            "file_name": f"shap_{i}"
        } for i in range(x.shape[0])]

        with torch.set_grad_enabled(True):
            outputs = self.model(input_dicts)  # ← this returns a list of dicts

        instances = outputs[0]["instances"]
        print(f"Instances found: {len(instances)}")
        logits = instances.all_logits[self.instance_idx][:-1]  # [num_classes], drop background
        for i in range(len(logits)):
            print(f"Class {self.class_names[i]}: {logits[i].item():.3f}")

        print("requires_grad?", logits.requires_grad)
        print("Logits shape:", logits.shape)
        return logits.unsqueeze(0)  # shape: [1, num_classes]

# --- Main SHAP logic ---
def compute_shap_for_row(cfg, model, row):
    file_name = row["file_name"]
    gt_class = row["gt_class"]
    pred_class = row["pred_class"]
    pred_instance_idx = int(row["pred_instance_idx"])

    assert gt_class in ["rider", "person"], f"Invalid gt_class: {gt_class}"
    assert pred_class in ["rider", "person"], f"Invalid pred_class: {pred_class}"

    class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
    class_idx = class_names.index(pred_class)

    print(f"📂 Processing: {file_name} | Instance: {pred_instance_idx} | Class: {pred_class}")

    # Load and transform image
    image = cv2.imread(file_name)[:, :, ::-1]  # BGR to RGB
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST
    )
    transform = transform_gen.get_transform(image)
    image_transformed = transform.apply_image(image)
    height, width = image_transformed.shape[:2]

    image_tensor = torch.as_tensor(image_transformed.copy().transpose(2, 0, 1)).float().to(cfg.MODEL.DEVICE)
    image_tensor.requires_grad_()

    # SHAP for this instance + class
    wrapped_model = WrappedModel(model, cfg, pred_instance_idx)
    explainer = shap.GradientExplainer(wrapped_model, image_tensor.unsqueeze(0))

    output = wrapped_model(image_tensor.unsqueeze(0))
    
    model.zero_grad()
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()

    shap_vals = explainer.shap_values(
        image_tensor.unsqueeze(0),
        ranked_outputs=1,
        output_rank_order='max',
        nsamples=10
    )

    # Save results
    img_id = os.path.splitext(os.path.basename(file_name))[0]
    save_name = f"shap_values_{pred_class}_{img_id}_inst{pred_instance_idx}.npz"
    save_path = os.path.join("output", save_name)
    for shap_val in shap_vals:
        print(f"SHAP value shape: {np.array(shap_val).shape}")
    np.savez_compressed(
        save_path,
        shap_values=np.array(shap_vals[0]),
        input_image=image_transformed,
        class_name=pred_class,
        image_path=file_name,
        instance_idx=pred_instance_idx,
        gt_class=gt_class
    )
    print(f"✅ Saved to: {save_path}")


if __name__ == "__main__":
    cfg = setup_cfg()
    setup_logger()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    df = pd.read_csv("output/misclassification_analysis/fn_person_person_details.csv")
    os.makedirs("output", exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            compute_shap_for_row(cfg, model, row)
        except Exception as e:
            print(f"❌ Error on row {row['file_name']}: {e}")