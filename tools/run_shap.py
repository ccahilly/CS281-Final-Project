import os
import cv2
import torch
import shap
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import default_setup
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


cfg = setup_cfg()
setup_logger()
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()

# --- Load and transform image ---
img_path = "datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png"
image = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB

transform_gen = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
    cfg.INPUT.MAX_SIZE_TEST
)
transform = transform_gen.get_transform(image)
image_transformed = transform.apply_image(image)
height, width = image_transformed.shape[:2]

# Convert to tensor
image_tensor = torch.as_tensor(image_transformed.copy().transpose(2, 0, 1)).float().to(cfg.MODEL.DEVICE)

# Wrap as Detectron2 input format
inputs = [{
    "image": image_tensor,
    "height": height,
    "width": width,
    "file_name": img_path
}]

# --- Forward pass ---
with torch.no_grad():
    outputs = model(inputs)

instances = outputs[0]["instances"]
if len(instances) == 0:
    raise ValueError("No instances detected in image.")

# --- Define wrapped model with class names ---
class WrappedModel(torch.nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes

    def forward(self, x):
        outputs = self.model([{
            "image": x[i],
            "height": x.shape[2],
            "width": x.shape[3],
            "file_name": f"shap_injected_{i}"
        } for i in range(x.shape[0])])
        return torch.stack([
            out["instances"].all_scores[0] for out in outputs
        ])

wrapped_model = WrappedModel(model, cfg)

# --- SHAP for "rider" and "person" ---
image_tensor.requires_grad_()
explainer = shap.GradientExplainer(wrapped_model, image_tensor.unsqueeze(0))

target_classes = ["rider", "person"]
target_class_indices = [wrapped_model.class_names.index(c) for c in target_classes]

print("🚀 Computing SHAP for selected classes...")

for class_idx in target_class_indices:
    print(f"🔍 Explaining class {class_idx} ({wrapped_model.class_names[class_idx]})...")
    # Run forward pass
    output = wrapped_model(image_tensor.unsqueeze(0))
    
    # Zero grads and backprop on just this class score
    model.zero_grad()
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()

    class_score = output[0, class_idx]
    class_score.backward(retain_graph=True)

    # SHAP computes attribution for that class
    # shap_vals = explainer.shap_values(image_tensor.unsqueeze(0))
    shap_vals = explainer.shap_values(image_tensor.unsqueeze(0), nsamples=10)
    shap.image_plot(shap_vals, np.array([image_transformed / 255.0]))

print("✅ Done.")