import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger


def roi_classifier_fn(model, box_features):
    """
    Takes ROI features (tensor: N x D), returns softmax probabilities: N x C
    """
    logits = model.roi_heads.box_predictor(box_features)
    probs = F.softmax(logits, dim=1)
    return probs


def main():
    # --- Config ---
    cfg = get_cfg()
    cfg.merge_from_file("configs/Cityscapes/shap_mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.ROI_HEADS.NAME = "ROIHeadsWithAllScores"
    cfg.MODEL.WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    setup_logger()

    # --- Model ---
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # --- Data ---
    dataset_name = cfg.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(dataset_name)
    dataloader = build_detection_test_loader(cfg, dataset_name)

    results = []

    for batch in tqdm(dataloader, desc="Processing validation set"):
        with torch.no_grad():
            outputs = model(batch)

        for input_im, output in zip(batch, outputs):
            file_name = input_im["file_name"]
            instances = output["instances"]

            if len(instances) == 0:
                continue

            # Get features and pooled ROIs
            feature_maps = [model.backbone[f] for f in model.roi_heads.box_in_features]
            proposal_boxes = [instances.pred_boxes for _ in range(len(feature_maps))]
            box_features = model.roi_heads.box_pooler(feature_maps, [instances.pred_boxes])
            box_features = model.roi_heads.box_head(box_features)

            softmax_probs = roi_classifier_fn(model, box_features)

            for i in range(len(instances)):
                probs = softmax_probs[i].cpu().numpy()
                pred_class = instances.pred_classes[i].item()
                pred_score = instances.scores[i].item()
                bbox = instances.pred_boxes[i].tensor.cpu().numpy().tolist()[0]

                result = {
                    "file_name": file_name,
                    "instance_idx": i,
                    "pred_class": pred_class,
                    "pred_score": pred_score,
                    "bbox": bbox
                }

                for cls_idx, p in enumerate(probs):
                    result[f"class_{cls_idx}_score"] = p

                results.append(result)

    # --- Save ---
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("output/all_class_softmax_scores_val.csv", index=False)
    print("✅ Saved softmax scores to output/all_class_softmax_scores_val.csv")


if __name__ == "__main__":
    main()
