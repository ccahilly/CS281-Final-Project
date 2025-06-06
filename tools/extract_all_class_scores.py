import os
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import pandas as pd
from tqdm import tqdm


def main():
    # --- Setup ---
    cfg = get_cfg()
    cfg.merge_from_file("configs/Cityscapes/shap_mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.ROI_HEADS.NAME = "ROIHeadsWithAllScores"
    cfg.MODEL.WEIGHTS = "detectron2://Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # To get all detections
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    setup_logger()

    # --- Model ---
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # --- Data ---
    dataset_name = cfg.DATASETS.TEST[0]  # Your val set
    metadata = MetadataCatalog.get(dataset_name)
    dataloader = build_detection_test_loader(cfg, dataset_name)

    # --- Score Extraction ---
    results = []

    for batch in tqdm(dataloader, desc="Processing validation set"):
        with torch.no_grad():
            outputs = model(batch)

        for input_im, output in zip(batch, outputs):
            file_name = input_im["file_name"]
            instances = output["instances"]

            for i in range(len(instances)):
                scores = instances.all_scores[i].cpu().numpy()
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

                # Add all class scores
                for cls_idx, cls_score in enumerate(scores):
                    result[f"class_{cls_idx}_score"] = cls_score

                results.append(result)

    # --- Save to CSV ---
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/all_class_scores_val.csv", index=False)
    print("✅ Saved all class scores to output/all_class_scores_val.csv")


if __name__ == "__main__":
    main()