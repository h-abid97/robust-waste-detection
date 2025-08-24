import os
import sys
import json
import argparse
import warnings
import yaml
import contextlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLOWorld

warnings.filterwarnings("ignore")

# Class name to COCO category ID mapping
CLASS_MAPPING = {
    "rigid plastic": 1,
    "cardboard":     2,
    "metal":         3,
    "soft plastic":  4
}

CLASS_ORDER = ["rigid plastic", "cardboard", "metal", "soft plastic"]


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser("Zero-shot OVOD eval with YOLOWorld")
    p.add_argument("--config", default="configs/yoloworld.yaml", help="Path to YOLOWorld YAML config")
    p.add_argument("--conf-thresh", type=float, help="Override eval.conf_threshold")
    p.add_argument("--device", type=str, help="Override device for YOLOWorld (e.g. cpu, cuda)")
    return p.parse_args()


def extract_evaluation_metrics(coco_eval):
    """
    Extract mAP, AP50, and per-category metrics from COCOeval object.

    Args:
        coco_eval (COCOeval): COCO evaluator object after accumulate()

    Returns:
        dict: overall mAP, AP50, and per-category APs
    """
    stats = getattr(coco_eval, "stats", None)
    if stats is None or len(stats) < 2:
        overall_map, overall_map50 = 0.0, 0.0
    else:
        overall_map, overall_map50 = float(stats[0]), float(stats[1])

    per_cat = {}
    try:
        precision = coco_eval.eval["precision"]
    except Exception:
        precision = None

    for idx, name in enumerate(CLASS_ORDER, start=1):
        if precision is None:
            ap = ap50 = 0.0
        else:
            P_all = precision[:, :, idx - 1, 0, -1]
            P50   = precision[0, :, idx - 1, 0, -1]
            ap    = float(np.nanmean(P_all)) if np.isfinite(np.nanmean(P_all)) else 0.0
            ap50  = float(np.nanmean(P50))   if np.isfinite(np.nanmean(P50))   else 0.0
        per_cat[name] = {"ap": ap, "ap50": ap50}

    return {
        "overall_map":   overall_map,
        "overall_map50": overall_map50,
        "categories":    per_cat
    }


def main():
    """Main entry point for zero-shot evaluation using YOLOWorld."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Override config values from CLI args
    if args.conf_thresh is not None:
        cfg["eval"]["conf_threshold"] = args.conf_thresh

    print(f"Loaded config: {args.config}")
    print(f"Confidence threshold: {cfg['eval']['conf_threshold']}")

    # Initialize YOLOWorld model
    device = args.device if args.device else None
    model = YOLOWorld(cfg["model"]["weights"])
    if device:
        model.device = device
    model.set_classes(cfg["classes"])

    # Prepare output directory
    out_dir = cfg["eval"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Try to load ground-truth COCO annotations
    coco_enabled = True
    try:
        gt_coco = COCO(cfg["data"]["labels_path"])
    except Exception as e:
        print(f"Warning: COCO load failed: {e}")
        coco_enabled = False

    coco_results = []
    results_data = []

    # Inference loop over all images
    for img_name in tqdm(os.listdir(cfg["data"]["image_dir"]), desc="Images"):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(cfg["data"]["image_dir"], img_name)
        image_id = -1

        # Match image with COCO image ID
        if coco_enabled:
            info = next((x for x in gt_coco.dataset["images"]
                         if os.path.basename(x["file_name"]) == img_name), None)
            if info:
                image_id = info["id"]

        # Run YOLOWorld inference
        results = model.predict(
            img_path,
            conf=cfg["eval"]["conf_threshold"],
            save=False
        )

        # Parse detections
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                cls = int(box.cls)
                name = results[0].names[cls]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                results_data.append({
                    "image_name": img_name,
                    "class_name": name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

                if coco_enabled and image_id != -1 and name in CLASS_MAPPING:
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": CLASS_MAPPING[name],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": conf
                    })
        else:
            # No detection case
            results_data.append({
                "image_name": img_name,
                "class_name": "no_detection",
                "confidence": 0.0,
                "bbox": []
            })

    # Save all raw detections to CSV
    pd.DataFrame(results_data).to_csv(
        os.path.join(out_dir, "yoloworld_detections.csv"),
        index=False
    )

    # Evaluate detections using COCO mAP if GT is available
    if coco_enabled and coco_results:
        pred_coco = gt_coco.loadRes(coco_results)
        coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics & save
        metrics = extract_evaluation_metrics(coco_eval)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(coco_results, f, indent=2)

        # Display results
        print(f"\nYOLOWorld eval complete    mAP={metrics['overall_map']:.3f}")
        print("  Per-category AP:")
        for cat, m in metrics["categories"].items():
            print(f"    {cat:<14}: AP={m['ap']:.3f}, AP50={m['ap50']:.3f}")
    else:
        print("No COCO results to evaluate.")


if __name__ == "__main__":
    main()