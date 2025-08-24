import os
import json
import torch
import argparse
import warnings
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments for OWLv2 evaluation."""
    p = argparse.ArgumentParser("OWLv2 ZeroWaste Prompt Evaluation")
    p.add_argument("--config", default="configs/owlv2.yaml", help="Path to YAML config")
    p.add_argument("--batch-size", type=int, help="Override eval.batch_size")
    p.add_argument("--num-workers", type=int, help="Override eval.num_workers")
    p.add_argument("--device", type=str, help="Override eval.device")
    p.add_argument("--score-thresh", type=float, help="Override eval.score_threshold")
    return p.parse_args()


class ZeroWastePromptDataset(Dataset):
    """
    Custom Dataset for ZeroWaste evaluation using prompt-based object detection.
    
    Each item includes image metadata, raw image, and tokenized processor inputs.
    """

    def __init__(self, coco_gt, images_dir, processor, text_prompt):
        self.processor = processor
        self.images_dir = images_dir
        self.prompt = text_prompt
        self.image_infos = coco_gt.dataset["images"]

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        info = self.image_infos[idx]
        path = os.path.join(self.images_dir, info["file_name"])
        image = Image.open(path).convert("RGB")
        inputs = self.processor(text=[self.prompt], images=image, return_tensors="pt")
        return {"info": info, "image": image, "inputs": inputs}


def collate_fn(batch):
    """Collate function to batch images and processor inputs."""
    batch = [b for b in batch if b]
    inputs = {k: torch.cat([b["inputs"][k] for b in batch]) for k in batch[0]["inputs"]}
    return {
        "image_infos": [b["info"] for b in batch],
        "images": [b["image"] for b in batch],
        "inputs": inputs
    }


def process_batch(batch, model, processor, score_thresh, category_ids, device):
    """
    Run OWLv2 inference on a batch and return formatted predictions.

    Returns:
        List[dict]: COCO-format prediction dictionaries
    """
    preds = []
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        sizes = torch.tensor([[im.size[1], im.size[0]] for im in batch["images"]], device=device)

        outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=score_thresh
        )

        for res, info in zip(results, batch["image_infos"]):
            for box, score, label_idx in zip(res["boxes"], res["scores"], res["labels"]):
                x0, y0, x1, y1 = box.tolist()
                w, h = x1 - x0, y1 - y0
                cat_id = category_ids[label_idx.item()]
                preds.append({
                    "image_id": int(info["id"]),
                    "category_id": int(cat_id),
                    "bbox": [x0, y0, w, h],
                    "score": float(score)
                })

    return preds


def evaluate_predictions(coco_gt, preds):
    """
    Evaluate predictions using COCO metrics (overall + per category).

    Args:
        coco_gt (COCO): Ground-truth object
        preds (list): List of predictions in COCO format

    Returns:
        dict: Dictionary with overall mAP, AP50 and per-category metrics
    """
    coco_dt = coco_gt.loadRes(preds)
    evaler = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()

    overall_map = float(evaler.stats[0])
    overall_map50 = float(evaler.stats[1])

    per_cat = {}
    for cat in coco_gt.loadCats(coco_gt.getCatIds()):
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            ec = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ec.params.catIds = [cat["id"]]
            ec.evaluate()
            ec.accumulate()
            ec.summarize()

        stats = getattr(ec, "stats", None)
        if stats is None or len(stats) < 2 or not np.isfinite(stats[0]):
            ap, ap50 = 0.0, 0.0
        else:
            ap = float(stats[0])
            ap50 = float(stats[1])

        per_cat[cat["name"]] = {"ap": ap, "ap50": ap50}

    return {
        "overall_map": overall_map,
        "overall_map50": overall_map50,
        "categories": per_cat
    }


def main():
    """Main entry point for OWLv2 prompt-based evaluation on ZeroWaste dataset."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # CLI overrides
    if args.batch_size:
        cfg["eval"]["batch_size"] = args.batch_size
    if args.num_workers:
        cfg["eval"]["num_workers"] = args.num_workers
    if args.device:
        cfg["device"] = args.device
    if args.score_thresh:
        cfg["eval"]["score_threshold"] = args.score_thresh

    print(f"Loaded config: {args.config}")
    print(f"Device: {cfg['device']}, batch_size: {cfg['eval']['batch_size']}, "
          f"num_workers: {cfg['eval']['num_workers']}, "
          f"score_threshold: {cfg['eval']['score_threshold']}")

    # Load model and processor
    device = torch.device(cfg["device"])
    model = Owlv2ForObjectDetection.from_pretrained(cfg["model_name"]).to(device).eval()
    processor = Owlv2Processor.from_pretrained(cfg["model_name"])

    # Load COCO ground-truth
    coco_gt = COCO(cfg["data"]["labels_path"])
    cats = sorted(coco_gt.dataset["categories"], key=lambda c: c["id"])
    category_ids = [c["id"] for c in cats]

    # Prepare root output directory
    root_out = os.path.join(cfg["eval"]["output_dir"], str(cfg["eval"]["score_threshold"]))
    os.makedirs(root_out, exist_ok=True)

    all_results = {}

    # Evaluate each prompt configuration
    for key, info in cfg["prompts"].items():
        raw_prompts = info["strings"]

        # Reorder prompts: [rigid plastic, cardboard, metal, soft plastic]
        ordered_prompts = [
            raw_prompts[1],
            raw_prompts[3],
            raw_prompts[2],
            raw_prompts[0],
        ]

        print("\n" + "=" * 50)
        print(f"Evaluating prompt set '{key}': {info['description']}")
        print("Prompts:", " | ".join(ordered_prompts))

        prompt_out = os.path.join(root_out, key)
        os.makedirs(prompt_out, exist_ok=True)

        ds = ZeroWastePromptDataset(
            coco_gt,
            cfg["data"]["test_images_dir"],
            processor,
            ordered_prompts
        )

        dl = DataLoader(
            ds,
            batch_size=cfg["eval"]["batch_size"],
            num_workers=cfg["eval"]["num_workers"],
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=True
        )

        # Inference
        preds = []
        for batch in tqdm(dl, desc=f"[{key}]"):
            preds.extend(process_batch(
                batch, model, processor,
                cfg["eval"]["score_threshold"],
                category_ids, device
            ))

        # Save predictions and metrics
        with open(os.path.join(prompt_out, "predictions.json"), "w") as f:
            json.dump(preds, f, indent=2)

        metrics = evaluate_predictions(coco_gt, preds)

        with open(os.path.join(prompt_out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{key}] done    mAP={metrics['overall_map']:.3f}\n")
        print("  Per-category AP:")
        for cat_name, vals in metrics["categories"].items():
            print(f"    {cat_name:<14}: AP={vals['ap']:.3f}, AP50={vals['ap50']:.3f}")

        all_results[key] = metrics

    # Save and print consolidated results
    with open(os.path.join(root_out, "all_prompt_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nComparative Results:")
    print("=" * 50)
    for key, res in sorted(all_results.items(), key=lambda x: x[1]["overall_map"], reverse=True):
        desc = cfg["prompts"][key]["description"]
        print(f"\n{key} ({desc}):")
        print(f"  Overall mAP:  {res['overall_map']:.3f}")
        print(f"  Overall AP50: {res['overall_map50']:.3f}")
        print("  Per-category AP:")
        for cat, m in res["categories"].items():
            print(f"    {cat}: AP={m['ap']:.3f}, AP50={m['ap50']:.3f}")


if __name__ == "__main__":
    main()