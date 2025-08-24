import os
import sys
import json
import argparse
import warnings
import yaml
import contextlib

import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add GroundingDINO to PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "external_modules", "GroundingDINO"))

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.util import box_ops, get_tokenlizer

warnings.filterwarnings("ignore")

CLASS_ORDER = [1, 3, 2, 0]  # rigid plastic, cardboard, metal, soft plastic


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser("Zero-shot OVOD eval with GroundingDINO")
    p.add_argument("--config", default="configs/groundingdino.yaml", help="Path to groundingdino config YAML")
    p.add_argument("--batch-size", type=int, help="Override eval.batch_size")
    p.add_argument("--num-workers", type=int, help="Override eval.num_workers")
    p.add_argument("--num-select", type=int, help="Override eval.num_select")
    p.add_argument("--device", type=str, help="Override device (cpu or cuda)")
    return p.parse_args()


def load_model(cfg):
    """Load and return the pretrained GroundingDINO model from checkpoint."""
    args = SLConfig.fromfile(cfg["model"]["config"])
    args.device = cfg["device"]
    model = build_model(args)
    ckpt = torch.load(cfg["model"]["checkpoint"], map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    return model.to(cfg["device"]).eval()


class CocoDetection(torchvision.datasets.CocoDetection):
    """Custom COCO Detection class that applies transforms and extracts boxes."""

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        boxes = torch.as_tensor([obj["bbox"] for obj in target], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        item = {
            "image_id": self.ids[idx],
            "boxes": boxes[keep],
            "orig_size": torch.tensor([h, w], dtype=torch.int)
        }
        if self._transforms:
            img, item = self._transforms(img, item)
        return img, item


class PostProcessMyDataset(torch.nn.Module):
    """
    Post-processing module for Grounding DINO output specific to this dataset.
    Selects top-k predictions using positive maps derived from prompts.
    """

    def __init__(self, num_select, prompts, tokenizer):
        super().__init__()
        self.num_select = num_select
        captions, cat2span = build_captions_and_token_span(prompts, True)
        pos_map = create_positive_map_from_span(
            tokenizer(captions),
            [cat2span[c] for c in prompts]
        )
        dim = pos_map.shape[1]
        new_map = torch.zeros((5, dim))
        for k, v in {0: 1, 1: 2, 2: 3, 3: 4}.items():
            new_map[v] = pos_map[k]
        self.positive_map = new_map

    @torch.no_grad()
    def forward(self, outputs, sizes):
        """Select top predictions and rescale boxes to original image size."""
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        prob = logits.sigmoid() @ self.positive_map.to(logits.device).T
        topv, topi = torch.topk(prob.view(logits.size(0), -1), self.num_select, dim=1)
        q_idx = topi // prob.size(2)
        labels = topi % prob.size(2)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = torch.gather(boxes, 1, q_idx.unsqueeze(-1).repeat(1, 1, 4))

        h, w = sizes.unbind(1)
        scale = torch.stack([w, h, w, h], dim=1)
        boxes = boxes * scale.unsqueeze(1)

        results = []
        for b in range(logits.size(0)):
            results.append({
                "scores": topv[b],
                "labels": labels[b],
                "boxes": boxes[b]
            })
        return results


def convert_to_coco(preds):
    """Convert predictions to COCO-style JSON format."""
    coco = []
    for img_id, det in preds.items():
        for score_t, label_t, box_t in zip(det["scores"], det["labels"], det["boxes"]):
            score = float(score_t)
            label = int(label_t)
            x1, y1, x2, y2 = box_t.tolist()
            coco.append({
                "image_id": img_id,
                "category_id": label,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score
            })
    return coco


def extract_evaluation_metrics(coco_eval):
    """Extract overall and per-category AP/mAP from COCO evaluator."""
    stats = getattr(coco_eval, "stats", None)
    if stats is None or len(stats) < 2:
        overall_map, overall_map50 = 0.0, 0.0
    else:
        overall_map, overall_map50 = float(stats[0]), float(stats[1])


    class_names = ["rigid plastic", "cardboard", "metal", "soft plastic"]
    per_cat = {}

    try:
        precision = coco_eval.eval["precision"]
    except Exception:
        precision = None

    for idx, name in enumerate(class_names, start=1):
        if precision is None:
            ap = ap50 = 0.0
        else:
            P_all = precision[:, :, idx - 1, 0, -1]
            P50 = precision[0, :, idx - 1, 0, -1]
            ap = float(np.nanmean(P_all)) if np.isfinite(np.nanmean(P_all)) else 0.0
            ap50 = float(np.nanmean(P50)) if np.isfinite(np.nanmean(P50)) else 0.0
        per_cat[name] = {"ap": ap, "ap50": ap50}

    return {
        "overall_map": overall_map,
        "overall_map50": overall_map50,
        "categories": per_cat
    }


def main():
    """Main execution function for evaluating zero-shot GroundingDINO on COCO-style dataset."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Override config with CLI args
    if args.batch_size: cfg["eval"]["batch_size"] = args.batch_size
    if args.num_workers: cfg["eval"]["num_workers"] = args.num_workers
    if args.num_select: cfg["eval"]["num_select"] = args.num_select
    if args.device: cfg["device"] = args.device

    device_str = args.device or cfg["device"]
    device = torch.device(device_str)

    print(f"Loaded config: {args.config}")
    print(f"Device: {device_str}, batch_size: {cfg['eval']['batch_size']}, "
          f"num_workers: {cfg['eval']['num_workers']}, num_select: {cfg['eval']['num_select']}")

    # Build DataLoader
    print("\n=> Building DataLoader…")
    tfm = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = CocoDetection(cfg["data"]["img_dir"], cfg["data"]["ann_file"], tfm)
    loader = DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["eval"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Load model and tokenizer
    print("=> Loading model…")
    model = load_model(cfg)
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        tokenizer = get_tokenlizer.get_tokenlizer(
            SLConfig.fromfile(cfg["model"]["config"]).text_encoder_type
        )

    all_results = {}

    for name, pset in cfg["prompts"].items():
        print("\n" + "=" * 80)
        print(f"Evaluating prompt set '{name}': {pset['description']}")
        raw = pset["strings"]
        ordered = [raw[i] for i in CLASS_ORDER]
        print("Prompts:", " | ".join(ordered))

        preds = {}
        for imgs, samples in tqdm(loader, desc=f"[{name}]", unit="batch"):
            imgs = imgs.tensors.to(device)
            sizes = torch.stack([s["orig_size"] for s in samples], 0).to(device)
            cap = " . ".join(ordered) + " ."
            outputs = model(imgs, captions=[cap] * imgs.size(0))
            results = PostProcessMyDataset(cfg["eval"]["num_select"], ordered, tokenizer)(outputs, sizes)
            for s, r in zip(samples, results):
                preds[s["image_id"]] = r

        evaluator = CocoGroundingEvaluator(ds.coco, iou_types=("bbox",), useCats=True)
        for img_id, r in preds.items():
            evaluator.update({img_id: r})
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        metrics = extract_evaluation_metrics(evaluator.coco_eval["bbox"])
        out_dir = os.path.join(cfg["eval"]["output_dir"], name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(convert_to_coco(preds), f, indent=2)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{name}] done    mAP={metrics['overall_map']:.3f}")
        print("  Per-category AP:")
        for cat, m in metrics["categories"].items():
            print(f"    {cat:<14}: AP={m['ap']:.3f}, AP50={m['ap50']:.3f}")

        all_results[name] = metrics

    # Final summary table
    print("\nComparative Results:")
    print("=" * 50)
    for name, metrics in sorted(all_results.items(), key=lambda kv: kv[1]["overall_map"], reverse=True):
        desc = cfg["prompts"][name]["description"]
        print(f"\n{name} ({desc}):")
        print(f"  Overall mAP:  {metrics['overall_map']:.3f}")
        print(f"  Overall AP50: {metrics['overall_map50']:.3f}")
        print("  Per-category AP:")
        for cat, m in metrics["categories"].items():
            print(f"    {cat:<14}: AP={m['ap']:.3f}, AP50={m['ap50']:.3f}")


if __name__ == "__main__":
    main()
