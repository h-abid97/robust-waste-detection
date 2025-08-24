"""
Consolidate multiple COCO-style prediction JSONs into a single JSON with model_id tags.

Run directly:
    python scripts/consolidate_preds.py
"""
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

# --- Ensure the repo root is on sys.path so local package imports work ---
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semi_sup.utils.consolidate_coco_annotations import (
    consolidate_predictions,
    confidence_threshold_filter,
    default_filter,
    get_predictions_stats,
    save_json,
)

INPUTS = [
    REPO_ROOT / "semi_sup" / "pseudo_annotations" / "co-detr_swin-l_zerowaste-s_pseudo_annotations.json",
    REPO_ROOT / "semi_sup" / "pseudo_annotations" / "deta_swin-l_zerowaste-s_pseudo_annotations.json",
    REPO_ROOT / "semi_sup" / "pseudo_annotations" / "dino_swin-l_zerowaste-s_pseudo_annotations.json",
    REPO_ROOT / "semi_sup" / "pseudo_annotations" / "gdino-swin-b_zerowaste-s_pseudo_annotations.json",
]

# Confidence threshold for filtering predictions 
THRESHOLD = 0.0 

# Output consolidated file path
OUT_PATH = REPO_ROOT / "data" / "pseudo_labels" / "zerowaste-s_consolidated_pseudo_annotations.json"


def main():
    # Validate inputs
    for p in INPUTS:
        if not Path(p).exists():
            raise FileNotFoundError(f"[ERROR] Input file not found: {p}")

    # Ensure output dir exists
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build filter
    filt = confidence_threshold_filter(THRESHOLD) if THRESHOLD is not None else default_filter

    # Consolidate
    consolidated = consolidate_predictions(INPUTS, filter_fn=filt)
    save_json(consolidated, str(out_path))
    print(f"[OK] Consolidated predictions saved to: {out_path}")

    # Stats
    stats = get_predictions_stats(consolidated)
    print("\n=== Consolidation Statistics ===")
    print(f"Total images        : {stats['total_images']}")
    print(f"Total categories    : {stats['total_categories']}")
    print(f"Total predictions   : {stats['total_predictions']}")
    print("\nPredictions per model:")
    for model_id, count in stats["predictions_per_model"].items():
        print(f"  {model_id:<12} {count}")

    fs = stats.get("filtering_stats", {})
    if fs:
        print("\nFiltering summary:")
        totals = fs.get("total_predictions", {})
        kept = fs.get("kept_predictions", {})
        filtered = fs.get("filtered_out", {})
        for model_id in totals:
            print(f"  {model_id}: total={totals[model_id]}, kept={kept.get(model_id, 0)}, filtered={filtered.get(model_id, 0)}")


if __name__ == "__main__":
    main()