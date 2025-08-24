import json
from collections import defaultdict
from typing import List, Dict, Any, Callable, Optional

__all__ = [
    "load_coco_json",
    "default_filter",
    "confidence_threshold_filter",
    "consolidate_predictions",
    "get_predictions_stats",
    "save_json",
]


def load_coco_json(file_path: str) -> Dict[str, Any]:
    """Load a COCO-style JSON file.

    Args:
        file_path: Path to the COCO JSON.

    Returns:
        Parsed JSON as a Python dict.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], file_path: str) -> None:
    """Save a Python dict to JSON with indentation.

    Args:
        obj: The dictionary to save.
        file_path: Output path for the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=2)


def default_filter(prediction: Dict[str, Any]) -> bool:
    """Default filter that keeps all predictions.

    Args:
        prediction: A single prediction (annotation) dict.

    Returns:
        True (keeps everything).
    """
    return True


def confidence_threshold_filter(threshold: float) -> Callable[[Dict[str, Any]], bool]:
    """Create a filter function with a confidence threshold.

    Args:
        threshold: Minimum score required to keep a prediction.

    Returns:
        A function that returns True if prediction['score'] >= threshold.
    """
    def filter_fn(prediction: Dict[str, Any]) -> bool:
        return prediction.get("score", 0.0) >= threshold
    return filter_fn


def consolidate_predictions(
    json_files: List[str],
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """Consolidate predictions from multiple COCO-style JSONs that share the same images.

    Each prediction gets a `model_id` (e.g., "model_0", "model_1", ...).
    Ensures required COCO fields (id, area, iscrowd, bbox as floats, etc.).
    Tracks filtering statistics per model.

    Args:
        json_files: List of paths to COCO JSON files.
        filter_fn: Optional function that takes a prediction dict and returns bool.
                   If provided, only predictions where filter_fn returns True are kept.

    Returns:
        Dictionary in COCO format with combined predictions and `filtering_stats`.
        Keys: images, categories, annotations, filtering_stats
    """
    filter_fn = filter_fn or default_filter

    # Load base structure (images, categories) from the first file
    base_coco = load_coco_json(json_files[0])

    consolidated: Dict[str, Any] = {
        "images": base_coco.get("images", []),
        "categories": base_coco.get("categories", []),
        "annotations": [],
    }

    filtered_counts = defaultdict(int)   # kept per model
    total_counts = defaultdict(int)      # total per model

    next_ann_id = 1

    # Append predictions from all models
    for model_idx, json_file in enumerate(json_files):
        coco_data = load_coco_json(json_file)
        model_id = f"model_{model_idx}"

        anns = coco_data.get("annotations", [])
        for ann in anns:
            total_counts[model_id] += 1

            prediction = ann.copy()

            # Ensure unique ID (avoid clashing across files)
            prediction["id"] = next_ann_id
            next_ann_id += 1

            # Tag with model_id
            prediction["model_id"] = model_id

            # Ensure required COCO-ish fields
            # bbox as [x, y, w, h] floats
            bbox = prediction.get("bbox", [0, 0, 0, 0])
            prediction["bbox"] = [float(x) for x in bbox]

            # area
            if "area" not in prediction:
                x, y, w, h = prediction["bbox"]
                prediction["area"] = float(w * h)
            else:
                prediction["area"] = float(prediction["area"])

            # iscrowd
            if "iscrowd" not in prediction:
                prediction["iscrowd"] = 0

            # score
            prediction["score"] = float(prediction.get("score", 1.0))

            # ids (ensure ints)
            prediction["image_id"] = int(prediction["image_id"])
            prediction["category_id"] = int(prediction["category_id"])

            # Apply filter
            if filter_fn(prediction):
                consolidated["annotations"].append(prediction)
                filtered_counts[model_id] += 1

    # Filtering statistics
    consolidated["filtering_stats"] = {
        "total_predictions": dict(total_counts),
        "kept_predictions": dict(filtered_counts),
        "filtered_out": {
            model_id: total_counts[model_id] - filtered_counts[model_id]
            for model_id in total_counts
        },
    }

    return consolidated


def get_predictions_stats(consolidated: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics about the consolidated predictions.

    Args:
        consolidated: The dictionary returned by `consolidate_predictions`.

    Returns:
        A dictionary of stats: totals, per-model counts, per-image counts, and filtering stats.
    """
    predictions_per_image = defaultdict(lambda: defaultdict(int))
    predictions_per_model = defaultdict(int)

    for ann in consolidated.get("annotations", []):
        image_id = ann["image_id"]
        model_id = ann["model_id"]
        predictions_per_image[image_id][model_id] += 1
        predictions_per_model[model_id] += 1

    return {
        "total_images": len(consolidated.get("images", [])),
        "total_categories": len(consolidated.get("categories", [])),
        "total_predictions": len(consolidated.get("annotations", [])),
        "predictions_per_model": dict(predictions_per_model),
        "filtering_stats": consolidated.get("filtering_stats", {}),
        "predictions_per_image": {
            image_id: dict(model_counts)
            for image_id, model_counts in predictions_per_image.items()
        },
    }