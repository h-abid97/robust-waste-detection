import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# ======================================================
# Ensemble-Based Soft Pseudo-Labeling Parameters
# Change these values here to adjust the behavior
# ======================================================
tau    = 0.05   # Initial confidence threshold
theta  = 0.65   # IoU threshold for clustering
m      = 2      # Minimum distinct model agreement
tau_f  = 0.35   # Final soft confidence threshold
alpha  = 0.1    # Spread decay factor
beta   = 0.05   # Model agreement bonus factor
# ======================================================

# -------------------------------
# Step 1: Load the Merged COCO JSON (Ensuring Full COCO Format)
# -------------------------------
def load_coco_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Ensure COCO format: check for required keys
    if "annotations" not in data or "images" not in data or "categories" not in data:
        raise ValueError(f"Invalid COCO JSON format: Missing required keys in {json_path}")
    return data

# Path to your consolidated JSON
merged_coco_path = Path("data") / "pseudo_labels" / "zerowaste-s_consolidated_pseudo_annotations.json"
coco_data = load_coco_json(merged_coco_path)

# Extract components
annotations = coco_data["annotations"]
images = coco_data["images"]
categories = coco_data["categories"]

# -------------------------------
# Step 2: Soft Confidence Filtering (Remove Very Low-Confidence Detections)
# -------------------------------
filtered_annotations = [ann for ann in annotations if ann["score"] >= tau]
print(f"Filtered out low-confidence detections. Remaining: {len(filtered_annotations)}")

# -------------------------------
# Step 3: Organize Bounding Boxes by Image and Class
# -------------------------------
def organize_predictions(annotations):
    predictions_by_image = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        # Ensure required keys exist
        if "image_id" not in ann or "category_id" not in ann or "bbox" not in ann or "score" not in ann:
            continue  # Skip malformed entries
        img_id = ann["image_id"]
        class_id = ann["category_id"]
        predictions_by_image[img_id][class_id].append(ann)
    return predictions_by_image

predictions_by_image = organize_predictions(filtered_annotations)

# -------------------------------
# Step 4: Combined IoU-Based Grouping and Weighted Box Fusion
# -------------------------------
def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0  # Invalid bounding box

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_spread_iou(cluster, fused_box):
    """
    Compute the spread based on IoU.
    For each prediction in the cluster, calculate the IoU with the fused box.
    """
    iou_values = []
    for pred in cluster:
        iou_val = calculate_iou(pred["bbox"], fused_box)
        iou_values.append(iou_val)
    avg_iou = np.mean(iou_values)
    spread = 1 - avg_iou  # lower spread when average IoU is high
    return spread

def consensus_factor(spread, num_models, alpha=alpha, beta=beta):
    """
    Convert the spread into an agreement factor using exponential decay.
    Apply a small bonus based on the number of agreeing models.
    """
    # Basic agreement factor from IoU spread
    agreement = np.exp(-alpha * spread)
    
    # Apply small bonus based on number of agreeing models (diminishing returns)
    # 2 models: 1.0, 3 models: 1.05, 4 models: 1.1
    model_bonus = min(1.1, 1.0 + beta * (num_models - 2))
    
    return agreement * model_bonus

def weighted_box_fusion(cluster):
    """
    Applies Weighted Box Fusion on a cluster of predictions.
    Returns the fused bounding box and a representative score.
    """
    boxes = np.array([pred["bbox"] for pred in cluster])
    scores = np.array([pred["score"] for pred in cluster])
    
    # If only one box is present, return it directly
    if len(cluster) == 1:
        return boxes[0].tolist(), scores[0]
    
    # Compute normalized weights from the scores
    weights = scores / scores.sum()
    # Fuse the bounding boxes using a weighted average
    fused_box = np.dot(weights, boxes)
    
    # Use a blend of max and mean for the base score
    # 70% weight to max score, 30% weight to mean score
    base_score = 0.7 * scores.max() + 0.3 * scores.mean()
    
    return fused_box.tolist(), base_score

def weighted_box_fusion_with_consensus(cluster, alpha=alpha, beta=beta):
    """
    Applies robust box fusion and adjusts the score with a consensus factor.
    """
    # Get the fused box and base score
    fused_box, base_score = weighted_box_fusion(cluster)
    
    # Get number of unique models
    distinct_model_ids = {pred["model_id"] for pred in cluster if "model_id" in pred}
    num_models = len(distinct_model_ids)
    
    # Compute the spread
    spread = compute_spread_iou(cluster, fused_box)
    
    # Calculate consensus factor
    agree_factor = consensus_factor(spread, num_models, alpha, beta)
    
    # Adjust the base score with the consensus factor
    final_score = base_score * agree_factor
    
    return fused_box, final_score

def iou_grouping_and_wbf(predictions, iou_threshold=theta, min_votes=m, alpha=alpha, beta=beta):
    """
    Groups predictions based on IoU and applies Weighted Box Fusion (WBF) within each cluster.
    Only clusters with detections from at least `min_votes` distinct models are kept.
    Assumes each prediction includes a "model_id" key.
    """
    predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)
    
    clusters = []  # Each cluster is a list of predictions
    for pred in predictions:
        bbox = pred["bbox"]
        assigned = False
        # Attempt to assign prediction to an existing cluster using the representative box (highest confidence)
        for cluster in clusters:
            rep_bbox = cluster[0]["bbox"]  # highest confidence box in the cluster
            if calculate_iou(bbox, rep_bbox) >= iou_threshold:
                cluster.append(pred)
                assigned = True
                break
        # If not assigned, start a new cluster with this prediction as the representative
        if not assigned:
            clusters.append([pred])
    
    fused_predictions = []
    for cluster in clusters:
        # Count distinct model IDs in the cluster
        distinct_model_ids = {pred["model_id"] for pred in cluster if "model_id" in pred}
        if len(distinct_model_ids) >= min_votes:
            fused_bbox, fused_score = weighted_box_fusion_with_consensus(cluster, alpha, beta)
            representative = cluster[0]
            fused_predictions.append({
                "image_id": representative["image_id"],
                "category_id": representative["category_id"],
                "bbox": fused_bbox,
                "score": fused_score,
                "model_ids": list(distinct_model_ids),  # Store contributing model IDs
                "num_models": len(distinct_model_ids)   # Store number of agreeing models
            })
    return fused_predictions


# Combine predictions for each image and each class using the combined approach
combined_predictions = []
for img_id, class_dict in predictions_by_image.items():
    for class_id, preds in class_dict.items():
        fused_preds = iou_grouping_and_wbf(preds, iou_threshold=theta, min_votes=m, alpha=alpha, beta=beta)
        combined_predictions.extend(fused_preds)

# -------------------------------
# Step 5: Clip Bounding Boxes to Stay Inside Image Dimensions
# -------------------------------
def clip_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    return [x, y, w, h]

# Create a lookup for image dimensions (using defaults if not provided)
image_dimensions = {img["id"]: (img.get("width", 1920), img.get("height", 1080)) for img in images}
final_predictions = []
annotation_id = 1

for pred in combined_predictions:
    img_id = pred["image_id"]
    img_width, img_height = image_dimensions[img_id]
    # Clip the bounding box so it doesn't exceed image dimensions
    clipped_bbox = clip_bbox(pred["bbox"], img_width, img_height)
    pred["bbox"] = clipped_bbox
    pred["id"] = annotation_id
    annotation_id += 1
    final_predictions.append(pred)

# -------------------------------
# Step 6: Final Confidence Filtering
# -------------------------------
final_predictions = [pred for pred in final_predictions if pred["score"] >= tau_f]

# -------------------------------
# Step 7: Save Final Pseudo-Labels as COCO JSON
# -------------------------------
output_file = Path("data") / "pseudo_labels" / "zerowaste-s_ensemble_consensus_pseudo_annotations.json"
final_coco = {
    "images": images,
    "categories": categories,
    "annotations": final_predictions
}

with open(output_file, "w") as f:
    json.dump(final_coco, f, indent=4)

print(f"Final pseudo-labels saved to {output_file}")