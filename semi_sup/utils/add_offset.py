"""
Add ID offsets to a COCO-format pseudo-annotations JSON.

- Loads the pseudo-label JSON
- Applies offsets to image IDs and annotation IDs
- Ensures each annotation has valid bbox, area, and iscrowd
- Saves the updated JSON to a new file
"""

import json
from pathlib import Path
from typing import Union

def add_offsets_to_coco(
    input_json: Union[str, Path],
    output_json: Union[str, Path],
    image_id_offset: int = 0,
    ann_id_offset: int = 0) -> None:
    
    # Load
    with open(input_json, "r") as f:
        coco = json.load(f)

    # Validate keys
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"Invalid COCO JSON: missing '{key}'")

    # Build id map for images
    id_map = {}
    for img in coco["images"]:
        old_id = int(img["id"])
        new_id = old_id + image_id_offset
        img["id"] = new_id
        id_map[old_id] = new_id

    # Update annotations
    anns_new = []
    for ann in coco["annotations"]:
        if "bbox" not in ann or len(ann["bbox"]) != 4:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue

        # Fix area if missing/invalid
        if "area" not in ann or ann["area"] <= 0:
            ann["area"] = float(w) * float(h)

        # Default iscrowd
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0

        # Update IDs
        ann["id"] = int(ann["id"]) + ann_id_offset
        ann["image_id"] = id_map.get(int(ann["image_id"]), ann["image_id"])
        anns_new.append(ann)

    coco["annotations"] = anns_new

    # Save updated JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    # Print summary
    print("Offset applied successfully!")
    print(f"Images updated      : {len(coco['images'])}")
    print(f"Annotations updated : {len(coco['annotations'])}")
    print(f"Image ID offset     : {image_id_offset}")
    print(f"Annotation ID offset: {ann_id_offset}")
    print(f"Saved to            : {output_json}")


if __name__ == "__main__":
    # Example usage (edit paths + offsets as needed)
    INPUT = Path("data") / "pseudo_labels" / "zerowaste-s_ensemble_consensus_pseudo_annotations.json"
    OUTPUT = Path("data") / "pseudo_labels" / "zerowaste-s_ensemble_consensus_pseudo_annotations_offset.json"

    add_offsets_to_coco(
        input_json=INPUT,
        output_json=OUTPUT,
        image_id_offset=100000,  # adjust as needed
        ann_id_offset=1000000,    # adjust as needed
    )