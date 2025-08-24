# Zero-Shot OVOD (Open-Vocabulary Object Detection)

This module reproduces the **zero-shot** experiments on the ZeroWaste dataset using:
- **Grounding DINO (Swin-B)**
- **OWLv2 (ViT-L)**
- **YOLO-World (Large)**

It supports **class-only prompts** and **optimized prompts** (richer textual cues). Results are written to `evaluation_results/`.

---

## Folder Layout

```
zero_shot_ovod/
├─ configs/
│  ├─ groundingdino.yaml
│  ├─ owlv2.yaml
│  └─ yoloworld.yaml
├─ evaluation_results/          # <-- outputs land here
├─ requirements/
│  ├─ groundingdino.txt
│  ├─ owlv2.txt
│  └─ yoloworld.txt
└─ src/
   ├─ groundingdino_eval.py
   ├─ owlv2_eval.py
   └─ yoloworld_eval.py
```

---

## 1) Environment Setup

> Each model has its own requirement file. Use **separate virtual envs** to avoid dependency conflicts.

**Conda (recommended)**

### 1: Clone the repo and change current directory
```bash
git clone https://github.com/h-abid97/robust-waste-detection
cd robust-waste-detection
```

### 2: Install dependencies

### a) Grounding DINO

#### *If you face issues creating the environment for Grounding DINO, visit https://github.com/IDEA-Research/GroundingDINO for help.*

#### 1. Create enivronment and activate it.
```bash
# Grounding DINO
conda create -n grounding-dino python=3.9 -y
conda activate grounding-dino
```

#### 2. Change the current directory to the GroundingDINO folder.
```bash
cd external_modules/GroundingDINO/
```

#### 3. Install the required dependencies in the current directory.
```bash
pip install -e .
```

#### 4. Download pre-trained model weights.
```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ../../..
```

### b) OWLv2
```bash
# OWLv2
conda create -n owlv2 python=3.9 -y
conda activate owlv2
pip install -r zero_shot_ovod/requirements/owlv2.txt
```

### c) YOLO-World
```bash
# YOLO-World
conda create -n yolo-world python=3.10 -y
conda activate yolo-world
pip install -r zero_shot_ovod/requirements/yoloworld.txt
```

> CUDA/PyTorch: ensure your local CUDA & torch versions match your GPU. If you hit import errors, install the official wheels for your CUDA version, then `pip install -r ...` again.

---

## 2) Data & Weights

#### - **Dataset root** (expected by configs): `robust-waste-detection/data/zerowaste-f`
#### - *Make sure your data folder looks like this:*
```
data/
├── zerowaste-f/
│   ├── test/
│   │   ├── data/
│   │   │   └── *.jpg
│   │   └── labels.json
│   ├── train/
│   │   ├── data/
│   │   │   └── *.jpg
│   │   └── labels.json
│   └── val/
│       ├── data/
│       │   └── *.jpg
│       └── labels.json
└── zerowaste-s/
    ├── data/
    │   └── *.jpg
    └── labels.json
```

---

## 3) Config Format

### A) Grounding DINO / OWLv2 (prompts embedded in the config)

```yaml
device: "cuda"

data:
  img_dir: "../data/zerowaste-f/test/data"
  ann_file: "../data/zerowaste-f/test/labels.json"

model:
  config: "../external_modules/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
  checkpoint: "../GroundingDINO/groundingdino/weights/groundingdino_swinb_cogcoor.pth"

eval:
  batch_size: 8
  num_workers: 2
  num_select: 300
  output_dir: "./zero_shot_ovod/evaluation_results/groundingdino"

prompts:
  class_only:
    description: "Baseline"
    strings: ["soft plastic","rigid plastic","metal","cardboard"]

  location_enhanced:
    description: "Strong emphasis on location and sorting context"
    strings: ["plastic bag among waste","plastic container in clutter","metal can on conveyor","cardboard box in sorting"]

  recycling_context:
    description: "Recycling-specific context"
    strings: ["recyclable plastic bag","recyclable plastic container","recyclable metal can","recyclable cardboard box"]

  enhanced_properties:
    description: "Detailed material properties with location"
    strings: ["crumpled translucent plastic bag or wrap","solid rigid plastic container or bottle","shiny reflective metal can or tin","thick brown cardboard box or packaging"]

  combined_success:
    description: "Combines elements from the most successful trials"
    strings: ["flexible plastic bag or wrap","hollow rigid plastic container or bottle","shiny metallic can","stiff brown cardboard box"]
```

> OWLv2 configs mirror this structure with their own `model:` section and `prompts:`.

---

### B) YOLO-World (class-only prompts only)

```yaml
model:
  weights: "yolov8l-world.pt"

data:
  image_dir: "../data/zerowaste-f/test/data"
  labels_path: "../data/zerowaste-f/test/labels.json"

classes: ["soft plastic","rigid plastic","metal","cardboard"]

eval:
  conf_threshold: 0.0
  output_dir: "./zero_shot_ovod/evaluation_results/yoloworld"
```

---

## 4) Running Evaluations

### A) Grounding DINO
```bash
conda activate grounding-dino
python zero_shot_ovod/src/groundingdino_eval.py   --config zero_shot_ovod/configs/groundingdino.yaml
```

### B) OWLv2
```bash
conda activate owlv2
python zero_shot_ovod/src/owlv2_eval.py   --config zero_shot_ovod/configs/owlv2.yaml
```

### C) YOLO-World
```bash
conda activate yolo-world
python zero_shot_ovod/src/yoloworld_eval.py   --config zero_shot_ovod/configs/yoloworld.yaml
```

---

## 5) Outputs

Each run creates: `evaluation_results/<model>/`. Each run outputs for each prompt set:

- `metrics.json` – COCO mAP@[50:95], AP50, AP75, per-class AP
- `predictions.json` – raw detections

---

## 6) Prompt Sets

- For **GroundingDINO/OWLv2**, prompt sets are embedded directly in each model config under the top-level `prompts:` key. Add/edit sets there.
- `YOLO-World` in this repo **only** uses class prompts (as per the paper).