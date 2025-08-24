# Robust and Label-Efficient Deep Waste Detection (BMVC'25)

> [**Robust and Label-Efficient Deep Waste Detection**](/media/TO_BE_UPDATED.md)<br><br>
> [Hassan Abid](https://scholar.google.com/citations?user=0kaOLSgAAAAJ&hl=en),
[Khan Muhammad](https://scholar.google.com/citations?user=k5oUZyQAAAAJ&hl=en), and
[Muhammad Haris](https://scholar.google.com/citations?user=ZgERfFwAAAAJ&hl=en)


<!-- [![page](https://img.shields.io/badge/Project-Page-F9D371)](/media/TO_BE_UPDATED.md) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](/media/bird_whisperer_interspeech2024.pdf)
[![poster](https://img.shields.io/badge/Presentation-Poster-blue)](/media/bird_whisperer_poster.pdf)

<p align="center"><img src="https://i.imgur.com/waxVImv.png" alt="Image"></p>

<hr />

| ![main figure](/media/GDINO_annot_comparison.png)|
|:--| 
| <p align="justify">We introduce a label-efficient framework for waste detection on the industrial ZeroWaste dataset that unifies zero-shot open-vocabulary evaluation, strong supervised baselines, and semi-supervised learning. Optimizing prompts nearly doubles zero-shot mAP for OVOD models, while fine-tuning modern detectors establishes a 51.6 mAP baseline. An ensemble-based soft pseudo-labeling pipeline (WBF + consensus weighting) then exploits unlabeled data to surpass full supervision (up to 54.3 mAP, +2.7–3.7 mAP). Together, these components provide a reproducible benchmark and a scalable annotation strategy for real-world material-recovery facilities.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Effective waste sorting is critical for sustainable recycling, yet AI research in this domain continues to lag behind commercial systems due to limited datasets and reliance on legacy object detectors. In this work, we advance AI-driven waste detection by establishing strong baselines and introducing an ensemble-based semi-supervised learning framework. We first benchmark state-of-the-art Open-Vocabulary Object Detection (OVOD) models on the real-world ZeroWaste dataset, demonstrating that while class-only prompts perform poorly, LLM-optimized prompts significantly enhance zero-shot accuracy. Next, to address domain-specific limitations, we fine-tune modern transformer-based detectors, achieving a new baseline of 51.6 mAP. We then propose a soft pseudo-labeling strategy that fuses ensemble predictions using spatial and consensus-aware weighting, enabling robust semi-supervised training. Applied to the unlabeled ZeroWaste-s subset, our pseudo-annotations achieve performance gains that surpass fully supervised training, underscoring the effectiveness of scalable annotation pipelines. Our work contributes to the research community by establishing rigorous baselines, introducing a robust ensemble-based pseudo-labeling pipeline, generating high-quality annotations for the unlabeled ZeroWaste-s subset, and systematically evaluating OVOD models under real-world waste sorting conditions.
</i></p>

<!-- </br>
<hr />
</br>

For more details, please refer to our or [arxive paper](). -->

</br>

## :rocket: Updates
- **July 25, 2025** : Accepted in [BMVC 2025](https://bmvc2025.bmva.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **August 25, 2025** : Released code for paper
- **August 26, 2025** : Paper is released [arXiv](https://bmvc2025.bmva.org/)

</br>

## :file_folder: Repository Structure

```
robust-waste-detection/
├── data/                  # Place dataset here (zerowaste-f, zerowaste-s)
├── experiments/           # Experiment configs and results
│   ├── swin-b_zerowaste-s_semi-supervised/
│   ├── swin-b_zerowaste-f_finetune/
│   └── ...
├── external_modules/      # External dependencies (GroundingDINO, mmdetection)
├── semi_sup/              # Scripts for pseudo-label generation and SSL
│   ├── pseudo_annotations/
│   ├── scripts/
│   └── utils/
├── weights/               # Pretrained weights
└── zero_shot_ovod/        # Zero-shot experiments (see sub-README)
```

</br>

## :wrench: Installation
*If you face issues creating the environment, visit https://mmdetection.readthedocs.io/en/latest/get_started.html for help.*

#### 1. Clone the repo and change current directory
```bash
git clone https://github.com/h-abid97/robust-waste-detection
cd robust-waste-detection
```

#### 2. Create a conda environment and activate it.
```bash
conda create --name mmdet-grounding-dino python=3.8 -y
conda activate mmdet-grounding-dino
```

#### 3. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.
#### For GPU Platforms (recommended).
```bash
conda install pytorch torchvision -c pytorch
```

#### We tested with PyTorch 2.0:
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 4. Install MMEngine and MMCV using MIM.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
```

#### 5. Install MMDetection.
```bash
cd external_modules/mmdetection
pip install -v -e .
cd ../..
```

#### 5. Install remaining dependencies.
```bash
pip install -r requirements.txt
```

</br>

 ## :file_cabinet: Dataset Setup

This project uses the [ZeroWaste](https://github.com/dbash/zerowaste/) dataset, which is publicly available.

You will need both:
- **ZeroWaste-f** (labeled subset)
- **ZeroWaste-s** (unlabeled subset)

Please download the ZeroWaste-f and ZeroWaste-s subsets from [here](https://ai.bu.edu/zerowaste/#overview), unzip them, and place them under `robust-waste-detection/data/`. Make sure to follow the following structure:

```
data/
├── zerowaste-f/
│   ├── train/
│   │   ├── data/*.jpg
│   │   └── labels.json
│   ├── val/
│   │   ├── data/*.jpg
│   │   └── labels.json
│   └── test/
│       ├── data/*.jpg
│       └── labels.json
└── zerowaste-s/
    ├── data/*.jpg
    └── labels.json
```

With this structure, all configs will run **out of the box**.

</br>

## :dart: Zero-Shot OVOD
![Optimized Prompt Generation Pipeline](/media/optimized_queries_pipeline.png)

We evaluate state-of-the-art Open-Vocabulary Object Detectors (OVOD) - **GroundingDINO**, **OWLv2**, and **YOLO-World** - on ZeroWaste-f in a **zero-shot** setting.

- **Class-only prompts** underperform due to domain complexity.
- **Optimized prompts** (LLM-enriched) improve results significantly.

📄 **Instructions for running experiments can be found in [zero_shot_ovod/README.md](zero_shot_ovod/README.md).**

### :bar_chart: Results (ZeroWaste-f, Test Set)
![Zero-Shot Class Only](/media/zs_class_only_bm.png)

![Zero-Shot Class Only vs Optimized Prompts](/media/zs_class_only_v_optimized.png)

</br>

## :hammer_and_wrench: Fully Fine-Tuned Baselines
We establish strong **closed-set baselines** on ZeroWaste-f. Models were fine-tuned with COCO-style configs.

### :bar_chart: Results (ZeroWaste-f, Test Set)
![Supervised Finetuning Results](/media/sft_benchmark.png)

### :gear: Training
The below commands will allow you to train GroudingDINO (Swin-T/Swin-B) on the ZeroWaste-f train set.

```bash
# GroundingDINO Swin-T (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/train.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_zerowaste_f.py


# GroundingDINO Swin-B (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/train.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_finetune_zerowaste_f.py
```

Results will be stored directly into `./experiments/`.

### :zap: Evaluation
We release all fine-tuned checkpoints on [Hugging Face](https://huggingface.co/h-abid/bmvc-gdino-zerowaste). You can download the weights and run evaluation as follows:
#### 1. Install the CLI:
```bash
pip install -U "huggingface_hub>=0.23.0"
```

#### 2. Download checkpoints directly into `./weights/`:
```bash
# GroundingDINO Swin-T (ZeroWaste-f fine-tuned)
huggingface-cli download h-abid/bmvc-gdino-zerowaste \
  --include "weights/gdino-swin-t/zerowaste_f_finetuned_best_coco_bbox_mAP.pth" \
  --local-dir ./

# GroundingDINO Swin-B (ZeroWaste-f fine-tuned)
huggingface-cli download h-abid/bmvc-gdino-zerowaste \
  --include "weights/gdino-swin-b/zerowaste_f_finetuned_best_coco_bbox_mAP.pth" \
  --local-dir ./
```

#### 3. Run Evaluation on ZeroWaste-f Test:
```bash
# GroundingDINO Swin-T (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/test.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-t_inference_zerowaste_f.py \
weights/gdino-swin-t/zerowaste_f_finetuned_best_coco_bbox_mAP.pth


# GroundingDINO Swin-B (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/test.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py \
weights/gdino-swin-b/zerowaste_f_finetuned_best_coco_bbox_mAP.pth
```

Results will be stored directly into `./experiments/`.

</br>

## :balance_scale: Semi-Supervised Learning
![Ensemble-based Pseudo-labeling Pipeline](/media/pseudo_label_pipeline.png)

We leverage **ZeroWaste-s (unlabeled)** by generating **ensemble-based pseudo-labels**.


### :bar_chart: Results (ZeroWaste-f, Test Set)
Semi-supervised training yields **+2–4 mAP improvements**, demonstrating the value of unlabeled data.

![Semi-supervised Training Results](/media/semi_sup_results.png)

### :gear: Training
Generate the ensemble-based pseudo-annotations for the unlabeled ZeroWaste-s subset by following the steps outlined below:

```bash
# 1. Generate a single consolidated pseudo-annotations json file
python semi_sup/scripts/consolidate_pseudo_annotations.py

# 2. Generate the consensus-based pseudo-annotations json file
python semi_sup/scripts/generate_ensemble_pseudo_annotations.py

# 3. Add offset to the consensus-based pseudo-annotations json file for use in semi-supervised training
python semi_sup/utils/add_offset.py
```

The below commands will allow you to train GroudingDINO (Swin-T/Swin-B) in a semi-supervised fashion on the labeled ZeroWaste-f train set and the pseudo-labeled ZeroWaste-s subset.

```bash
# GroundingDINO Swin-T (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/train.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-t_semi-sup_zerowaste.py


# GroundingDINO Swin-B (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/train.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py
```

Results will be stored directly into `./experiments/`.

### :zap: Evaluation
We release all fine-tuned checkpoints on [Hugging Face](https://huggingface.co/h-abid/bmvc-gdino-zerowaste). You can download the weights and run evaluation as follows:
#### 1. Install the CLI:
```bash
pip install -U "huggingface_hub>=0.23.0"
```

#### 2. Download checkpoints directly into `./weights/`:
```bash
# GroundingDINO Swin-T (ZeroWaste-f fine-tuned)
huggingface-cli download h-abid/bmvc-gdino-zerowaste \
  --include "weights/gdino-swin-t/zerowaste_semi-sup_best_coco_bbox_mAP.pth" \
  --local-dir ./

# GroundingDINO Swin-B (ZeroWaste-f fine-tuned)
huggingface-cli download h-abid/bmvc-gdino-zerowaste \
  --include "weights/gdino-swin-b/zerowaste_semi-sup_best_coco_bbox_mAP.pth" \
  --local-dir ./
```

#### 3. Run Evaluation on ZeroWaste-f Test:
```bash
# GroundingDINO Swin-T (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/test.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-t_inference_zerowaste_f.py \
weights/gdino-swin-t/zerowaste_semi-sup_best_coco_bbox_mAP.pth \
--work-dir experiments/swin-t_semi-sup_trained_zerowaste_evaluation


# GroundingDINO Swin-B (ZeroWaste-f fine-tuned)
python external_modules/mmdetection/tools/test.py \
external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py \
weights/gdino-swin-b/zerowaste_semi-sup_best_coco_bbox_mAP.pth \
--work-dir experiments/swin-b_semi-sup_trained_zerowaste_evaluation
```

Results will be stored directly into `./experiments/`.

</br>

## :book: Citation
If you find our work or this repository useful, please consider giving a star :star: and citation.

```bibtex
@inproceedings{abid2025robust,
  title={Robust and Label-Efficient Deep Waste Detection},
  author={Abid, Hassan and Muhammad, Khan and Khan, Muhammad Haris},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2025}
}
```

</br>

## :scroll: License
- This repository is released under the MIT License.  
- Please also respect the license terms of the [ZeroWaste](https://github.com/dbash/zerowaste/) dataset.

</br>

## :mailbox: Contact 
Should you have any questions, please create an issue on this repository or contact us at **hassan.abid@mbzuai.ac.ae**

</br>

## :pray: Acknowledgement
We used the following open-source codebases in our work and gratefully acknowledge the authors for releasing them:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for open-vocabulary detection.  
- [OWLv2](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_v2) for zero-shot evaluation.  
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO11, RT-DETR, and YOLO-World implementation.  
- [MMDetection](https://github.com/open-mmlab/mmdetection) as the training framework for fine-tuning.  
- DETR-based variants used in fine-tuning: [Co-DETR](https://github.com/Sense-X/Co-DETR), [DETA](https://github.com/gaopengcuhk/DETA), [DINO](https://github.com/IDEACVR/DINO).

We also thank the authors of the [ZeroWaste](https://github.com/dbash/zerowaste/) dataset for making the data publicly available, enabling reproducible research in sustainable AI.

