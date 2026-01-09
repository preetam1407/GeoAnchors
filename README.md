# GeoFormer: Multi-View Vision–Language Model for Autonomous Driving (VQA)

GeoFormer is a **multi-view Visual Question Answering (VQA)** system for autonomous driving that fuses **6 synchronized surround-view camera assests/images** with a **natural language question** and generates an **open-ended answer** (yes/no, counts, short explanations).

> Core idea: compress multi-view visual information into a **compact scene representation** using **AnchorFormer + Gated Pooling**, then use **T5** to generate answers.

---

## Table of Contents
- Motivation
- Problem Statement
- Challenges & Research Gap
- Dataset
- Method (GeoFormer)
  - High-Level Architecture (image)
  - Detailed Architecture (image)
- Training Setup
- Results
  - Overall Metrics (image + optional table)
  - Baseline Comparison (image)
- Qualitative Results (Inferences)
- Future Work (Phase-2)
- References
- Acknowledgements

---

## Motivation
Autonomous vehicles operate in a dynamic world where decisions must be made quickly and should be explainable.
Beyond perception outputs (boxes/lanes/trajectories), people ask scene-level questions like:
- “What are we waiting for?”
- “Is it safe to go?”
GeoFormer adds a language interface on top of multi-view perception.

---

## Problem Statement
Task: **Multi-view VQA for driving**.

Input:
- 6 synchronized surround-view camera assests/images (front, front-left, front-right, back, back-left, back-right)
- A natural language question

Output:
- An open-ended answer (yes/no, counts, short explanation)

Goal:
Learn p(a | I1..6, q) where I1..6 are the 6 views and q is the question.

---

## Challenges & Research Gap
- 6 cameras + occlusions + crowded roads → harder than single-view reasoning
- Naively pushing all patch tokens into the language model is compute-heavy and redundant
- Many AV-VLM systems are front-view biased → limited 360° understanding
- Need an efficient multi-view fusion method that retains only relevant information

---

## Dataset
Built on **nuScenes**:
- ~1000 scenes, 6 RGB cameras → true 360° view
- VQA-style annotations (keyframes + questions + answers)
- ~377k Q/A over 696 scenes (example split):
  - Train: ~341k
  - Val: ~19.8k
  - Test: ~16.8k
- Question mix (example):
  - Perception, Prediction, Planning, Behavior

> NOTE: Put your dataset download + preprocessing instructions here.
> If you already have scripts, link them below.

---

## Method: GeoFormer
GeoFormer consists of:
1) Shared Image Encoder (CLIP ViT-L/14)
   - Encodes each of the 6 views using shared weights

2) Token Compression (AnchorFormer)
   - Uses CLS→patch attention to select important patches
   - Refines selected anchors for compact multi-view reasoning

3) Scene Fusion + Generation
   - Gated Pooling fuses anchors into a single compact scene token
   - T5 encoder–decoder generates answer conditioned on (scene token + question)

---

## High-Level Architecture (IMAGE PLACEHOLDER)
Replace the image below with your high-level pipeline figure.

![High-Level Overview](assets/architecture/high_level.png)

Suggested path:
- assets/architecture/high_level.png

---

## Detailed Architecture (IMAGE PLACEHOLDER)
Replace the image below with your detailed model diagram (AnchorFormer + Gated Pooling + T5).

![Detailed Architecture](assets/architecture/architecture_detailed.png)

Suggested path:
- assets/architecture/architecture_detailed.png

---

## Training Setup
Example setup (adjust to your actual config):
- Objective: seq2seq cross-entropy
- Optimizer: AdamW
- Learning Rate: 1e-4 (example)
- Batch Size: 2 (example)
- CLIP frozen, trainable modules: fusion + T5 (example)
- GPU: NVIDIA A6000 (example)

---

## Installation (PLACEHOLDER)
Create a venv and install dependencies:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## Data Preparation (PLACEHOLDER)
Example structure:

data/
  nuscenes/
    samples/
    sweeps/
    v1.0-trainval/
  qa_annotations/
    train.json
    val.json
    test.json

---

## Training (PLACEHOLDER)
Example:

python train.py \
  --config configs/train.yaml \
  --data_root data/nuscenes \
  --ann_root data/qa_annotations \
  --save_dir checkpoints/geoforger

---

## Evaluation (PLACEHOLDER)
Example:

python eval.py \
  --ckpt checkpoints/geoforger/best.pt \
  --split test \
  --data_root data/nuscenes \
  --ann_root data/qa_annotations

---

## Results

### Evaluation Metrics
| Model     | EM (%) | F1 (%) | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|----------|--------:|-------:|-------:|-------:|--------:|------:|
| GeoFormer | 51.02  | 69.60  | 47.10  | 37.39  | 69.12   | 2.99  |

---

### Baseline Comparison

| Model | BLEU-4 [%] ↑ | METEOR [%] ↑ | ROUGE-L [%] ↑ | CIDEr ↑ |
|------|--------------:|-------------:|--------------:|--------:|
| EM-VLM4AD<sub>Base</sub> [1] | 45.36 | 34.49 | 71.98 | 3.20 |
| EM-VLM4AD<sub>Q-Large</sub> [1] | 40.11 | 34.34 | 70.72 | 3.10 |
| DriveLM-Agent [2] | 53.09 | 36.19 | 66.79 | 2.79 |
| **GeoFormer (T5-base, subdata_10)** | **47.10** | **37.39** | **69.12** | **2.99** |

---

## Qualitative Results (Inferences)
_This report shows random, best, and worst test examples based on token-level F1, with all six nuScenes camera views._


## Example idx=3648

## **Q:** Question: What is the future state of <c1,CAM_BACK,750.8,541.7>? 

## Answer: Keep going straight-
<!-- **Pred:** `Keep going straight.` -->

<!-- **Gold:** `Keep going straight.` -->

<!-- *Metrics:* ✅ **Exact match** | (EM=1, F1=1.000) -->

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0000_n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621953112404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0001_n008-2018-07-26-12-13-50-0400__CAM_FRONT_LEFT__1532621953104799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0002_n008-2018-07-26-12-13-50-0400__CAM_FRONT_RIGHT__1532621953120482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0003_n008-2018-07-26-12-13-50-0400__CAM_BACK__1532621953137562.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0004_n008-2018-07-26-12-13-50-0400__CAM_BACK_LEFT__1532621953147405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0005_n008-2018-07-26-12-13-50-0400__CAM_BACK_RIGHT__1532621953128113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=819

**Q:** Question: Are there barriers to the front of the ego car? Answer:
**Pred:** `Yes.`
**Gold:** `Yes.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0006_n008-2018-09-18-13-41-50-0400__CAM_FRONT__1537293306662404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0007_n008-2018-09-18-13-41-50-0400__CAM_FRONT_LEFT__1537293306654799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0008_n008-2018-09-18-13-41-50-0400__CAM_FRONT_RIGHT__1537293306670482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0009_n008-2018-09-18-13-41-50-0400__CAM_BACK__1537293306687558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0010_n008-2018-09-18-13-41-50-0400__CAM_BACK_LEFT__1537293306697405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0011_n008-2018-09-18-13-41-50-0400__CAM_BACK_RIGHT__1537293306678113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=9012

**Q:** Question: Which lanes are each important object on in the scene? Answer:
**Pred:** `< c1,CAM_BACK,850.0,500.0> is on the ego lane, < c2,CAM_FRONT,840.0,500.0> is on the ego lane, and < c3,CAM_FRONT,840.0,500.0> is on the left lane.`
**Gold:** `< c2,CAM_BACK,977.5,545.8> is in the ego lane.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.286)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0012_n015-2018-10-08-15-52-24+0800__CAM_FRONT__1538985526612472.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0013_n015-2018-10-08-15-52-24+0800__CAM_FRONT_LEFT__1538985526604844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0014_n015-2018-10-08-15-52-24+0800__CAM_FRONT_RIGHT__1538985526620339.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0015_n015-2018-10-08-15-52-24+0800__CAM_BACK__1538985526637525.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0016_n015-2018-10-08-15-52-24+0800__CAM_BACK_LEFT__1538985526647423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0017_n015-2018-10-08-15-52-24+0800__CAM_BACK_RIGHT__1538985526627893.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=8024

**Q:** Question: What is the status of the truck that is to the front of the ego car? Answer:
**Pred:** `One truck is parked.`
**Gold:** `The truck in front of the ego car is parked.`

*Metrics:* 🟡 **Partial match** | (EM=0, F1=0.500)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0018_n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532707769162404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0019_n008-2018-07-27-12-07-38-0400__CAM_FRONT_LEFT__1532707769154799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0020_n008-2018-07-27-12-07-38-0400__CAM_FRONT_RIGHT__1532707769170482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0021_n008-2018-07-27-12-07-38-0400__CAM_BACK__1532707769187558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0022_n008-2018-07-27-12-07-38-0400__CAM_BACK_LEFT__1532707769197405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0023_n008-2018-07-27-12-07-38-0400__CAM_BACK_RIGHT__1532707769178113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=7314

**Q:** Question: Based on the observations of <c3,CAM_FRONT,350.8,512.5>, what are possible actions to be taken by <c4,CAM_FRONT,1157.3,521.4>? What is the reason? Answer:
**Pred:** `The action is none, the reason is there is no safety issue.`
**Gold:** `The action is none, the reason is there is no safety issue.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0024_n015-2018-09-26-11-17-24+0800__CAM_FRONT__1537932160912460.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0025_n015-2018-09-26-11-17-24+0800__CAM_FRONT_LEFT__1537932160904844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0026_n015-2018-09-26-11-17-24+0800__CAM_FRONT_RIGHT__1537932160920339.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0027_n015-2018-09-26-11-17-24+0800__CAM_BACK__1537932160937525.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0028_n015-2018-09-26-11-17-24+0800__CAM_BACK_LEFT__1537932160947423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0029_n015-2018-09-26-11-17-24+0800__CAM_BACK_RIGHT__1537932160927893.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=4572

**Q:** Question: Are there moving pedestrians to the front of the ego car? Answer:
**Pred:** `Yes.`
**Gold:** `No.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0030_n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537290738762404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0031_n008-2018-09-18-13-10-39-0400__CAM_FRONT_LEFT__1537290738754799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0032_n008-2018-09-18-13-10-39-0400__CAM_FRONT_RIGHT__1537290738770482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0033_n008-2018-09-18-13-10-39-0400__CAM_BACK__1537290738787558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0034_n008-2018-09-18-13-10-39-0400__CAM_BACK_LEFT__1537290738797405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0035_n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__1537290738778113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=3358

**Q:** Question: What is the status of the cars that are to the front left of the ego car? Answer:
**Pred:** `Many cars are parked.`
**Gold:** `Many cars are parked.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0036_n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532707774762404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0037_n008-2018-07-27-12-07-38-0400__CAM_FRONT_LEFT__1532707774754799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0038_n008-2018-07-27-12-07-38-0400__CAM_FRONT_RIGHT__1532707774770482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0039_n008-2018-07-27-12-07-38-0400__CAM_BACK__1532707774787558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0040_n008-2018-07-27-12-07-38-0400__CAM_BACK_LEFT__1532707774797405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0041_n008-2018-07-27-12-07-38-0400__CAM_BACK_RIGHT__1532707774778113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=2848

**Q:** Question: What is the moving status of object <c1,CAM_BACK_RIGHT,1416.7,522.5>? Answer:
**Pred:** `Going ahead.`
**Gold:** `Turn right.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0042_n015-2018-10-08-16-03-24+0800__CAM_FRONT__1538986223262460.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0043_n015-2018-10-08-16-03-24+0800__CAM_FRONT_LEFT__1538986223254844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0044_n015-2018-10-08-16-03-24+0800__CAM_FRONT_RIGHT__1538986223270356.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0045_n015-2018-10-08-16-03-24+0800__CAM_BACK__1538986223287525.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0046_n015-2018-10-08-16-03-24+0800__CAM_BACK_LEFT__1538986223297423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0047_n015-2018-10-08-16-03-24+0800__CAM_BACK_RIGHT__1538986223277893.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=13825

**Q:** Question: Is <c2,CAM_BACK,901.7,555.8> an object that the ego vehicle should consider in the current scene? Answer:
**Pred:** `No.`
**Gold:** `No.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0048_n015-2018-10-08-15-52-24+0800__CAM_FRONT__1538985531112460.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0049_n015-2018-10-08-15-52-24+0800__CAM_FRONT_LEFT__1538985531104844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0050_n015-2018-10-08-15-52-24+0800__CAM_FRONT_RIGHT__1538985531120339.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0051_n015-2018-10-08-15-52-24+0800__CAM_BACK__1538985531137525.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0052_n015-2018-10-08-15-52-24+0800__CAM_BACK_LEFT__1538985531147423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0053_n015-2018-10-08-15-52-24+0800__CAM_BACK_RIGHT__1538985531127893.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=1041

**Q:** Question: Are there parked cars to the front right of the ego car? Answer:
**Pred:** `Yes.`
**Gold:** `Yes.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0054_n008-2018-08-29-16-04-13-0400__CAM_FRONT__1535573958012404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0055_n008-2018-08-29-16-04-13-0400__CAM_FRONT_LEFT__1535573958004799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0056_n008-2018-08-29-16-04-13-0400__CAM_FRONT_RIGHT__1535573958020491.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0057_n008-2018-08-29-16-04-13-0400__CAM_BACK__1535573958037558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0058_n008-2018-08-29-16-04-13-0400__CAM_BACK_LEFT__1535573958047405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0059_n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573958028113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


# Top 5 examples by F1


## Example idx=16816

**Q:** Question: Except for the ego vehicle, what object would consider <c2,CAM_FRONT,243.3,512.5> to be most relevant to its decision? Answer:
**Pred:** `None.`
**Gold:** `None.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0060_n008-2018-08-01-15-52-19-0400__CAM_FRONT__1533153256912404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0061_n008-2018-08-01-15-52-19-0400__CAM_FRONT_LEFT__1533153256904799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0062_n008-2018-08-01-15-52-19-0400__CAM_FRONT_RIGHT__1533153256920482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0063_n008-2018-08-01-15-52-19-0400__CAM_BACK__1533153256937558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0064_n008-2018-08-01-15-52-19-0400__CAM_BACK_LEFT__1533153256947405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0065_n008-2018-08-01-15-52-19-0400__CAM_BACK_RIGHT__1533153256928113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=16815

**Q:** Question: What is the status of the cars that are to the front left of the ego car? Answer:
**Pred:** `Many cars are parked.`
**Gold:** `Many cars are parked.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0066_n008-2018-09-18-15-26-58-0400__CAM_FRONT__1537299155012404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0067_n008-2018-09-18-15-26-58-0400__CAM_FRONT_LEFT__1537299155004799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0068_n008-2018-09-18-15-26-58-0400__CAM_FRONT_RIGHT__1537299155020482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0069_n008-2018-09-18-15-26-58-0400__CAM_BACK__1537299155037572.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0070_n008-2018-09-18-15-26-58-0400__CAM_BACK_LEFT__1537299155047405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0071_n008-2018-09-18-15-26-58-0400__CAM_BACK_RIGHT__1537299155028113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=16814

**Q:** Question: Is it necessary for the ego vehicle to take <c3,CAM_FRONT,837.3,421.3> into account? Answer:
**Pred:** `Yes.`
**Gold:** `Yes.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0072_n008-2018-09-18-12-53-31-0400__CAM_FRONT__1537289943362404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0073_n008-2018-09-18-12-53-31-0400__CAM_FRONT_LEFT__1537289943354799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0074_n008-2018-09-18-12-53-31-0400__CAM_FRONT_RIGHT__1537289943370482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0075_n008-2018-09-18-12-53-31-0400__CAM_BACK__1537289943387558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0076_n008-2018-09-18-12-53-31-0400__CAM_BACK_LEFT__1537289943397405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0077_n008-2018-09-18-12-53-31-0400__CAM_BACK_RIGHT__1537289943378113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=16812

**Q:** Question: What is the moving status of object <c2,CAM_FRONT_RIGHT,801.7,529.2>? Answer:
**Pred:** `Going ahead.`
**Gold:** `Going ahead.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0078_n008-2018-09-18-12-53-31-0400__CAM_FRONT__1537290192912404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0079_n008-2018-09-18-12-53-31-0400__CAM_FRONT_LEFT__1537290192904799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0080_n008-2018-09-18-12-53-31-0400__CAM_FRONT_RIGHT__1537290192920487.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0081_n008-2018-09-18-12-53-31-0400__CAM_BACK__1537290192937558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0082_n008-2018-09-18-12-53-31-0400__CAM_BACK_LEFT__1537290192947405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0083_n008-2018-09-18-12-53-31-0400__CAM_BACK_RIGHT__1537290192928113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=16811

**Q:** Question: What is the status of the bicycle that is to the back of the ego car? Answer:
**Pred:** `One bicycle is without a rider.`
**Gold:** `One bicycle is without a rider.`

*Metrics:* ✅ **Exact match** | (EM=1, F1=1.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0084_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385040612404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0085_n008-2018-08-27-11-48-51-0400__CAM_FRONT_LEFT__1535385040604799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0086_n008-2018-08-27-11-48-51-0400__CAM_FRONT_RIGHT__1535385040620482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0087_n008-2018-08-27-11-48-51-0400__CAM_BACK__1535385040637558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0088_n008-2018-08-27-11-48-51-0400__CAM_BACK_LEFT__1535385040647405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0089_n008-2018-08-27-11-48-51-0400__CAM_BACK_RIGHT__1535385040628113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


# Bottom 5 examples by F1


## Example idx=13

**Q:** Question: What is the traffic signal that the ego vehicle should pay attention to? Answer:
**Pred:** `None.`
**Gold:** `Red light.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0054_n008-2018-08-29-16-04-13-0400__CAM_FRONT__1535573958012404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0055_n008-2018-08-29-16-04-13-0400__CAM_FRONT_LEFT__1535573958004799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0056_n008-2018-08-29-16-04-13-0400__CAM_FRONT_RIGHT__1535573958020491.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0057_n008-2018-08-29-16-04-13-0400__CAM_BACK__1535573958037558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0058_n008-2018-08-29-16-04-13-0400__CAM_BACK_LEFT__1535573958047405.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0059_n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573958028113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=18

**Q:** Question: Is <c3,CAM_BACK_LEFT,91.8,645.9> a traffic sign or a road barrier? Answer:
**Pred:** `No.`
**Gold:** `Yes.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0096_n015-2018-10-08-16-03-24+0800__CAM_FRONT__1538985816412460.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0097_n015-2018-10-08-16-03-24+0800__CAM_FRONT_LEFT__1538985816404844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0098_n015-2018-10-08-16-03-24+0800__CAM_FRONT_RIGHT__1538985816420339.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0099_n015-2018-10-08-16-03-24+0800__CAM_BACK__1538985816437550.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0100_n015-2018-10-08-16-03-24+0800__CAM_BACK_LEFT__1538985816447423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0101_n015-2018-10-08-16-03-24+0800__CAM_BACK_RIGHT__1538985816428118.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=28

**Q:** Question: What is the priority of the objects that the ego vehicle should consider?(in descending order) Answer:
**Pred:** `< c1,CAM_FRONT_RIGHT,1128.3,510.0>, < c2,CAM_FRONT,840.0,500.0>, < c3,CAM_FRONT,840.0,500.0>.`
**Gold:** `< c1,CAM_FRONT,39.2,440.0>, < c3,CAM_FRONT,250.8,418.3>, < c2,CAM_BACK,1009.2,655.0>.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0102_n015-2018-10-08-16-03-24+0800__CAM_FRONT__1538986219262460.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0103_n015-2018-10-08-16-03-24+0800__CAM_FRONT_LEFT__1538986219254844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0104_n015-2018-10-08-16-03-24+0800__CAM_FRONT_RIGHT__1538986219270339.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0105_n015-2018-10-08-16-03-24+0800__CAM_BACK__1538986219287535.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0106_n015-2018-10-08-16-03-24+0800__CAM_BACK_LEFT__1538986219297423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0107_n015-2018-10-08-16-03-24+0800__CAM_BACK_RIGHT__1538986219277893.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=31

**Q:** Question: What actions taken by the ego vehicle can lead to a collision with <c1,CAM_BACK,880.0,498.3>? Answer:
**Pred:** `Brake suddenly.`
**Gold:** `Back up.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0108_n008-2018-09-18-14-43-59-0400__CAM_FRONT__1537296531662404.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0109_n008-2018-09-18-14-43-59-0400__CAM_FRONT_LEFT__1537296531654799.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0110_n008-2018-09-18-14-43-59-0400__CAM_FRONT_RIGHT__1537296531670482.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0111_n008-2018-09-18-14-43-59-0400__CAM_BACK__1537296531687558.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0112_n008-2018-09-18-14-43-59-0400__CAM_BACK_LEFT__1537296531697406.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0113_n008-2018-09-18-14-43-59-0400__CAM_BACK_RIGHT__1537296531678113.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---


## Example idx=34

**Q:** Question: Would <c3,CAM_FRONT,1155.4,528.6> be in the moving direction of the ego vehicle? Answer:
**Pred:** `No.`
**Gold:** `Yes.`

*Metrics:* 🔴 **Low match** | (EM=0, F1=0.000)

<table>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0114_n015-2018-10-08-16-03-24+0800__CAM_FRONT__1538985827412460.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0115_n015-2018-10-08-16-03-24+0800__CAM_FRONT_LEFT__1538985827404844.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0116_n015-2018-10-08-16-03-24+0800__CAM_FRONT_RIGHT__1538985827420339.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px;font-size:12px;background:#111;color:#eee'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0117_n015-2018-10-08-16-03-24+0800__CAM_BACK__1538985827437532.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0118_n015-2018-10-08-16-03-24+0800__CAM_BACK_LEFT__1538985827447423.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
<td style='text-align:center;padding:4px;background:#000;'><img src="assests/images/0119_n015-2018-10-08-16-03-24+0800__CAM_BACK_RIGHT__1538985827427893.jpg" alt="" width="340" style='border-radius:8px;border:1px solid #333;'/></td>
</tr>
</table>

---
---

## Future Work (Phase-2)
- Curating region-specific driving data (India)
- Full ablations (anchor count, gating variants, encoder/decoder choices)
---

## References
- EM-VLM4AD (CVPRW VLADR 2024)
- DriveLM-Agent (ECCV 2024)
- LingoQA (ECCV 2024)
- Transformer / ViT / CLIP / Perceiver / T5
- Metrics: BLEU, ROUGE, METEOR, CIDEr, BERTScore

---

## Acknowledgements
- IIT Roorkee
- Mehta Family School of AI & Data Science
- Supervisor: Prof. Indrajit Ghosh
