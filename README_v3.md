# GVLM2.0 Landslide Understanding and Decision

## 1. Dataset Overview

This directory contains the GVLM2.0 multi-view landslide text VQA dataset, organized into three annotation tasks:

- `NS`: Natural Scene view
- `RS`: Remote Sensing single-temporal view
- `CS`: Change Sensing (pre/post-event) view

The three core annotation files are:

- `landslide_vqa_NS.json`
- `landslide_vqa_RS.json`
- `landslide_vqa_CS.json`

Each sample includes a `caption` (text description) and a `vqa` list (question-answer items).

---

## 2. Directory Structure

The `landslide_xx_GVLM` file contains 17 events from GVLM, cropped to 1024 pixels, showing pre- and post-landslide images and corresponding masks, along with natural view images matched to the events. `landslide_xx_NEW` contains 7 events from GVLM 1.1. `landslide_vqa_RS.json`, `landslide_vqa_NS.json`, and `landslide_vqa_CS.json` store the VQA and captions for remote sensing, natural, and changing perspectives, respectively.
Main structure of the current directory (some subfolders omitted):

```text
GVLM2.0/
├─ landslide_vqa_CS.json
├─ landslide_vqa_NS.json
├─ landslide_vqa_RS.json
├─ landslide_NS_GVLM/
│  ├─ ALuoi_Vietnam/
│  ├─ Asakura_Japan/
│  ├─ ...
├─ landslide_NS_new/
│  ├─ Brazil/
│  ├─ Enshi/
│  ├─ ...
├─ landslide_RS_GVLM/
│  ├─ ALuoi_Vietnam/
│  │  ├─ img1/
│  │  ├─ img2/
│  │  └─ mask/
│  ├─ Asakura_Japan/
│  ├─ ...
├─ landslide_RS_new/
│  ├─ Brazil/
│  │  ├─ img1/
│  │  ├─ img2/
│  │  ├─ mask/
│  ├─ Enshi/
│  ├─ ...
└─ 
```


---

## 3. JSON Sample Schemas

### 3.1 `landslide_vqa_NS.json` (Natural Scene)

Typical fields in one sample:

- `relative_path`: image path (e.g., `landslide_NS_GVLM\\Jiuzhaigou_China\\xxx.jpg`)
- `caption`: image-level text description
- `vqa`: QA list
  - `question`: question text
  - `type`: question type (e.g., `multiple_choice` / `yes_no`)
  - `options`: candidate options
  - `answer`: ground-truth answer

### 3.2 `landslide_vqa_RS.json` (Remote Sensing, Single-temporal)

Typical fields in one sample:

- `relative_path`: remote-sensing image path (e.g., `landslide_RS_GVLM\\ALuoi_Vietnam\\img2\\patch_0.tif`)
- `caption`: image-level text description
- `vqa`: QA list (same field definitions as above)

### 3.3 `landslide_vqa_CS.json` (Change Sensing, Pre/Post)

Typical fields in one sample:

- `image_pre`: pre-event image path (e.g., `landslide_RS_GVLM/ALuoi_Vietnam/img1/patch_0.tif`)
- `image_post`: post-event image path (e.g., `landslide_RS_GVLM/ALuoi_Vietnam/img2/patch_0.tif`)
- `caption`: pre/post change description
- `vqa`: QA list (same field definitions as above)

---

## 4. Text VQA Statistics (Current Version)

Counting rules:

- `caption` count: number of samples containing a `caption` field
- `question` count: total number of `question` fields across all `vqa` items

| File | Samples | Captions | Questions | Avg Questions per Sample |
|---|---:|---:|---:|---:|
| `landslide_vqa_CS.json` | 1068 | 1068 | 2430 | 2.275 |
| `landslide_vqa_NS.json` | 1027 | 1027 | 3081 | 3.000 |
| `landslide_vqa_RS.json` | 1068 | 1068 | 6211 | 5.816 |

Question type distribution (`type`):

- `landslide_vqa_CS.json`
  - `multiple_choice`: 908
  - `yes_no`: 1522
- `landslide_vqa_NS.json`
  - `multiple_choice`: 2280
  - `yes_no`: 801
- `landslide_vqa_RS.json`
  - `multiple_choice`: 3094
  - `yes_no`: 3117

## 5. Download
Will be released soon.

