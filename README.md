# Open Science Artifacts

This directory contains the code, configurations, and representative data
necessary for evaluating the reproducibility of our paper.
Heavy data (e.g., full physical-world videos) and large pretrained weights
are intentionally not bundled; instead, instructions for obtaining or
substituting them are provided in this README.

---

## Quick Start (for Artifact Evaluation)

This artifact supports **partial reproduction** of the main claims in the paper.
For Artifact Evaluation, we recommend the following minimal steps:

1. Inspect representative digital examples in `data/digital_examples/`
2. Run a digital attack generation example (Section 4.1 or 4.2)
3. Run inference using the provided DetectorAPI or Classifier
4. (Optional) Inspect system-level CARLA scripts (Section 6)

**Full physical-world reproduction is NOT required for artifact evaluation.**

---

## Table of Contents
1. Structure  
2. Code Overview  
3. Preparation  
4. Attack Generation  
5. Target Models  
6. CARLA Driving (System-Level Evaluation)  
7. Data (Digital / Physical Experiments)  
8. Weights and External Dependencies  
9. Example Commands  
10. External Resources  

---

## 1. Structure

OpenScience/
  AttackGeneration/
    1_ParameterSetting
    2_Optimization
      Altering/ (classification attacks)
      Hiding/ (object detection attacks)
  TargetModel/
    Detector/ (DetectorAPI)
    Classifier/
  Carla_TSR_Driver/ (system-level evaluation)

---


---

## 2. Code Overview

- **AttackGeneration/Altering**  
  Attacks against **classification models**, aiming to **alter the predicted
  class** of traffic signs (misclassification).

- **AttackGeneration/Hiding**  
  Attacks against **object detection models**, aiming to **hide traffic signs**
  by causing misdetection (i.e., the detector fails to detect the sign).

- **AttackGeneration/1_ParameterSetting**  
  Estimation of optical parameters for retroreflection reproduction,
  including Specular Tint estimation and color optimization.

- **TargetModel/Detector**  
  DetectorAPI for object detectors (e.g., YOLO trained on ARTS).

- **TargetModel/Classifier**  
  Classification model inference (e.g., SimpleCNN).

- **Carla_TSR_Driver**  
  System-level evaluation using the CARLA simulator, including scripts
  and configuration files for autonomous driving experiments.

- **traffic_sign_classifiers**  
  Jupyter notebooks for inference and evaluation.

---

## 3. Preparation

- Install **Blender 4.1.1** from the official website.
- Blender `.blend` files are stored under:
  `Archive/2_Optimization copy/`
  (e.g.,  
  `Archive/2_Optimization copy/1_Altering/SL65/20240615_us_sl65_reflector_multiple_color.blend`)
- Attack scripts expect `.blend` files under `blender_file/`.  
  Set `BLEND_FILE` explicitly or copy/link `.blend` files accordingly.
- Install **CARLA 0.9.15** using the official Docker image and additional assets.
- Replace `<PATH_TO_...>` in scripts according to your environment.
- Background images are assumed to come from **BDD100K**, but any driving
  image dataset can be used by setting `BACKGROUND_IMAGE_DIR`.

### Path Configuration (Blender Experiments)

The following environment variables are supported:

- `PROJECT_ROOT`: repository root (auto-detected if unset)
- `HOME_GO`: alternative to `$HOME`
- `BLENDER_PATH`: path to Blender executable
- `BLEND_FILE`: path to `.blend` file
- `ATTACKER_SCRIPT`: Python script passed to Blender
- `BACKGROUND_IMAGE_DIR`: directory for background images
- `OUTPUT_CSV_ROOT`: root directory for output CSV/JSON logs

Relative paths passed via `--background_image_path` and `--output_csv_dir`
are resolved relative to `PROJECT_ROOT`.

---

## 4. Attack Generation (AttackGeneration)

### 4.0 Parameter Setting (Retroreflection Modeling)

- `calc_specular_Tint.ipynb`: estimates Specular Tint from headlight spectra
- `20240613_no_opt_color.blend`: Blender file containing Python code for
  color optimization
- Auxiliary script: `get_average_RGB.py`

### 4.1 Altering (Classification Attacks)

- Organized by STOP / SL65
- Includes **non-targeted** and **targeted** attacks
- Running `open_blender.py` / `open_blender2.py` launches Blender and Python
  scripts jointly to generate attacks
- Patch material parameters are derived from Section 4.0

### 4.2 Hiding (Object Detection Attacks)

- Organized by STOP / SL65
- `open_blender*.py`, `hiding_attacker*.py`, and `.blend` files operate together

### 4.3 Digital Outputs

- Execution of `open_blender*.py` produces CSV/JSON logs in `--output_csv_dir`
- Rendered images are saved under `blender_file/render/optuna/...`
- Aggregated logs (e.g., ASR) are stored in files such as:
  `attack_effectiveness_evaluation2/<patch>/log.json`

---

## 5. Target Models (TargetModel)

- **DetectorAPI**: inference for object detectors (YOLO/ARTS)
- **Classifier**: inference for classification models (SimpleCNN)
- Digital and physical evaluations use these models
- Detector inference is run via `test.ipynb`
- Classifier inference is run via `infer.ipynb` (traffic sign regions are
  manually cropped)

---

## 6. CARLA Driving (System-Level Evaluation)

This directory contains scripts and configuration files for system-level
evaluation using CARLA.

**AE Note:**  
For artifact evaluation, reviewers are **not required** to reproduce full
CARLA simulations. Inspection of scripts and configurations is sufficient.

Preparation:
- CARLA 0.9.15 (official Docker)
- Additional assets loaded and saved as a server image
- CARLA PythonAPI installed via `pip`

Example commands:

python Carla_TSR_Driver/src/place_stop_sign.py
python Carla_TSR_Driver/src/run_autopilot.py



Logs are written to `logs/autopilot.csv`, where `dist_to_stop_m` indicates
the distance to the stop line.

---

## 7. Data (Digital / Physical Experiments)

### Digital Experiments
- One representative image per condition (STOP/SL65 × patch type × size)
- Full image sets are not included; logs/CSV files summarize all trials
- Background images are assumed to be from BDD100K (or substitutes)

### Physical Experiments
- Representative benign / attack images for STOP and SL65 (15 m, frontal)
- Minimal optimization process videos
- Demonstration driving videos (illustrative only; **not required for AE**)
- Defense before/after comparison images
- User study stimuli (subset only)
- Perceptual similarity evaluated on cropped sign regions

---

## 8. Weights and External Dependencies

- Only **core pretrained weights** are included (e.g., YOLO/ARTS/SimpleCNN)
- Other weights must be obtained or trained separately
- Commercial models and large binaries are not included

Minimal dependencies:
- Python ≥ 3.8, Jupyter
- PyTorch ≥ 1.8, torchvision
- Ultralytics 8.4.12
- Blender 4.1.1
- CARLA 0.9.15 + CARLA PythonAPI
- MMDetection (mmdet, mmcv, mmengine)
- Optuna, Albumentations, OpenCV
- NumPy, Pandas, Pillow, Matplotlib, scikit-learn
- LPIPS, scikit-image

---

## 9. Example Commands

DetectorAPI:
python OpenScience/TargetModel/Detector/DetectorAPI/app.py
Classifier inference:
jupyter notebook OpenScience/traffic_sign_classifiers/infer.ipynb
Specular Tint estimation:
jupyter notebook OpenScience/AttackGeneration/1_ParameterSetting/calc_specular_Tint.ipynb
Blender attack generation (example):
python OpenScience/AttackGeneration/Hiding/STOP/open_blender2.py



---

## 10. External Resources

- Blender 4.1: https://www.blender.org/download/releases/4-1/
- BDD100K: https://bair.berkeley.edu/blog/2018/05/30/bdd/
- Mapillary Traffic Sign Dataset: https://www.mapillary.com/dataset/trafficsign
- ARTS Dataset: https://swshah.w3.uvm.edu/vail/datasets.php
- CARLA 0.9.15: https://github.com/carla-simulator/carla/releases/tag/0.9.15
