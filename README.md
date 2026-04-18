# Cataract Detection System

End-to-end medical computer vision pipeline for cataract screening using deep learning, with YOLOv8-based localization and binary cataract classification.

Project period: Dec 2023 - Jan 2024
Affiliation: Kafr El-Sheikh University
Role: Applied AI Engineer

## Overview

This project is a practical AI screening system for detecting cataracts from eye images. It was designed as an engineering pipeline, not only a model-training exercise.

Core capabilities:
- Data preprocessing and augmentation for robust training behavior.
- Optional ROI localization with YOLOv8 to focus inference on clinically relevant image regions.
- Binary image classification (Cataract vs Normal).
- Reproducible training artifacts and metric-driven evaluation.

Primary use case:
- Automated pre-screening and triage support in ophthalmology workflows, including high-volume clinics and telemedicine intake pipelines.

## Architecture

The system follows an end-to-end computer vision workflow:

Input Images -> Preprocessing and Augmentation -> (Optional) YOLOv8 Localization -> Classifier Inference -> Clinical Screening Output

System components:
- Input layer: curated cataract and normal eye images.
- Processing layer: normalization, resizing, augmentation, dataset splitting.
- Detection layer: YOLOv8 training and inference to localize eye regions and crop ROIs.
- Classification layer: TensorFlow/Keras binary model for cataract prediction.
- Evaluation layer: accuracy, precision, recall, mAP50, and mAP50-95 reporting.
- Inference utilities: Python script utilities in detection model/Main.py for batch detection/cropping.

### Architecture Diagram

![Eye Cataract Detection Architecture](./assets/Eye%20Cataract%20Detection%20Architecture.png)

Diagram flow:
- Stage 1: Raw eye image ingestion from dataset folders.
- Stage 2: Image preprocessing (resizing, normalization, augmentation).
- Stage 3: Detection branch (YOLOv8) producing localized eye/cataract ROIs.
- Stage 4: Classification branch producing Cataract vs Normal probabilities.
- Stage 5: Evaluation and model artifact export (best.pt and h5 model files).
- Stage 6: Inference integration point for screening tools or APIs.

## Features

- End-to-end medical CV pipeline from training to inference.
- Transfer learning workflow for object detection with YOLOv8.
- Binary classifier pipeline in TensorFlow/Keras.
- Batch ROI extraction utility for downstream analysis.
- Experiment outputs tracked under runs/ with best-weight checkpoints.
- Research-ready setup for future API or clinical dashboard integration.

## Technical Highlights

- Two-stage architecture decision:
	- Localization + classification improves focus on eye structures and reduces background noise sensitivity.

- Robust training controls:
	- EarlyStopping and ReduceLROnPlateau are used in classifier training to reduce overfitting and stabilize convergence.

- Practical preprocessing strategy:
	- ImageDataGenerator augmentation and normalized inputs are used to improve generalization under variable lighting and capture conditions.

- Production-minded artifact strategy:
	- Final model artifacts are versioned in repository paths, including classifier and detection checkpoints.

- Evaluation-first engineering:
	- Detection performance is measured with precision/recall/mAP metrics, while classification tracks validation accuracy.

## Tech Stack

- Language: Python
- Deep Learning: TensorFlow/Keras, Ultralytics YOLOv8
- Computer Vision: OpenCV
- Data and Utilities: NumPy, pandas
- Notebook and Experimentation: Jupyter Notebook
- Visualization and Analysis: Matplotlib

## Getting Started

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd Cataract-Model
```

### 2) Create environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow ultralytics opencv-python numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3) Prepare dataset paths

- Verify dataset directories under detection model/data.
- Update detection model/config.yaml to local paths if needed (it currently includes a Colab-style path entry).

### 4) Train or evaluate the classifier

- Open cataract classifier.ipynb.
- Run cells for preprocessing, training, and validation.
- Existing classifier artifact in repo: model/model-96.26%.h5.

### 5) Train or evaluate YOLOv8 detection

- Open detection model/TrainYolov8CustomDataset.ipynb.
- Run training against detection model/config.yaml.
- Existing detection outputs include:
	- detection model/runs/train-50epoch
	- detection model/runs/train-100epoch

### 6) Run detection inference utilities

Main.py includes a batch inference utility class. Update input/output paths and weight file path before running.

## Results

Classifier:
- Best observed validation accuracy: 96.99% (val_accuracy = 0.9699).
- Saved classifier artifact in repository: model/model-96.26%.h5.

YOLOv8 Detection (train-100epoch):
- Max precision: 1.0000 (epoch 20)
- Max recall: 1.0000 (epoch 77)
- Max mAP50: 0.9950 (epoch 30)
- Max mAP50-95: 0.9061 (epoch 88)

YOLOv8 Detection (train-50epoch):
- Max precision: 1.0000 (epoch 29)
- Max recall: 1.0000 (epoch 23)
- Max mAP50: 0.9950 (epoch 41)
- Max mAP50-95: 0.8868 (epoch 37)

## Real-World Use Cases

- Ophthalmology pre-screening and patient triage.
- Tele-ophthalmology workflows where specialist availability is limited.
- Decision-support prototyping for AI-enabled diagnostic tools.
- Academic and applied AI experimentation in medical imaging.
