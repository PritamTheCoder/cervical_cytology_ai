# Cervical Cytology AI — End-to-End Clinical Workflow Simulation (PoC)

This project builds an end-to-end AI system that simulates a **real clinical cervical cytology analysis workflow**.  
It ingests cytology microscope images, segments individual cells, classifies cell types, aggregates findings, and produces structured reports — similar to what an AI-assisted digital pathology system would do in practice.

> ⚠️ This project is a **research / portfolio proof-of-concept**.  
> It is NOT a medical device and NOT intended for clinical diagnosis.

---

## 🩺 Problem Context

Cervical cytology screening helps detect early precancerous cellular changes.  
Modern AI systems can support cytologists by:

- detecting and segmenting cells
- classifying cell morphological types
- highlighting suspicious cells
- providing statistical and structured reports

However, real clinical datasets and whole slide images (WSIs) are often restricted.  
This project simulates a realistic pipeline using open datasets and pseudo slide generation to demonstrate end-to-end system capability.

---

## 🎯 Project Goals

This project aims to demonstrate:

✔️ A **production-like medical AI workflow**, not just a classifier  
✔️ End-to-end system engineering capability  
✔️ Reliable segmentation + classification performance  
✔️ Clinically relevant result summarization  
✔️ API readiness and deployment potential  

Target model performance: **≥ 92% accuracy** with strong recall for abnormal classes.

---

## 🧠 System Overview

### 1️⃣ Image / Slide Ingestion
- Loads cytology images from SIPaKMeD (and optionally Herlev / APC datasets)

### 2️⃣ Preprocessing
- color normalization  
- resizing & standardization  
- augmentation support  

### 3️⃣ Cell Segmentation
Uses:
- **Cellpose** (cyto2)

Outputs:
- cell masks
- bounding boxes
- cropped cell patches

### 4️⃣ Cell Classification
Uses a lightweight deep network (MobileViT or similar) to classify:

- Dyskeratotic  
- Koilocytotic  
- Metaplastic  
- Parabasal  
- Superficial/Intermediate  

Achieved benchmark example: ~92–95% accuracy on SIPaKMeD.

### 5️⃣ Clinical Workflow Simulation
Because full WSIs are not publicly available, this project:

- generates **pseudo whole-slide images**
- tiles images
- runs segmentation + classification iteratively
- aggregates results to slide-level statistics

### 6️⃣ Reporting
Produces:

- counts of each cell type  
- abnormality ratios  
- highlight overlays  
- structured JSON report  
- optional PDF summary  

---

## 🛠 Tech Stack

**Computer Vision**
- PyTorch
- torchvision
- CellPose / CellSAM

**Experiment Tracking**
- TensorBoard / W&B / MLFlow

**Deployment**
- FastAPI / Flask
- Python 3.9+

---

## 🚀 Running The Project

### 1️⃣ Install
```
pip install -r requirements.txt
```

### 2️⃣ Add Dataset
Place SIPaKMeD (and others if used) into:
data/raw/


### 3️⃣ Train / Evaluate Model
python src/classification/train.py
python src/classification/infer.py


### 4️⃣ Run Full Pipeline


python src/pipeline.py


---

## 📊 Metrics & Evaluation

Tracked metrics include:

- Accuracy
- Macro Precision / Recall / F1
- Per-class performance
- Cohen’s Kappa
- Confusion Matrix
- Segmentation success rate

Results are stored in:


results/metrics/
results/confusion_matrices/


---

## 📡 API

An optional API is included to demonstrate deployment potential.

Start API:


python api/app.py


Endpoints:
- `/upload`
- `/analyze`
- `/report`

---

## 🔍 Documentation
Detailed documentation located in `/docs`:

- Project brief
- Pipeline architecture
- Evaluation results
- Limitations
- Future work

---

## ⚠️ Disclaimer

This project is for **research, learning, and demonstration**.  
It is NOT approved for clinical use, diagnosis, or patient care.

---

## 🏁 Status

- [x] Dataset onboarding
- [x] Classification model
- [x] Segmentation pipeline
- [x] Slide simulation workflow
- [x] Metrics & reporting
- [ ] API polish

---

## 🙌 Author

Developed as part of advanced medical AI research and engineering practice.

---

## 📜 License

MIT
