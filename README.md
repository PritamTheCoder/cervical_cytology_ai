# Cervical Cytology AI — End-to-End Clinical Workflow Simulation (PoC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A production-grade AI system simulating a real-world cervical cytology analysis workflow.**

This project goes beyond simple classification by implementing a full pipeline: identifying cells in whole-slide equivalent images, segmenting them, classifying their pathology, and aggregating findings into a structured clinical report.

> **⚠️ Disclaimer**: This project is a **research proof-of-concept**. It is NOT a medical device and is NOT intended for clinical diagnosis.

---

## 🩺 Problem Context

Cervical cancer screening relies on the meticulous analysis of thousands of cells per slide. Fatigue and human error can lead to missed diagnoses.
This system demonstrates how AI can support cytologists by:
-   **Automating Detection**: Finding and segmenting cells in complex scenes.
-   **Triaging**: Classifying cells (e.g., *Dyskeratotic*, *Koilocytotic*) to flag high-risk slides.
-   **Reporting**: Generating interpretative summaries and PDF reports.

---

## 🎯 Project Goals

-   **End-to-End Pipeline**: From raw image to PDF report.
-   **Clinical Relevance**: Aggregating cell-level predictions into slide-level risk assessments.
-   **Modern Tech Stack**: utilizing **FastAPI** for serving and **Cellpose** for state-of-the-art segmentation.
-   **Reproducibility**: Clear structure and modular design.

---

## 🧠 System Architecture

The pipeline consists of four main stages:

### 1. Segmentation (Cellpose)
Extracts individual cells from original microscope images using `Cellpose` (cyto2 model), robust to overlapping and staining variations.

### 2. Classification (MobileViT)
A lightweight Vision Transformer (MobileViT) classifies each segmented cell into one of 5 categories:
-   **Superficial-Intermediate** (Normal)
-   **Parabasal** (Normal)
-   **Metaplastic** (Benign/Reactive)
-   **Koilocytotic** (Low Grade Lesion)
-   **Dyskeratotic** (High Grade Lesion)

*Trained on the SIPaKMeD dataset.*

> **💡 Research Note**: We use **MobileViT-S** (~5.6M params) to enable efficient **Edge AI** deployment.  
> Read our [Research Perspective](docs/research_perspective.md) on why this architecture is best for clinical integration.

### 3. Aggregation Logic
The `ClinicalAggregator` analyzes the distribution of cell types. It applies thresholds to determine if a slide is "Normal", "Low Risk", or "High Risk".

### 4. Reporting
Generates a JSON summary and a downloadable PDF report including:
-   Cell counts per class.
-   Risk assessment.
-   Processing timestamps.

---

## 🛠 Tech Stack

-   **Deep Learning**: PyTorch, torchvision, timm.
-   **Segmentation**: Cellpose.
-   **API Framework**: FastAPI, Uvicorn.
-   **Image Processing**: OpenCV, Albumentations, Pillow.
-   **Reporting**: ReportLab.
-   **Data Processing**: Numpy, Pandas.

---

## 📂 Project Structure

```
├── data/               # Dataset directory (SIPaKMeD)
├── src/                # Source code
│   ├── main.py         # FastAPI application entry point
│   ├── pipeline.py     # Orchestration of segmentation & inference
│   ├── train.py        # Training script for the classifier
│   ├── segment.py      # Cellpose wrapper
│   ├── cell_infer.py   # Classification inference engine
│   └── ...
├── outputs/            # Generated reports and crops
├── weights/            # Model checkpoints
└── requirements.txt    # Dependencies
```

### 2️⃣ Add Dataset
Place SIPaKMeD (and others if used) into:
data/raw/


### 3️⃣ Train / Evaluate Model

**Option A: Python Script**
```bash
python src/train.py
```
*Checkpoints will be saved to `weights/`.*

**Option B: Jupyter Notebook**
You can also train using the provided notebook:
`notebooks/train_mobilevit_s_on_SIPKAMED.ipynb`

> **Note**: The finetuned MobileViT-S model is also freely available on **HuggingFace** at -> [pretrained weights](https://huggingface.co/AurevinP/cervical-cytology-mobilevit-sipakmed).

To evaluate:
```bash
python src/infer.py
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PritamTheCoder/cervical_cytology_ai.git
cd cervical_cytology_ai
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the **SIPaKMeD Database** and extract it into `data/`.
Ensure the structure uses classwise folders (no pre-split Train/Test):
```
data/
  SIPAKMED/
    im_Dyskeratotic/
    im_Koilocytotic/
    im_Metaplastic/
    im_Parabasal/
    im_Superficial_Intermediate/
```

### 3. Training the Classifier

To train the MobileViT model on your data:

```bash
python src/train.py
```
*Checkpoints will be saved to `weights/`.*

### 4. Running the Application (API)

Start the FastAPI server:

```bash
python src/main.py
```

The API will be available at `http://localhost:8000`.
-   **Swagger UI**: Visit `http://localhost:8000/docs` to test the `/analyze-slide/` endpoint interactively.

---

## 📊 Evaluation & Metrics

### Model Performance (MobileViT)

**Device**: CUDA | **Test Samples**: 500  
**Overall Accuracy**: **92.60%**

python src/main.py

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Dyskeratotic** | 0.96 | 1.00 | 0.98 | 100 |
| **Koilocytotic** | 0.80 | 0.93 | 0.86 | 100 |
| **Metaplastic** | 0.93 | 0.87 | 0.90 | 100 |
| **Parabasal** | 0.97 | 0.98 | 0.98 | 100 |
| **Superficial/Int** | 1.00 | 0.85 | 0.92 | 100 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | **500** |

#### Confusion Matrix

```text
[[100   0   0   0   0]  <- Dyskeratotic
 [  3  93   3   1   0]  <- Koilocytotic
 [  0  13  87   0   0]  <- Metaplastic
 [  0   0   2  98   0]  <- Parabasal
 [  1  10   2   2  85]] <- Superficial-Intermediate
```

The system tracks inference latency and per-class performance to ensure clinical relevance.

---

## 📚 References & Citations

**Dataset**:
> Plissiti, M.E., et al. "SIPaKMeD: A new dataset for feature extraction and classification of cells in Pap smear images." *Image Analysis and Stereology*, 2018.

---

## 🤝 Contributing

Contributions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
