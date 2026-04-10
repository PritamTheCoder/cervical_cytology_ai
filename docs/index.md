# Cervical Cytology AI — End-to-End Clinical Workflow Simulation

**A production-grade AI system simulating a real-world cervical cytology analysis workflow.**

This project goes beyond simple classification by implementing a full pipeline: identifying cells in whole-slide equivalent images, segmenting them, classifying their pathology, and aggregating findings into a structured clinical report.

!!! warning "Disclaimer"
    This project is a **research proof-of-concept**. It is NOT a medical device and is NOT intended for clinical diagnosis.

---

## 🩺 Problem Context

Cervical cancer screening relies on the meticulous analysis of thousands of cells per slide. Fatigue and human error can lead to missed diagnoses. This system demonstrates how AI can support cytologists by:

- **Automating Detection**: Finding and segmenting cells in complex scenes.
- **Triaging**: Classifying cells (e.g., *Dyskeratotic*, *Koilocytotic*) to flag high-risk slides.
- **Reporting**: Generating interpretative summaries and PDF reports.

## 🎯 Project Goals

- **End-to-End Pipeline**: From raw image to PDF report.
- **Clinical Relevance**: Aggregating cell-level predictions into slide-level risk assessments.
- **Modern Tech Stack**: utilizing **FastAPI** for serving and **Cellpose** for state-of-the-art segmentation.
- **Reproducibility**: Clear structure and modular design.

## 🧠 System Architecture Overview

The pipeline consists of four main stages:

1. **Segmentation (Cellpose)**: Extract individual cells from original microscope images.
2. **Classification (MobileViT)**: Classify each cell into 5 categories (Normal to High Grade Lesion).
3. **Aggregation Logic**: Analyze the cell type distribution to assess slide-level risk.
4. **Reporting**: Generate JSON summaries and structured clinical PDF reports.

Read more in our [Architecture Details](architecture.md).

## 📊 Quick Performance Review

**Model Performance (MobileViT)**

- **Device**: CUDA
- **Overall Accuracy**: **92.60%**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Dyskeratotic** | 0.96 | 1.00 | 0.98 | 100 |
| **Koilocytotic** | 0.80 | 0.93 | 0.86 | 100 |
| **Metaplastic** | 0.93 | 0.87 | 0.90 | 100 |
| **Parabasal** | 0.97 | 0.98 | 0.98 | 100 |
| **Superficial/Int** | 1.00 | 0.85 | 0.92 | 100 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | **500** |

## 🚀 Ready to start?

Check out the [Installation Guide](installation.md) to set up your environment and run the pipeline.
