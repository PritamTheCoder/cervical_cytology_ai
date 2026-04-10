# Architecture & Pipeline Overview

The **Cervical Cytology AI** project is an end-to-end framework. Rather than a simple toy application, it simulates a production medical AI pipeline, handling raw whole-slide Equivalent images (WSI), segmenting individual cells, classifying the cells to ascertain slide-level risk, and finally reporting the outcome.

## The 4-Stage Pipeline

### 1. Segmentation (Cellpose)

Extracts individual cells from original microscope images.

- **Tool**: `Cellpose` (`cyto2` model).
- **Function**: Robust to overlapping cells, cytoplasm artifacts, and various staining techniques.
- **Output**: Segmentation masks and individual bounding-box cell crops, ready for classification.

### 2. Classification (MobileViT)

A lightweight Vision Transformer (MobileViT) is assigned to classify each segmented cell into one of 5 categories:

- **Superficial-Intermediate** (Normal)
- **Parabasal** (Normal)
- **Metaplastic** (Benign/Reactive)
- **Koilocytotic** (Low Grade Lesion)
- **Dyskeratotic** (High Grade Lesion)

**Why MobileViT-S?**
We use `MobileViT-S` (~5.6M params) for efficient **Edge AI** deployment, allowing near real-time processing and the possibility of running on constrained hospital hardware. Read more in our [Research Perspective](research_perspective.md).

### 3. Aggregation Logic

The `ClinicalAggregator` (found in `src/aggregate_adaptive.py`) analyzes the distribution of cell types across a slide.

- **Role**: Rather than reporting on single cells, it groups all cell predictions for the slide (or Field of View).
- **Logic**: It applies thresholds to determine if a slide is "Negative for Intraepithelial Lesion or Malignancy" (NILM), "Low-Grade Squamous Intraepithelial Lesion" (LSIL), or "High-Grade Squamous Intraepithelial Lesion" (HSIL).

### 4. Clinical Reporting

Translates JSON findings into structured PDF reports.

- **Modules**: `src/report_gen.py`
- **Output Types**:
    - **JSON Reports**: Machine-readable statistics of the AI's findings.
    - **PDF Reports**: A completely formatted, multi-section clinical document including Summary, Specimen Adequacy, Quantitative Analysis (color-coded by severity), and mandatory Disclaimers suitable for simulated clinical workflows.

> Check the `/data/reports/` directories (e.g. `data/reports/json` and `data/reports/pdf/`) for sample outputs demonstrating this reporting capability.

## Folder Organization

```text
├── data/               # Datasets, reports, intermediate processing data
├── docs/               # Markdown documentation and Github Pages files
├── outputs/            # Output from analysis
├── notebooks/          # Training notebooks and experiments
├── sandbox/            # Test lab
├── scripts/            # Executable helper scripts and CLI testing tools
├── src/                # Core AI Pipeline
│   ├── config.py       # Global Configuration and Data Contracts
│   ├── aggregate_adaptive.py # Aggregation engine
│   ├── report_gen.py   # PDF Generation Engine
│   └── ...             # Segmentation and inference
└── weights/            # Trained PyTorch state dicts (.pth)
```