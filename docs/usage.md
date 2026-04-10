# Usage Guide

This document explains how to use the different parts of the **Cervical Cytology AI** system.

## 1. Running the FastAPI Application

The primary way to use the pipeline is via its FastAPI backend.

```bash
python src/main.py
```

The API will be available at `http://localhost:8000`.

-   **Swagger UI**: Visit `http://localhost:8000/docs` to test the `/analyze-slide/` endpoint interactively without needing a separate frontend app.

---

## 2. End-to-End Pipeline (Standalone)

When testing outside an API request, this script simulates a clinical workflow: processing a directory of images as if they were a single slide, performing segmentation, classification, aggregation, and report generation in sequence.

```bash
python src/pipeline.py
```

-   **Input**: Place raw frames in `data/Test_APC/` (or the folder defined by your config).
-   **Output**: Results—both `json` logs and a PDF report—are saved to `data/reports/`.

---

## 3. Testing PDF Generation Individually

If you only want to test the PDF generator (e.g., verifying aesthetic margins, clinical text formatting, or row coloring) on a directory of existing JSON predictions without running the full neural network, use the provided test script:

```bash
python scripts/test_report_from_json.py
```

**Custom Input/Output:**

```bash
python scripts/test_report_from_json.py data/reports/json --output-dir data/reports/pdf/test_runs
```

This reads structured `SlideReport` and `ClinicalSummary` payloads from JSON files and instantly converts them into aesthetically finished clinical PDFs.

---

## 4. Running Inference Only

You can run the classifier on individual images or directories without the full segmentation pipeline.

**Single Image:**

```bash
python src/infer.py --image path/to/image.png
```

**Directory Evaluation:**

```bash
python src/infer.py --test_dir path/to/dataset --device cuda
```

## 5. Training the Classifier (MobileViT)

**Option A: Python Script**

```bash
python src/train.py
```

Checkpoints will be automatically saved to `weights/`. Ensure `data/SIPAKMED/` is properly formatted before running.

**Option B: Jupyter Notebook**

You can also explore and train interactively using the provided notebook:

```text
notebooks/train_mobilevit_s_on_SIPKAMED.ipynb
```
