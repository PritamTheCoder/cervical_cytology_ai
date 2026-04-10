# Data Contracts & File Utilities

The `src/data/` and `src/contracts/` packages manage the clinical and raw data formats throughout the pipeline.

## Contracts (`src/contracts/`)

Data contracts are enforced with Python `dataclasses`. These guarantee strong type-checking when JSON is mapped between the FastAPI routers and the background worker processes.

| Contract | Purpose |
|----------|---------|
| `Cell` | Individual cell properties (bounding box, confidence, predicted class). |
| `Slide` | Holds metadata about a digital slide/field of view. |
| `ClinicalSummary` | Aggregated report details, e.g., cell counts, abnormal thresholds. |
| `SlideReport` | The root payload sent into the `ClinicalReportGenerator`. Includes Model Info, Disclaimer, and the summary. |

## Datasets (`src/data/`)

Modules dealing with transformations, dataloading, and dataset splits.

- `dataset.py` and `sipakmed.py`: Wrappers around `torch.utils.data.Dataset` specifically tailored for parsing SIPaKMeD.
- `transforms.py`: Preprocessing logic (Albumentations, normalization, resizing).
