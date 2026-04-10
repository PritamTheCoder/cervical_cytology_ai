# Sample Reports

The **Cervical Cytology AI** system produces clinical summary reports in both JSON and PDF formats. Here's a look at what the generated data entails.

## JSON Payload Structure

The raw outputs from the pipeline are saved into `data/reports/json/`. This JSON schema guarantees downstream interoperability, enabling integration into any hospital Information System (HIS) or Electronic Health Record (EHR) database.

### Example: `raw_report.json`

```json
{
  "report_id": "40b2f568-1234-5678-abcd-1234567890ab",
  "slide_id": "0011 ASC US C494 19",
  "patient_id": "UNKNOWN",
  "timestamp": "2024-05-19T10:15:30.123456",
  "schema_version": "2.0",
  "model_info": "MobileViT-S / Cellpose cyto2",
  "total_cells": 120,
  "abnormal_cells": 15,
  "summary": {
    "Superficial-Intermediate": 80,
    "Parabasal": 20,
    "Metaplastic": 5,
    "Koilocytotic": 10,
    "Dyskeratotic": 5,
    "abnormal_threshold": 0.1,
    "diagnostic_risk": "HSIL",
    "clinical_interpretation": "HSIL - High-Grade Squamous Intraepithelial Lesion. The AI detected high-risk (Dyskeratotic) cellular anomalies encompassing 4.1% of the overall specimen.",
    "adequacy_note": "Satisfactory for evaluation (AI detected exactly 120 recognizable cells in the analyzed fields limits)"
  },
  "limitations": "These results were generated automatically by AI and have not been validated by a cytopathologist."
}
```

## PDF Reports

Generated in `data/reports/pdf/`, these files contain structured tables and explicit clinical disclaimers.

The AI groups cells into severity rows:
- **Normal**: Cells presenting baseline healthy morphologies.
- **Benign/Reactive**: Cells exhibiting non-neoplastic changes.
- **Abnormal (Low/High Risk)**: Cells suspected of significant intraepithelial lesions.

### Report Generation Test Tool

If you have a customized folder of JSON inferences, you can generate reports en masse using the provided testing script:

```bash
python scripts/test_report_from_json.py data/reports/json --output-dir data/reports/pdf/test_runs
```

This renders the AI Summary, the Specimen Adequacy logic, and all Quantitive Analysis directly onto a formatted page, saving a local copy (`*_test_report.pdf`) for validation.
