# Aggregation & PDF Reporting (`report_gen.py`, `aggregate_adaptive.py`)

## Risk Aggregation (`src/aggregate_adaptive.py`)

Cells are mapped to slide-level diagnoses. Instead of generating a single overall "Abnormal" label per picture, this logic uses a rule-based algorithm mapping cell counts to classical clinical diagnostic categories:

- **NILM** (Negative for Intraepithelial Lesion or Malignancy)
- **LSIL** (Low-Grade Squamous Intraepithelial Lesion)
- **HSIL** (High-Grade Squamous Intraepithelial Lesion)

If Koilocytotic (Low Grade) counts exceed a specific stringency threshold without the presence of Dyskeratotic (High Grade) cells, the slide is given an LSIL warning. If Dyskeratotic cells exceed the high-risk limit, the slide is escalated to HSIL.

The aggregator also builds dynamic **Clinical Interpretation** sentences out of raw metrics.

## PDF Reporting (`src/report_gen.py`)

Uses `ReportLab` library (v4.1.0) to parse the JSON aggregated context into a professionally formatted document.

**Included Sections:**

1. **Header & Metadata:** Patient (or Slide ID), Timestamp, and Version.
2. **AI Summary:** A fast, color-coded high-level summary.
3. **Specimen Adequacy:** Cell counts, quality notations.
4. **Quantitative Analysis:** A multi-colored table listing detected Normal, Benign/Reactive, and Abnormal categories along with their respective cell counts and percentages.
5. **Clinical Interpretation:** Human-readable phrasing that expands on the numbers.
6. **Model Limitations & Disclaimers:** Explicitly states the tool is a research PoC and should not replace primary clinician diagnosis.
