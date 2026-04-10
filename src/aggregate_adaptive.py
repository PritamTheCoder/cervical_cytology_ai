import json
import logging
from datetime import datetime
from collections import Counter
from typing import List, Dict
from dataclasses import asdict

# Import schema defined above
from config import (
    ReportConfig,
    SlideReport,
    ClinicalSummary,
    CellEvidence,
    report_config as default_report_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClinicalAggregator")

class ClinicalAggregator:
    def __init__(self, config: ReportConfig = default_report_config):
        """
        Initialize with the centralized ReportConfig.
        """
        self.config = config

    def _format_ai_result(self, risk_flag: str) -> str:
        return self.config.AI_RESULT_LABELS.get(risk_flag, f"AI Screening Result: {risk_flag}")

    def _format_recommendation(self, risk_flag: str) -> str:
        return self.config.RISK_RECOMMENDATIONS.get(
            risk_flag,
            "Expert cytology review is recommended before any clinical decision."
        )

    def _build_specimen_adequacy_note(self, total_cells: int, cellularity: str) -> str:
        if cellularity == "ADEQUATE":
            return (
                f"Cellularity is ADEQUATE because {total_cells} analyzable epithelial-cell candidates were detected "
                "(>=50 threshold for stable AI screening behavior)."
            )
        if cellularity == "MARGINAL":
            return (
                f"Cellularity is MARGINAL because only {total_cells} analyzable epithelial-cell candidates were detected "
                "(<50). Interpret results with caution and correlate clinically."
            )
        return "Cellularity is INSUFFICIENT for reliable AI screening interpretation."

    def _build_clinical_interpretation(
        self,
        class_counts: Counter,
        abnormal_count: int,
        abnormal_ratio: float,
        total_valid: int
    ) -> str:
        parts = [
            (
                "AI-assisted interpretation: "
                f"abnormal-appearing cells represent {abnormal_ratio:.1%} "
                f"({abnormal_count}/{total_valid}) of analyzed cells."
            )
        ]

        koilo_count = class_counts.get("im_Koilocytotic", 0)
        dysk_count = class_counts.get("im_Dyskeratotic", 0)

        if koilo_count > 0:
            parts.append(
                f"Koilocytotic candidates detected: {koilo_count}. "
                f"{self.config.CLASS_INTERPRETATION.get('im_Koilocytotic', '')}"
            )

        if dysk_count > 0:
            parts.append(
                f"Dyskeratotic candidates detected: {dysk_count}. "
                f"{self.config.CLASS_INTERPRETATION.get('im_Dyskeratotic', '')}"
            )

        parts.append(
            "These are AI suggestions for triage support and are not a definitive diagnosis; "
            "expert cytology/pathology confirmation is mandatory."
        )

        return " ".join(parts)
        
    def _get_adaptive_thresholds(self, total_cells: int):
        """
        Determines risk thresholds dynamically based on slide cellularity.
        """
        if total_cells < 50:
            return {"min_count": 1, "min_ratio": 0.02, "mode": "High Sensitivity (Sparse)"}
        elif total_cells < 500:
            return {"min_count": 2, "min_ratio": 0.015, "mode": "Balanced"}
        else:
            return {"min_count": 5, "min_ratio": 0.01, "mode": "Noise Supression (WSI)"}
        
    def analyze_slide(self, slide_id: str, predictions: List[Dict]) -> SlideReport:
        """
        Input: List of cell predictions with 'class_probs' and 'bbox',
        Output: Structured Report Object in config.py.
        """
        # Filter & validate
        valid_cells = []
        abnormal_evidence = []
        
        for p in predictions:
            pred_class = p.get('predicted_class')
            
            # Mapping
            if pred_class is None:
                print(f"[ERROR] No predicted class found for cell {p.get('cell_id')}!")
                continue
            group = self.config.CLINICAL_MAPPING.get(pred_class, "IGNORE")
            
            if group == "IGNORE":
                continue
            
            # Extract confidence
            probs = p.get('class_probs', {})
            confidence = probs.get(pred_class, 0.0) if probs else 0.0
            
            # CONFIDENCE GATING:
            # Use threshold from CONFIG
            if group == "ABNORMAL" and confidence < self.config.AGGREGATION_CONFIDENCE_THRESHOLD:
                logger.warning(f"Downgrading low-conf abnormal cell {p.get('cell_id')} ({confidence:.2f})")
                group = "BENIGN_UNCERTAIN" # Re-bucket for internal logic
                pred_class = "im_Uncertain_Low_Conf" # Avoid contradiction in final report tables

            valid_cells.append({
                "class": pred_class, 
                "group": group, 
                "conf": confidence,
                "raw": p
            })

            # Collect Evidence for Gallery
            if group == "ABNORMAL":
                abnormal_evidence.append(CellEvidence(
                    cell_id=p.get('cell_id', 'unknown'),
                    cell_class=pred_class,
                    confidence=confidence,
                    bbox=p.get('bbox', []),
                    heatmap_path=None # Placeholder for Phase 2A/XAI integration
                ))

        # --- 2. Calculate Statistics ---
        total_valid = len(valid_cells)
        if total_valid == 0:
            return self._empty_report(slide_id)

        # Count groups
        group_counts = Counter([x['group'] for x in valid_cells])
        abnormal_count = group_counts["ABNORMAL"]
        abnormal_ratio = abnormal_count / total_valid

        # Count specific classes
        class_counts = Counter([x['class'] for x in valid_cells])

        # --- 3. Adaptive Risk Logic ---
        thresh = self._get_adaptive_thresholds(total_valid)
        
        risk_flag = "NORMAL"
        primary_finding = "AI screening did not identify a significant burden of abnormal-appearing cells."

        # Logic comparison using CONFIG values where applicable
        if abnormal_count >= thresh['min_count']:
            if abnormal_ratio >= self.config.HIGH_RISK_RATIO:
                risk_flag = "HIGH_RISK"
                primary_finding = (
                    f"AI suggests a high proportion of abnormal-appearing cells ({abnormal_ratio:.1%}). "
                    "Suspicious for high-grade abnormality (AI suggestion); expert review is required."
                )
            elif abnormal_ratio >= thresh['min_ratio']:
                risk_flag = "ELEVATED_RISK"
                primary_finding = (
                    f"AI detected abnormal-appearing cells above the screening threshold ({abnormal_count} cells). "
                    "Clinical significance should be determined by expert review."
                )
            else:
                risk_flag = "NORMAL"
                primary_finding = (
                    "AI detected isolated abnormal-appearing cells at low burden; "
                    "significance is uncertain and requires expert correlation."
                )

        cellularity = "ADEQUATE" if total_valid >= 50 else "MARGINAL"
        low_confidence_count = group_counts.get("BENIGN_UNCERTAIN", 0)
        now = datetime.now()
        timestamp = now.isoformat()

        ai_screening_result = self._format_ai_result(risk_flag)
        recommendation = self._format_recommendation(risk_flag)
        adequacy_note = self._build_specimen_adequacy_note(total_valid, cellularity)
        clinical_interpretation = self._build_clinical_interpretation(
            class_counts,
            abnormal_count,
            abnormal_ratio,
            total_valid
        )
        report_id = f"{self.config.REPORT_ID_PREFIX}-{slide_id}-{now.strftime('%Y%m%d%H%M%S')}"

        # --- 4. Rank Evidence (Optimization for PDF) ---
        # Sort abnormal cells by confidence (descending)
        abnormal_evidence.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit evidence count using CONFIG
        top_evidence = abnormal_evidence[:self.config.MAX_EVIDENCE_CELLS]

        # --- 5. Construct Report ---
        summary = ClinicalSummary(
            slide_id=slide_id,
            timestamp=timestamp,
            risk_flag=risk_flag,
            primary_finding=primary_finding,
            cellularity=cellularity,
            abnormal_ratio=round(abnormal_ratio, 4),
            logic_mode=thresh['mode'],
            total_cells=total_valid,
            abnormal_cells=abnormal_count,
            low_confidence_cells=low_confidence_count,
            ai_screening_result=ai_screening_result,
            recommendation=recommendation,
            specimen_adequacy_note=adequacy_note,
            clinical_interpretation=clinical_interpretation,
            report_id=report_id
        )

        return SlideReport(
            summary=summary,
            class_counts=dict(class_counts),
            clinical_group_counts=dict(group_counts),
            top_abnormal_cells=top_evidence,
            model_info={
                "model_name": self.config.MODEL_NAME_DISPLAY,
                "model_type": self.config.MODEL_TYPE,
                "model_version": self.config.MODEL_VERSION,
                "training_dataset": self.config.TRAINING_DATASET,
                "accuracy": self.config.METRIC_ACCURACY,
                "precision": self.config.METRIC_PRECISION,
                "recall": self.config.METRIC_RECALL,
                "f1": self.config.METRIC_F1,
                "aggregation_confidence_threshold": self.config.AGGREGATION_CONFIDENCE_THRESHOLD,
            },
            limitations=list(self.config.LIMITATIONS_DEFAULT),
            display_labels={
                "risk_display": ai_screening_result,
                "uncertain_class_label": "Low Confidence Predictions",
                "abnormal_hint": "Suspicious for High-Grade Abnormality (AI Suggestion)",
            },
            schema_version=self.config.REPORT_SCHEMA_VERSION,
        )

    def _empty_report(self, slide_id):
        """Fail-safe for empty slides."""
        now = datetime.now()
        timestamp = now.isoformat()
        report_id = f"{self.config.REPORT_ID_PREFIX}-{slide_id}-{now.strftime('%Y%m%d%H%M%S')}"
        risk_flag = "INDETERMINATE"
        return SlideReport(
            summary=ClinicalSummary(
                slide_id=slide_id,
                timestamp=timestamp,
                risk_flag=risk_flag,
                primary_finding="AI screening could not be completed because no analyzable cells were detected.",
                cellularity="INSUFFICIENT",
                abnormal_ratio=0.0,
                logic_mode="N/A",
                total_cells=0,
                abnormal_cells=0,
                low_confidence_cells=0,
                ai_screening_result=self._format_ai_result(risk_flag),
                recommendation=self._format_recommendation(risk_flag),
                specimen_adequacy_note=self._build_specimen_adequacy_note(0, "INSUFFICIENT"),
                clinical_interpretation=(
                    "AI-assisted interpretation is not available due to insufficient analyzable cells. "
                    "Repeat sampling or expert review is recommended."
                ),
                report_id=report_id,
            ),
            class_counts={},
            clinical_group_counts={},
            top_abnormal_cells=[],
            model_info={
                "model_name": self.config.MODEL_NAME_DISPLAY,
                "model_type": self.config.MODEL_TYPE,
                "model_version": self.config.MODEL_VERSION,
                "training_dataset": self.config.TRAINING_DATASET,
                "accuracy": self.config.METRIC_ACCURACY,
                "precision": self.config.METRIC_PRECISION,
                "recall": self.config.METRIC_RECALL,
                "f1": self.config.METRIC_F1,
                "aggregation_confidence_threshold": self.config.AGGREGATION_CONFIDENCE_THRESHOLD,
            },
            limitations=list(self.config.LIMITATIONS_DEFAULT),
            display_labels={
                "risk_display": self._format_ai_result(risk_flag),
                "uncertain_class_label": "Low Confidence Predictions",
            },
            schema_version=self.config.REPORT_SCHEMA_VERSION,
        )

    def save_for_pdf(self, report: SlideReport, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2)

# --- Example Usage ---
if __name__ == "__main__":
    # Mock Data with Probabilities
    mock_preds = [
        {"cell_id": "c1", "predicted_class": "im_Superficial_Intermediate", "class_probs": {"im_Superficial_Intermediate": 0.95}},
        {"cell_id": "c2", "predicted_class": "im_Dyskeratotic", "class_probs": {"im_Dyskeratotic": 0.99}, "bbox": [10, 10, 50, 50]}, # High Conf
        {"cell_id": "c3", "predicted_class": "im_Dyskeratotic", "class_probs": {"im_Dyskeratotic": 0.55}, "bbox": [60, 60, 50, 50]}, # Low Conf (Below 0.75)
    ]

    # Initialize with default config from config.py
    agg = ClinicalAggregator()
    report = agg.analyze_slide("SLIDE_001", mock_preds)
    
    print(f"Risk: {report.summary.risk_flag}")
    print(f"Abnormal Count (High Conf): {report.clinical_group_counts.get('ABNORMAL', 0)}")
    print(f"Top Evidence: {len(report.top_abnormal_cells)} cells ready for visualization.")
    
    # Save for Phase 7 (Reporting)
    agg.save_for_pdf(report, "slide_001_data.json")