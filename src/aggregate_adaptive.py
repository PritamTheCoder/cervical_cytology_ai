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
        primary_finding = "No significant abnormalities detected."

        # Logic comparison using CONFIG values where applicable
        if abnormal_count >= thresh['min_count']:
            if abnormal_ratio >= self.config.HIGH_RISK_RATIO:
                risk_flag = "HIGH_RISK"
                primary_finding = f"High burden of abnormal cells ({abnormal_ratio:.1%})."
            elif abnormal_ratio >= thresh['min_ratio']:
                risk_flag = "ELEVATED_RISK"
                primary_finding = f"Abnormal cells detected above threshold ({abnormal_count} cells)."
            else:
                risk_flag = "NORMAL"
                primary_finding = f"Isolated abnormal cells detected (likely noise)."

        # --- 4. Rank Evidence (Optimization for PDF) ---
        # Sort abnormal cells by confidence (descending)
        abnormal_evidence.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit evidence count using CONFIG
        top_evidence = abnormal_evidence[:self.config.MAX_EVIDENCE_CELLS]

        # --- 5. Construct Report ---
        summary = ClinicalSummary(
            slide_id=slide_id,
            timestamp=datetime.now().isoformat(),
            risk_flag=risk_flag,
            primary_finding=primary_finding,
            cellularity="ADEQUATE" if total_valid >= 50 else "MARGINAL",
            abnormal_ratio=round(abnormal_ratio, 4),
            logic_mode=thresh['mode']
        )

        return SlideReport(
            summary=summary,
            class_counts=dict(class_counts),
            clinical_group_counts=dict(group_counts),
            top_abnormal_cells=top_evidence
        )

    def _empty_report(self, slide_id):
        """Fail-safe for empty slides."""
        return SlideReport(
            summary=ClinicalSummary(
                slide_id, 
                datetime.now().isoformat(), 
                "INDETERMINATE", 
                "No Cells", 
                "INSUFFICIENT", 
                0.0, 
                "N/A"
            ),
            class_counts={}, 
            clinical_group_counts={}, 
            top_abnormal_cells=[]
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