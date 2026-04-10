import datetime
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import report_config


class ClinicalReportGenerator:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.summary = self.data.get("summary", {})
        self.model_info = self.data.get("model_info", {})
        self.limitations = self.data.get("limitations") or list(report_config.LIMITATIONS_DEFAULT)
        self.display_labels = self.data.get("display_labels", {})

        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.footer_meta = {
            "report_id": self.summary.get("report_id", "N/A"),
            "timestamp": self._format_timestamp(self.summary.get("timestamp", "")),
            "model_version": self.model_info.get("model_version", report_config.MODEL_VERSION),
        }

    def setup_custom_styles(self):
        """Define visual styles for clinically-oriented PDF sections."""
        self.styles["Heading1"].alignment = 1

        self.styles.add(
            ParagraphStyle(
                name="InstitutionHeader",
                parent=self.styles["Heading2"],
                textColor=colors.navy,
                alignment=1,
                spaceAfter=4,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="SummaryLabel",
                parent=self.styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="SummaryValue",
                parent=self.styles["Normal"],
                fontSize=9,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="ClinicalNote",
                parent=self.styles["Normal"],
                fontSize=9,
                leading=12,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="StrongDisclaimer",
                parent=self.styles["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#7A1C1C"),
                leading=12,
            )
        )

    def _draw_footer(self, canvas, doc):
        """Draw footer metadata on every page."""
        canvas.saveState()
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(40, 50, 570, 50)

        canvas.setFont("Helvetica-Oblique", 8)
        canvas.setFillColor(colors.grey)

        canvas.drawString(40, 36, f"Medical AI | {report_config.REPORT_TITLE}")
        canvas.drawRightString(570, 36, f"Report ID: {self.footer_meta.get('report_id', 'N/A')}")
        canvas.drawString(
            40,
            24,
            (
                f"Generated: {self.footer_meta.get('timestamp', 'N/A')} | "
                f"Model Version: {self.footer_meta.get('model_version', 'N/A')}"
            ),
        )
        canvas.drawRightString(570, 24, f"Page {doc.page}")
        canvas.restoreState()

    @staticmethod
    def _format_timestamp(raw_ts: str) -> str:
        if not raw_ts:
            return "N/A"
        try:
            return datetime.datetime.fromisoformat(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return raw_ts

    @staticmethod
    def _risk_color(risk_flag: str):
        if risk_flag == "HIGH_RISK":
            return colors.HexColor("#B00020")
        if risk_flag == "ELEVATED_RISK":
            return colors.HexColor("#B57600")
        if risk_flag == "INDETERMINATE":
            return colors.HexColor("#666666")
        return colors.HexColor("#0B6E4F")

    @staticmethod
    def _humanize_class_name(cell_class: str) -> str:
        if cell_class == "im_Uncertain_Low_Conf":
            return "Low Confidence Predictions"
        return cell_class.replace("im_", "").replace("_", " ")

    def _clinical_group(self, cell_class: str) -> str:
        if cell_class == "im_Uncertain_Low_Conf":
            return "LOW_CONFIDENCE"
        return report_config.CLINICAL_MAPPING.get(cell_class, "UNKNOWN")

    def _ai_screening_result(self, risk_flag: str) -> str:
        return self.summary.get("ai_screening_result") or self.display_labels.get("risk_display") or report_config.AI_RESULT_LABELS.get(
            risk_flag,
            f"AI Screening Result: {risk_flag}",
        )

    def _recommendation(self, risk_flag: str) -> str:
        return self.summary.get("recommendation") or report_config.RISK_RECOMMENDATIONS.get(
            risk_flag,
            "Expert review is recommended before any clinical action.",
        )

    def _adequacy_note(self, cellularity: str, total_cells: int) -> str:
        summary_note = self.summary.get("specimen_adequacy_note")
        if summary_note:
            return summary_note

        if cellularity == "ADEQUATE":
            return (
                f"Specimen is marked ADEQUATE because approximately {total_cells} analyzable epithelial-cell candidates "
                "were detected, which is generally sufficient for AI screening interpretation."
            )
        if cellularity == "MARGINAL":
            return (
                f"Specimen is marked MARGINAL because only about {total_cells} analyzable epithelial-cell candidates "
                "were detected; interpret findings with additional caution."
            )
        return "Specimen is INSUFFICIENT for reliable AI-assisted interpretation."

    def _default_interpretation(self, abnormal_ratio: float, class_counts: dict, abnormal_cells: int, total_cells: int) -> str:
        parts = [
            (
                "AI-assisted interpretation: "
                f"abnormal-appearing cells are estimated at {abnormal_ratio:.1%} "
                f"({abnormal_cells}/{total_cells}) of analyzed cells."
            )
        ]

        koilo_count = class_counts.get("im_Koilocytotic", 0)
        if koilo_count > 0:
            parts.append(
                f"Koilocytotic candidates: {koilo_count}. "
                f"{report_config.CLASS_INTERPRETATION.get('im_Koilocytotic', '')}"
            )

        dysk_count = class_counts.get("im_Dyskeratotic", 0)
        if dysk_count > 0:
            parts.append(
                f"Dyskeratotic candidates: {dysk_count}. "
                f"{report_config.CLASS_INTERPRETATION.get('im_Dyskeratotic', '')}"
            )

        parts.append(
            "These findings are AI suggestions and must be reviewed and confirmed by a qualified cytotechnologist or pathologist."
        )
        return " ".join(parts)

    def generate_pdf(self, output_path: str):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=35, bottomMargin=70)
        story = []

        risk_flag = self.summary.get("risk_flag", "INDETERMINATE")
        ts = self._format_timestamp(self.summary.get("timestamp", ""))
        slide_id = self.summary.get("slide_id", "N/A")
        primary_finding = self.summary.get("primary_finding", "N/A")
        cellularity = self.summary.get("cellularity", "INSUFFICIENT")
        abnormal_ratio = float(self.summary.get("abnormal_ratio", 0.0) or 0.0)

        class_counts = self.data.get("class_counts", {})
        group_counts = self.data.get("clinical_group_counts", {})
        total_cells = int(self.summary.get("total_cells") or sum(group_counts.values()))
        abnormal_cells = int(self.summary.get("abnormal_cells") or group_counts.get("ABNORMAL", 0))
        low_conf_cells = int(self.summary.get("low_confidence_cells") or group_counts.get("BENIGN_UNCERTAIN", 0))

        ai_result = self._ai_screening_result(risk_flag)
        recommendation = self._recommendation(risk_flag)
        adequacy_note = self._adequacy_note(cellularity, total_cells)
        interpretation = self.summary.get("clinical_interpretation") or self._default_interpretation(
            abnormal_ratio,
            class_counts,
            abnormal_cells,
            total_cells,
        )

        # Header
        story.append(Paragraph(report_config.INSTITUTION_NAME, self.styles["InstitutionHeader"]))
        story.append(Paragraph(report_config.REPORT_TITLE, self.styles["Heading1"]))
        story.append(Paragraph(f"Schema Version: {self.data.get('schema_version', report_config.REPORT_SCHEMA_VERSION)}", self.styles["Italic"]))
        story.append(Spacer(1, 10))

        # AI Summary Box
        story.append(Paragraph("AI Summary", self.styles["Heading2"]))
        summary_rows = [
            ["Report ID", self.summary.get("report_id", "N/A")],
            ["Slide ID", slide_id],
            ["Analysis Timestamp", ts],
            ["AI Screening Result", ai_result],
            ["Total Cells Analyzed", str(total_cells)],
            ["Abnormal Cells", str(abnormal_cells)],
            ["Abnormal Cell Ratio", f"{abnormal_ratio:.1%}"],
            ["Recommendation", recommendation],
        ]
        summary_table = Table(summary_rows, colWidths=[170, 340], hAlign="LEFT")
        summary_table_style = [
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FB")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E9EEF5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D2DC")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TEXTCOLOR", (1, 3), (1, 3), self._risk_color(risk_flag)),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        summary_table.setStyle(TableStyle(summary_table_style))
        story.append(summary_table)
        story.append(Spacer(1, 14))

        # Specimen Adequacy
        story.append(Paragraph("Specimen Adequacy", self.styles["Heading2"]))
        story.append(
            Paragraph(
                f"Cellularity: <b>{cellularity}</b>. {adequacy_note}",
                self.styles["ClinicalNote"],
            )
        )
        story.append(Spacer(1, 10))

        # Quantitative Analysis
        story.append(Paragraph("Quantitative Analysis", self.styles["Heading2"]))
        class_rows = [["Cell Classification", "Count", "Clinical Group"]]
        class_order = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cell_type, count in class_order:
            group = self._clinical_group(cell_type)
            class_rows.append([self._humanize_class_name(cell_type), str(count), group])

        if not class_order:
            class_rows.append(["No analyzable classes", "0", "N/A"])

        class_table = Table(class_rows, hAlign="LEFT", colWidths=[220, 80, 150])
        class_table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6EE")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B6C2CF")),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFD")]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]

        for row_idx in range(1, len(class_rows)):
            group_val = class_rows[row_idx][2]
            if group_val == "ABNORMAL":
                class_table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#FDEBEC")))
            elif group_val == "LOW_CONFIDENCE":
                class_table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#FFF8DE")))
            elif group_val == "BENIGN":
                class_table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#ECF8F2")))

        class_table.setStyle(TableStyle(class_table_style))
        story.append(class_table)

        if low_conf_cells > 0:
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(
                    (
                        f"Low Confidence Predictions: {low_conf_cells} cells were not confidently classified and are shown as uncertain candidates. "
                        "These entries should be interpreted with extra caution."
                    ),
                    self.styles["Italic"],
                )
            )

        story.append(Spacer(1, 12))

        # Detected Abnormal Cells
        story.append(Paragraph("Detected Abnormal Cells", self.styles["Heading2"]))
        story.append(
            Paragraph(
                (
                    "The table below lists top AI-selected abnormal-morphology candidates. "
                    "Confidence is a model score and does not represent diagnostic certainty."
                ),
                self.styles["ClinicalNote"],
            )
        )

        evidence_rows = [["Cell ID", "Class", "Confidence", "Coordinates (BBox)"]]
        evidence_cells = self.data.get("top_abnormal_cells", [])
        for cell in evidence_cells[:8]:
            evidence_rows.append(
                [
                    str(cell.get("cell_id", "N/A")),
                    self._humanize_class_name(cell.get("cell_class", "N/A")),
                    f"{float(cell.get('confidence', 0.0)):.2%}",
                    str(cell.get("bbox", [])),
                ]
            )

        if len(evidence_rows) == 1:
            evidence_rows.append(["N/A", "No high-confidence abnormal candidates", "N/A", "N/A"])

        ev_table = Table(evidence_rows, hAlign="LEFT", colWidths=[70, 150, 90, 140])
        ev_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF4")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC7D1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFCFE")]),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(ev_table)

        # Optional visual evidence (if heatmaps are available)
        available_visuals = []
        for cell in evidence_cells[:3]:
            heatmap_path = cell.get("heatmap_path")
            if heatmap_path and Path(heatmap_path).exists():
                available_visuals.append((cell.get("cell_id", "N/A"), Path(heatmap_path)))

        story.append(Spacer(1, 8))
        if available_visuals:
            story.append(Paragraph("Visual Evidence", self.styles["Heading3"]))
            for cell_id, img_path in available_visuals:
                story.append(Paragraph(f"Cell {cell_id} (annotated evidence)", self.styles["Italic"]))
                story.append(Image(str(img_path), width=150, height=150))
                story.append(Spacer(1, 6))
        else:
            story.append(
                Paragraph(
                    "Visual evidence files are not available in this run. Bounding-box coordinates are provided for traceability.",
                    self.styles["Italic"],
                )
            )

        story.append(Spacer(1, 12))

        # Clinical Interpretation
        story.append(Paragraph("Clinical Interpretation (AI-Assisted)", self.styles["Heading2"]))
        story.append(Paragraph(interpretation, self.styles["ClinicalNote"]))
        story.append(Paragraph(f"Primary AI Finding: {primary_finding}", self.styles["ClinicalNote"]))
        story.append(Spacer(1, 10))

        # AI Model Information
        story.append(Paragraph("AI Model Information", self.styles["Heading2"]))
        model_info = {
            "model_name": self.model_info.get("model_name", report_config.MODEL_NAME_DISPLAY),
            "model_type": self.model_info.get("model_type", report_config.MODEL_TYPE),
            "model_version": self.model_info.get("model_version", report_config.MODEL_VERSION),
            "training_dataset": self.model_info.get("training_dataset", report_config.TRAINING_DATASET),
            "accuracy": float(self.model_info.get("accuracy", report_config.METRIC_ACCURACY) or 0.0),
            "precision": float(self.model_info.get("precision", report_config.METRIC_PRECISION) or 0.0),
            "recall": float(self.model_info.get("recall", report_config.METRIC_RECALL) or 0.0),
            "f1": float(self.model_info.get("f1", report_config.METRIC_F1) or 0.0),
            "aggregation_confidence_threshold": float(
                self.model_info.get(
                    "aggregation_confidence_threshold",
                    report_config.AGGREGATION_CONFIDENCE_THRESHOLD,
                )
                or 0.0
            ),
        }

        model_rows = [
            ["Model", model_info["model_name"]],
            ["Model Type", model_info["model_type"]],
            ["Model Version", model_info["model_version"]],
            ["Training Data", model_info["training_dataset"]],
            ["Accuracy", f"{model_info['accuracy']:.2%}"],
            ["Precision", f"{model_info['precision']:.2%}"],
            ["Recall", f"{model_info['recall']:.2%}"],
            ["F1-Score", f"{model_info['f1']:.2%}"],
            ["Confidence Threshold", f"{model_info['aggregation_confidence_threshold']:.2%}"],
        ]
        model_table = Table(model_rows, colWidths=[170, 340], hAlign="LEFT")
        model_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C5D0DB")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(model_table)
        story.append(Spacer(1, 10))

        # Limitations
        story.append(Paragraph("Limitations", self.styles["Heading2"]))
        for idx, item in enumerate(self.limitations, start=1):
            story.append(Paragraph(f"{idx}. {item}", self.styles["ClinicalNote"]))

        story.append(Spacer(1, 10))

        # Strong disclaimer
        story.append(Paragraph("Disclaimer (Mandatory)", self.styles["Heading2"]))
        strong_disclaimer = (
            "This report is generated by an AI-based screening system and is intended for research and "
            "preliminary analysis only. It is not a diagnostic tool. All findings must be reviewed and "
            "confirmed by a qualified cytotechnologist or pathologist before any clinical decision is made."
        )
        story.append(Paragraph(strong_disclaimer, self.styles["StrongDisclaimer"]))

        doc.build(story, onFirstPage=self._draw_footer, onLaterPages=self._draw_footer)
        print(f"Report successfully generated at: {output_path}")


if __name__ == "__main__":
    import os

    json_path = "data/reports/json/predictions_global_report.json"
    pdf_path = "data/reports/pdf/Final_Clinical_Report.pdf"

    if os.path.exists(json_path):
        gen = ClinicalReportGenerator(json_path)
        gen.generate_pdf(pdf_path)
    else:
        print(f"Sample JSON not found at {json_path}. Please run pipeline.py first to generate reports.")