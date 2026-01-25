import json
import datetime
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from config import report_config

class ClinicalReportGenerator:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        self.summary = self.data['summary']
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()

    def setup_custom_styles(self):
        """Define branding and status colors."""
        self.styles['Heading1'].alignment = 1
        self.styles.add(ParagraphStyle(
            name='Status_HIGH_RISK',
            parent=self.styles['Normal'],
            textColor=colors.red,
            fontSize=12,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='InstitutionHeader',
            parent=self.styles['Heading2'],
            textColor=colors.navy,
            alignment=1 # Center
        ))

    def _draw_footer(self, canvas, doc):
        """Internal helper to draw footer on every page."""
        canvas.saveState()
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(50, 50, 560, 50) # Horizontal line above footer
        
        canvas.setFont('Helvetica-Oblique', 8)
        canvas.setFillColor(colors.grey)
        
        # Left Side: Project Branding
        canvas.drawString(50, 35, f"Medical AI | {report_config.REPORT_TITLE}")
        
        # Right Side: Personal Credit
        canvas.drawRightString(560, 35, "Developed by @PritamTheCoder | GitHub: PritamTheCoder")
        
        # Page Number
        canvas.drawCentredString(300, 20, f"Page {doc.page}")
        canvas.restoreState()
        
    def generate_pdf(self, output_path: str):
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # Header & Branding
        story.append(Paragraph(report_config.INSTITUTION_NAME, self.styles['InstitutionHeader']))
        story.append(Paragraph(report_config.REPORT_TITLE, self.styles['Heading1']))
        story.append(Spacer(1, 12))

        # Case Summary Block
        ts = datetime.datetime.fromisoformat(self.summary['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        summary_data = [
            ["Slide ID:", self.summary['slide_id'], "Date:", ts],
            ["Risk Status:", self.summary['risk_flag'], "Cellularity:", self.summary['cellularity']],
            ["Finding:", self.summary['primary_finding'], "", ""]
        ]
        summary_table = Table(summary_data, colWidths=[80, 200, 60, 150])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1,1), (1,1), colors.red if self.summary['risk_flag'] == "HIGH_RISK" else colors.green),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Quantitative Analysis
        story.append(Paragraph("Quantitative Cell Analysis", self.styles['Heading2']))
        class_data = [["Cell Classification", "Count", "Clinical Group"]]
        
        # Merge mapping info into the table
        for cell_type, count in self.data['class_counts'].items():
            # Clinical mapping from config.py
            group = report_config.CLINICAL_MAPPING.get(cell_type, "UNKNOWN")
            class_data.append([cell_type.replace("im_", ""), str(count), group])

        class_table = Table(class_data, hAlign='LEFT', colWidths=[180, 80, 120])
        class_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ]))
        story.append(class_table)
        story.append(Spacer(1, 20))

        # Clinical Evidence Gallery
        story.append(Paragraph("Top Clinical Evidence (Abnormal Candidates)", self.styles['Heading2']))
        story.append(Paragraph("The following cells were flagged with high confidence as abnormal (Koilocytes/Dyskeratotic).", self.styles['Italic']))
        
        # Table of top cells
        evidence_data = [["Cell ID", "Class", "Confidence", "Coordinates (BBox)"]]
        for cell in self.data['top_abnormal_cells'][:8]: # Limit for brevity
            evidence_data.append([
                str(cell['cell_id']),
                cell['cell_class'].replace("im_", ""),
                f"{cell['confidence']:.2%}",
                str(cell['bbox'])
            ])
        
        ev_table = Table(evidence_data, hAlign='LEFT', colWidths=[60, 120, 80, 180])
        ev_table.setStyle(TableStyle([
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('LINEBELOW', (0,0), (-1,0), 1, colors.black),
        ]))
        story.append(ev_table)

        # Footer / Disclaimer
        story.append(Spacer(1, 40))
        disclaimer = "DISCLAIMER: This is an AI-generated screening report. All high-risk flags must be reviewed by a certified cytotechnologist or pathologist."
        story.append(Paragraph(disclaimer, self.styles['Italic']))

        doc.build(story, onFirstPage=self._draw_footer, onLaterPages=self._draw_footer)
        print(f"Report successfully generated at: {output_path}")

if __name__ == "__main__":
    # Point this to your generated global report
    gen = ClinicalReportGenerator("data/reports/json/predictions_global_report.json")
    gen.generate_pdf("data/reports/pdf/Final_Clinical_Report.pdf")