import json
import logging
import time
from pathlib import Path
from tqdm import tqdm  

from aggregate_adaptive import ClinicalAggregator
from config import report_config, SlideReport

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aggregation_audit.log"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger("PipelineRunner")

def run_batch_aggregation():
    """
    Consolidates all cell predictions from multiple JSON frames into a 
    single master list and performs a global slide-level analysis.
    """
    input_dir = report_config.INPUT_PREDICTIONS_DIR
    output_json_dir = report_config.OUTPUT_JSON_DIR
    
    # Use the folder name as the Slide ID 
    slide_id = input_dir.name
    
    logger.info(f"Starting Global Aggregation for slide: {slide_id}")
    logger.info(f"Reading frames from: {input_dir.resolve()}")

    aggregator = ClinicalAggregator(config=report_config)
    frame_files = list(input_dir.glob("*.json"))
    
    if not frame_files:
        logger.error(f"No JSON frames found in {input_dir}!")
        return

    all_detected_cells = []
    frames_successfully_read = 0

    # Batch Collection
    for json_file in tqdm(frame_files, desc="Consolidating Frames"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract the cell list using the 'processed_cells' key
            frame_cells = []
            if isinstance(data, dict) and "processed_cells" in data:
                frame_cells = data["processed_cells"]
            elif isinstance(data, list):
                frame_cells = data
            else:
                logger.warning(f"Skipping {json_file.name}: Unknown JSON structure.")
                continue

            all_detected_cells.extend(frame_cells)
            frames_successfully_read += 1

        except Exception as e:
            logger.error(f"Failed to read frame {json_file.name}: {str(e)}")

    # Single Global Analysis
    if not all_detected_cells:
        logger.error("Global Aggregation failed: No valid cell data collected.")
        return

    try:
        report: SlideReport = aggregator.analyze_slide(slide_id, all_detected_cells)
        
        out_path = output_json_dir / f"{slide_id}_global_report.json"
        aggregator.save_for_pdf(report, str(out_path))

        # frame consolidation and total cell count
        logger.info(
            f"Pipeline Complete. Successfully consolidated {frames_successfully_read}/{len(frame_files)} "
            f"frames into 1 global report for {slide_id} (Total Cells: {len(all_detected_cells)})."
        )

        if report.summary.risk_flag == "HIGH_RISK":
            logger.warning(f"Slide {slide_id} identified as HIGH_RISK.")

    except Exception as e:
        logger.error(f"Global analysis failed for {slide_id}: {str(e)}")

if __name__ == "__main__":
    run_batch_aggregation()