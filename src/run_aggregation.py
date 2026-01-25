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
    Orchestrates the aggregation of all slide predictions found in the input directory.
    """
    # Setup Paths from Config
    input_dir = report_config.INPUT_PREDICTIONS_DIR
    output_json_dir = report_config.OUTPUT_JSON_DIR
    
    logger.info(f"Starting Batch Aggregation...")
    logger.info(f"Reading from: {input_dir.resolve()}")
    logger.info(f"Writing to:   {output_json_dir.resolve()}")

    # Initialize Logic
    aggregator = ClinicalAggregator(config=report_config)

    # Find all prediction files
    # Assuming filenames are like "SLIDE_123.json"
    slide_files = list(input_dir.glob("*.json"))
    
    if not slide_files:
        logger.error(f"No JSON files found in {input_dir}!")
        return

    logger.info(f"Found {len(slide_files)} slides to process.")

    # Process Loop
    success_count = 0
    
    for json_file in tqdm(slide_files, desc="Aggregating Slides"):
        slide_id = json_file.stem  # extract "SLIDE_123" from filename
        
        try:
            # Load Raw Predictions
            with open(json_file, 'r') as f:
                predictions = json.load(f)

            # Validation: Handle dictionary wrappers
            if isinstance(predictions, dict):
                if "processed_cells" in predictions:
                    predictions = predictions["processed_cells"]
                elif "predictions" in predictions:
                    predictions = predictions["predictions"]
                else:
                    logger.warning(f"Skipping {slide_id}: JSON is a dict but missing 'processed_cells' key.")
                    continue
                
            # Final check to ensure we have a list before processing
            if not isinstance(predictions, list):
                logger.warning(f"Skipping {slide_id}: JSON format incorrect (expected list).")
                continue

            # Run Adaptive Logic
            report: SlideReport = aggregator.analyze_slide(slide_id, predictions)

            # Save Result
            out_path = output_json_dir / f"{slide_id}_report.json"
            aggregator.save_for_pdf(report, str(out_path))
            
            # Log High Risk cases immediately for audit
            if report.summary.risk_flag == "HIGH_RISK":
                logger.info(f"ALERT: {slide_id} flagged as HIGH_RISK ({report.summary.abnormal_ratio:.2%})")
            
            success_count += 1

        except Exception as e:
            logger.error(f"Failed to process {slide_id}: {str(e)}")

    logger.info(f"Pipeline Complete. Successfully processed {success_count}/{len(slide_files)} slides.")

if __name__ == "__main__":
    run_batch_aggregation()