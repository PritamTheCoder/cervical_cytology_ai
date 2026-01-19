import cv2
import numpy as np
import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from cellpose import models
from config import SegmentationConfig

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SegmentationEngine:
    def __init__(self, config: SegmentationConfig):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
        
        logging.info(f"Initializing Cellpose model: {config.MODEL_TYPE} on {self.device}")
        self.model = models.Cellpose(gpu=config.USE_GPU, model_type=config.MODEL_TYPE)
        
        # Create output directories
        self.dirs = {
            "masks": config.OUTPUT_DIR / "masks",
            "crops": config.OUTPUT_DIR / "crops",
            "overlays": config.OUTPUT_DIR / "overlays",
            "metadata": config.OUTPUT_DIR / "metadata"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
    def process_directory(self):
        """Main pipeline runner"""
        images = list(self.cfg.INPUT_DIR.glob("*.bmp")) + list(self.cfg.INPUT_DIR.glob("*.jpg"))
        
        if not images:
            logging.error(f"No images found in {self.cfg.INPUT_DIR}")
            return
        
        logging.info(f"Found {len(images)} images to process.")
        
        for img_path in tqdm(images, desc="Segmenting Images"):
            try:
                self._process_single_image(img_path)
            except Exception as e:
                logging.error(f"Failed to process {img_path.name}: {str(e)}")
                
        
    def _process_single_image(self, img_path: Path):
        # Image Load
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError("Image could not be loaded")
        
        # Cellpose expects RBG
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run Inference
        masks, flows, styles, diams = self.model.eval(
            image_rgb,
            diameter=self.cfg.DIAMETER,
            channels=[0,0], # grayscale/no nucleus channel specified
            flow_threshold=self.cfg.FLOW_THRESHOLD,
            cellprob_threshold=self.cfg.CELLPROB_THRESHOLD
        )
        
        # Extract Metadata & Crops
        if isinstance(masks, list):
            masks = masks[0]
            
        cell_count = masks.max()
        if cell_count == 0:
            logging.warning(f"No cells detected in {img_path.name}")
            return
        
        # Save full mask array
        np.save(self.dirs["masks"] / f"{img_path.stem}_mask.npy", masks)
        
        # Option B: Save as .png (Visual, but be careful with scaling if >255 cells)
        # valid_mask = masks.astype(np.uint16) # use uint16 to support up to 65535 cells
        # cv2.imwrite(str(self.dirs["masks"] / f"{img_path.stem}_mask.png"), valid_mask)
        
        image_meta = {
            "filename": img_path.name,
            "total_cells": int(cell_count),
            "cells": []
        }
        
        # Process each detected cell
        for cell_id in range(1, cell_count + 1):
            cell_data = self._extract_cell(image, masks, cell_id, img_path.stem)
            if cell_data:
                image_meta["cells"].append(cell_data)
                
        # Save Metadata
        with open(self.dirs["metadata"] / f"{img_path.stem}.json", "w") as f:
            json.dump(image_meta, f, indent=2)
            
        # save overlay
        self._save_overlay(image, masks, img_path.stem)
        
    def _extract_cell(self, original_image, masks, cell_id, base_name):
        """Crops the individual cell and saves it."""
        # Find where mask == cell_id
        y_indices, x_indices = np.where(masks == cell_id)
        
        if len(y_indices) < self.cfg.MIN_AREA:
            return None # Skip noises
        
        # BBOX calculation
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Add Padding
        pad = self.cfg.CROP_PADDING
        y_min = max(0, y_min - pad)
        y_max = min(original_image.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(original_image.shape[1], x_max + pad)
        
        # Crop
        cell_crop = original_image[y_min:y_max, x_min:x_max]
        
        # save crop
        crop_filename = f"{base_name}_cell_{cell_id}.png"
        cv2.imwrite(str(self.dirs["crops"] / crop_filename), cell_crop)
        
        return {
            "cell_id": int(cell_id),
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "area": int(len(y_indices)),
            "crop_path": crop_filename
        }
        
    def _save_overlay(self, image, masks, base_name):
        """Draws contours on image for visual validation."""
        # Create outlines
        from cellpose.utils import outlines_list
        outlines = outlines_list(masks)
        
        overlay =image.copy()
        
        # Draw contours in Red
        for o in outlines:
            pts = o.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
            
        cv2.imwrite(str(self.dirs["overlays"] / f"{base_name}_overlay.jpg"), overlay)
        

if __name__ == "__main__":
    config = SegmentationConfig()
    engine = SegmentationEngine(config)
    engine.process_directory()