import torch
import json
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import Dict, List

from config import CellConfig

try:
    from model import get_model
except ImportError:
    raise ImportError("Could not import 'get_import' from 'model.py'. Ensure your model definition is accessible.")

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CELL INFER - %(levelname)s - %(message)s'
)

class CellInferenceEngine:
    def __init__(self, config: CellConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # load model
        logging.info(f"Loading model: {config.model_name}...")
        self.model = get_model(
            model_name=config.model_name,
            num_classes=config.NUM_CLASSES,
            pretrained=False    # custom weights
        )
        
        # Load Trained Weights
        if config.weights_path.exists():
            checkpoint = torch.load(config.weights_path, map_location=self.device)
            
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Load the weights
            try:
                self.model.load_state_dict(state_dict)
                logging.info(f"Weights loaded successfully from {config.weights_path}")
            except RuntimeError as e:
                logging.warning(f"Direct load failed, attempting to remove 'module.' prefixes...")
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)       
        else:
            logging.error(f"Weights not found at {config.weights_path}!")
            raise FileNotFoundError("Model weights missing.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])        
        
        self.classes = config.classes
        
    def run(self):
        """Main execution loop."""
        json_files = list(self.cfg.INPUT_METADATA_DIR.glob("*.json"))
        
        if not json_files:
            logging.warning(f"No metadata JSON files found in {self.cfg.INPUT_METADATA_DIR}")
            return
        
        logging.info(f"Processing {len(json_files)} slides...")
        
        for json_path in tqdm(json_files, desc="Classifying Cells"):
            self._process_slide(json_path)
            
    def _process_slide(self, json_path: Path):
        """Process a single slide's metadata file."""
        with open(json_path, 'r') as f:
            slide_data = json.load(f)
            
        full_slide_predictions = {
            "filename": slide_data.get("filename"),
            "total_cells": slide_data.get("total_cells"),
            "processed_cells": []
        }
        
        # Iterate through every segmented cell in the frame
        for cell in slide_data.get("cells", []):
            cell_id = cell.get("cell_id")
            crop_filename = cell.get("crop_path")
            crop_path = self.cfg.INPUT_CROPS_DIR /crop_filename
            
            if not crop_path.exists():
                logging.warning(f"Crop missing: {crop_filename}")
                continue
            
            # Run Inference
            prediction = self._infer_single_cell(crop_path)
            
            # Merge existing segmentation data with new classification data
            enriched_cell_data = {**cell, **prediction}
            full_slide_predictions["processed_cells"].append(enriched_cell_data)
            
        # save final output
        output_path = self.cfg.OUTPUT_DIR / f"pred_{json_path.name}"
        with open(output_path, 'w') as f:
            json.dump(full_slide_predictions, f, indent=2)
            
    def _infer_single_cell(self, img_path: Path) -> Dict:
        """Loads image, transforms, predicts, returns probabilities."""
        try:
            # Load & Preprocess
            image = Image.open(img_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                
            # Parse Results
            probs_cpu = probs.cpu().squeeze().tolist()
            top_prob, top_idx = torch.max(probs, dim=1)
            predicted_class = self.classes[int(top_idx.item())]
            
            # specific schema of return
            return {
                "predicted_class": predicted_class,
                "confidence": float(top_prob.item()),
                "class_probs": {
                    cls_name: round(p, 4)
                    for cls_name, p in zip(self.classes, probs_cpu)
                }
            }
            
        except Exception as e:
            logging.error(f"Error inferring {img_path.name}: {e}")
            return {
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "error": str(e)
            }
            
if __name__ == "__main__":
    from config import cell_config
    
    try:
        engine = CellInferenceEngine(cell_config)
        engine.run()
        logging.info("Cell Inference Complete.")
    except Exception as e:
        logging.critical(f"Cell Inference failed: {e}")