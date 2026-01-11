from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class Phase1Config:
    # Paths
    raw_data_dir: Path = Path("./data/SIPAKMED")
    output_dir: Path = Path("./outputs/phase1_audit")
    
    # Image Contracts
    img_size: int = 224 # Trained img size
    input_channels: int = 3
    
    # Normalization (ImageNet standards)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    # Class Schema
    classes: tuple = (
        "im_Dyskeratotic",
        "im_Koilocytotic",
        "im_Metaplastic",
        "im_Parabasal",
        "im_Superficial_Intermediate",
    )
    
    # seed
    seed: int= 42
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

# Global singleton for config
CONFIG = Phase1Config()