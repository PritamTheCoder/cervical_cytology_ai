import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

# --- Phase 1: Data & Contracts ---
@dataclass
class Phase1Config:
    """Base configuration for Data, Preprocessing & Audit (Phase 1)."""
    # Paths
    raw_data_dir: Path = Path("./data/SIPAKMED")
    output_dir: Path = Path("./outputs/phase1_audit")
    
    # Image Contracts
    img_size: int = 224
    input_channels: int = 3
    
    # Normalization (ImageNet standards)
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Class Schema
    classes: Tuple[str, ...] = (
        "im_Dyskeratotic",
        "im_Koilocytotic",
        "im_Metaplastic",
        "im_Parabasal",
        "im_Superficial_Intermediate",
    )
    
    NUM_CLASSES = int(len(classes))
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# --- Phase 2: Classification Core ---
@dataclass
class Phase2Config(Phase1Config):
    """
    Configuration for Classification Model (Phase 2).
    Inherits Phase 1 settings but overrides outputs and adds model params.
    """
    # Model Architecture
    model_name: str = "mobilevit_s"
    pretrained: bool = True
    
    # Phase 2 Specific Paths
    output_dir: Path = Path("./outputs/phase2_classification")
    weights_path: Path = Path("./weights/mobilevit_s_sipakmed_stain_normalized.pth")
    
    # Training Hyperparameters 
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    epochs: int = 25
    
    # Hardware
    num_workers: int = 2
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    def __post_init__(self):
        # Phase 2 specific directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)


# --- EXPORTS ---

# 1. Instantiate both specifically
phase1_instance = Phase1Config()
phase2_instance = Phase2Config()

# 2. Global Registry (Optional, useful for automated testing loops)
configs: Dict[str, Phase1Config] = {
    "phase1": phase1_instance,
    "phase2": phase2_instance
}

# 3. Default "CONFIG"
CONFIG = phase1_instance