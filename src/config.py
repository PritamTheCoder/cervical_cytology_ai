import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict
from typing import Optional

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


@dataclass
class SegmentationConfig:
    # Paths
    INPUT_DIR: Path = Path("data/Test_APC")
    OUTPUT_DIR: Path = Path("data/segmented")
    
    # Cellpose Settings
    MODEL_TYPE: str = "cyto2"
    DIAMETER: int = 150     # approximate diameter of cells in pixels (Needs Tuning) 
    FLOW_THRESHOLD: float = 0.6     # Strictness of shape ( lower = more strict)
    CELLPROB_THRESHOLD: float = -2.0         # Threshold for cell probability
    
    # Cropping Settings
    CROP_PADDING: int = 15
    MIN_AREA: int = 50
    
    # Hardware
    USE_GPU: bool = False
    
# --- Cell-Centric Inference ---
@dataclass
class CellConfig(Phase2Config):
    """
    Configuration for Phase 4: Integrating Segmentation + Classification.
    Reads crops from segmentation output, runs Inference, saves enriched Metadata.
    """
    # Inputs
    INPUT_METADATA_DIR: Path = Path("data/segmented/metadata")
    INPUT_CROPS_DIR: Path = Path("data/segmented/crops")
    
    # Outputs
    OUTPUT_DIR: Path = Path("data/predictions")
    
    # Inference threshold (optional filter)
    CONFIDENCE_THRESHOLD: float = 0.5
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
# --- REPORTING CONTRACTS (SCHEMAS) ---
@dataclass
class CellEvidence:
    """Represents a single interesting cell to be visualized in the PDF."""
    cell_id:str
    cell_class: str
    confidence: float
    bbox: list[int]
    heatmap_path: Optional[str] = None # For XAI

@dataclass
class ClinicalSummary:
    """Hight-level diagnosis summary."""
    slide_id: str
    timestamp: str
    risk_flag: str 
    primary_finding: str
    cellularity: str
    abnormal_ratio: float
    logic_mode: str
    
@dataclass
class SlideReport:
    """Root object for PDF report generation."""
    summary: ClinicalSummary
    
    # Detailed counts for tables
    class_counts: Dict[str, int]
    clinical_group_counts: Dict[str, int]   # BENIGN, ABNORMAL
    
    # The 'Evidence' - Top N abnormal cells for visual grid
    top_abnormal_cells: list[CellEvidence] = field(default_factory=list)


# --- Phase 6 CONFIGURATION ---

@dataclass
class ReportConfig:
    """
    Configuration for Aggregation and PDF Reporting.
    """
    # --- Paths ---
    # Input : Per cell JSON result
    INPUT_PREDICTIONS_DIR: Path = Path("data/predictions")

    # Output: Final JSON and PDFs reports location
    OUTPUT_JSON_DIR: Path = Path("data/reports/json")
    OUTPUT_PDF_DIR: Path = Path("data/reports/pdf")

    # --- Aggregation logic ---
    # Cell conf: Below this is treated as "Uncertain/Benign" to reduce noise
    AGGREGATION_CONFIDENCE_THRESHOLD: float = 0.75
    
    # If abnormal_ratio > 0.15, flag as HIGH RISK
    HIGH_RISK_RATIO: float = 0.15
    
    # Mapping for aggregation buckets
    CLINICAL_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "im_Superficial_Intermediate": "BENIGN",
        "im_Parabasal": "BENIGN",
        "im_Metaplastic": "BENIGN",
        "im_Dyskeratotic": "ABNORMAL",
        "im_Koilocytotic": "ABNORMAL",
        "background": "IGNORE"
    })
    
    # --- PDF Visuals ---
    REPORT_TITLE: str = "AI-Assisted Cervical Cancer Cytology Analysis Report"
    INSTITUTION_NAME: str = "Cytology-AI Lab"
    
    # Evidence Gallery
    MAX_EVIDENCE_CELLS: int = 12
    GRID_COLUMNS: int = 4
    
    def __post_init__(self):
        self.OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    
# --- EXPORTS ---

# 1. Instantiate specifically
phase1_instance = Phase1Config()
phase2_instance = Phase2Config()
segmentation_config = SegmentationConfig()
cell_config = CellConfig()
report_config = ReportConfig()

# 2. Global Registry (Optional, useful for automated testing loops)
configs = {
    "phase1": phase1_instance,
    "phase2": phase2_instance,
    "segmentation": segmentation_config,
    "cell": cell_config,
    "report": report_config,
}

# 3. Default "CONFIG"
CONFIG = phase1_instance