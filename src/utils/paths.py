"""Path management utilities."""
from pathlib import Path
from typing import Dict, Optional

class PathManager:
    """Centralized path management."""
    def __init__(self, root: Path, output_dir: Path):
        """Initialize path manager.
        Args:
            root: Dataset root directory
            output_dir: Pipeline output directory
        """
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        
    def get_data_path(self, subset: str = "") -> Path:
        """Get dataset path."""
        if subset:
            return self.root / subset
        return self.root
    
    def get_output_path(self, stage: str, create: bool = True) -> Path:
        """Get output path for pipeline stage.
        
        Args:
            stage: Pipeline stage name
            create: Whether to create directory
            
        Returns:
            Path to stage output directory.
        """
        path = self.output_dir / stage
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def validate_dataset_structure(self, classes: list) -> bool:
        """validate expected dataset directory structure.
        
        Args:
            claddes: Expected class directory names
            
        Returns:
            True if structure is valid
        """
        if not self.root.exists():
            return False
        
        for cls in classes:
            cls_path = self.root / cls
            if not cls_path.exists() or not cls_path.is_dir():
                return False
        
        return True
        