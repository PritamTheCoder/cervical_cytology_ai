"""SIPaKMeD dataset loader."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import logging

class SIPaKMeDLoader(Dataset):
    """Dataset loader for SIPaKMeD cervical cytology images."""
    def __init__(self, root: Path, classes: List[str],
                 split: str = "train", transform=None,
                 seed: int = 42):
        """Initialize dataset loader.
        
        Args:
            root: Dataset root directory
            classes: List of class names (immutable)
            split: Dataset split ('train', 'val', 'test')
            transform: Optional preprocessing transform
            seed: Random seed for reproductibility
        """
        self.root = Path(root)
        self.classes = classes
        self.split = split
        self.transofrm = transform
        self.seed = seed
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[Path, int, str]]:
        """Load all image paths with labels.
            
        Returns:
            List of (image_path, label_idx, class_name) tuples
        """
        samples = []
            
        for class_name in self.classes:
            class_dir = self.root / class_name
                
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.glob("*.bmp"):
                label_idx = self.class_to_idx[class_name]
                samples.append((img_path, label_idx, class_name))
                    
        self.logger.info(f"Loaded {len(samples)} sampels for split '{self.split}'")
        return samples
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample by index.
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, label, and metadata
        """
        img_path, label_idx, class_name = self.samples[idx]
        
        # load image
        image = Image.open(img_path).convert("RGB")
        image_array = np.array(image)
        
        # Apply transform if provided
        if self.transform:
            image_array = self.transform(image_array)
        
        return {
            'image': image_array,
            'label': label_idx,
            'class_name': class_name,
            'image_id': img_path.stem,
            'path': str(img_path)
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Compute class distribution.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = {cls: 0 for cls in self.classes}
        for _, _, class_name in self.samples:
            distribution[class_name] += 1
        return distribution