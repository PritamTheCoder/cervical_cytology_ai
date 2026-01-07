"""Cell level data contracts"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import numpy as np

@dataclass
class CellPatch:
    """Contract for individual cell image patch."""
    
    image_id:str
    image_array:np.ndarray # Shape: (H, W, C) or (C, H, W)
    label:str
    label_idx: int
    source_path: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        data = asdict(self)
        data['image_array'] = {
            'shape': self.image_array.shape,
            'dtype': str(self.image_array.dtype)
        }
        return data
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    
@dataclass
class ClassProbabilities:
    """Contract for model predictions."""
    image_id: str
    class_names: List[str]
    probabilities: np.ndarray  # Shape: (num_classes,)
    predicted_class: str
    predicted_idx: int
    confidence: float
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'image_id': self.image_id,
            'class_names': self.class_names,
            'probabilities': self.probabilities.tolist(),
            'predicted_class': self.predicted_class,
            'predicted_idx': self.predicted_idx,
            'confidence': float(self.confidence),
            'metadata': self.metadata or {}
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClassProbabilities':
        """Deserialize from dictionary."""
        data['probabilities'] = np.array(data['probabilities'])
        return cls(**data)