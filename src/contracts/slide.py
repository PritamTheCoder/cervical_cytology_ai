"""Slide-level data contracts."""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json

@dataclass
class TileMetadata:
    """Contract for tile extracted from whole slide image."""
    
    tile_id: str
    slide_id: str
    coordinates: tuple # (x, y, width, height)
    magnification: float
    tissue_percentage: float
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SlideStats:
    """Contract for aggregated slide-level statistics."""
    
    slide_id: str
    total_tiles: int
    class_distribution: Dict[str, int]
    average_confidence: float
    predicted_diagnosis: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SlideStats':
        """Deserialize from dictionary."""
        return cls(**data)