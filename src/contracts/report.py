"""Reporting and validation contracts."""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
import json


@dataclass
class ValidationReport:
    """Contract for preprocessing validation report."""
    
    timestamp: str
    config_hash: str
    total_samples: int
    class_distribution: Dict[str, int]
    preprocessing_params: Dict
    validation_checks: Dict[str, bool]
    sample_images: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def create(cls, config_hash: str, total_samples: int,
               class_dist: Dict[str, int], prep_params: Dict,
               checks: Dict[str, bool]) -> 'ValidationReport':
        """Factory method for creating report."""
        return cls(
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            total_samples=total_samples,
            class_distribution=class_dist,
            preprocessing_params=prep_params,
            validation_checks=checks
        )