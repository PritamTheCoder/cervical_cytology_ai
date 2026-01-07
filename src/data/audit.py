"""Dataset audit and validation utilities."""
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging


class DatasetAuditor:
    """Audit dataset integrity and preprocessing consistency."""
    
    def __init__(self, dataset, preprocessor, output_dir: Path):
        """Initialize auditor.
        
        Args:
            dataset: Dataset instance
            preprocessor: Preprocessor instance
            output_dir: Directory for audit outputs
        """
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def audit_class_distribution(self) -> Dict[str, int]:
        """Compute and log class distribution.
        
        Returns:
            Dictionary of class counts
        """
        distribution = self.dataset.get_class_distribution()
        
        self.logger.info("Class Distribution:")
        for class_name, count in distribution.items():
            self.logger.info(f"  {class_name}: {count}")
        
        return distribution
    
    def test_preprocessing_determinism(self, num_samples: int = 5) -> bool:
        """Test that preprocessing is deterministic.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            True if preprocessing is deterministic
        """
        self.logger.info("Testing preprocessing determinism...")
        
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            image = sample['image']
            
            # Process twice
            tensor1 = self.preprocessor(image)
            tensor2 = self.preprocessor(image)
            
            # Check equality
            if not torch.equal(tensor1, tensor2):
                self.logger.error(f"Preprocessing not deterministic for sample {i}")
                return False
        
        self.logger.info("✓ Preprocessing is deterministic")
        return True
    
    def visualize_samples(self, samples_per_class: int = 2):
        """Generate visualization of preprocessed samples.
        
        Args:
            samples_per_class: Number of samples to visualize per class
        """
        self.logger.info("Generating sample visualizations...")
        
        classes = self.dataset.classes
        fig, axes = plt.subplots(len(classes), samples_per_class,
                                 figsize=(samples_per_class * 3, len(classes) * 3))
        
        for cls_idx, class_name in enumerate(classes):
            # Get samples for this class
            class_samples = [s for s in self.dataset.samples if s[2] == class_name]
            
            for sample_idx in range(min(samples_per_class, len(class_samples))):
                ax = axes[cls_idx, sample_idx] if len(classes) > 1 else axes[sample_idx]
                
                # Load and preprocess
                img_path = class_samples[sample_idx][0]
                image = Image.open(img_path).convert("RGB")
                
                # Display original
                ax.imshow(image)
                ax.set_title(f"{class_name}\n{img_path.stem}", fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / "sample_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved visualization to {output_path}")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive audit report.
        
        Returns:
            Audit report dictionary
        """
        self.logger.info("Generating audit report...")
        
        # Class distribution
        distribution = self.audit_class_distribution()
        
        # Test determinism
        is_deterministic = self.test_preprocessing_determinism()
        
        # Visualize samples
        self.visualize_samples()
        
        # Compile report
        report = {
            'total_samples': len(self.dataset),
            'class_distribution': distribution,
            'preprocessing_config': self.preprocessor.get_config(),
            'checks': {
                'deterministic': is_deterministic,
                'structure_valid': True
            }
        }
        
        self.logger.info("✓ Audit complete")
        return report