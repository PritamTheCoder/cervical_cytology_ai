"""Main pipeline orchestrator."""
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch

from utils.logging import setup_logger
from utils.paths import PathManager
from data.sipakmed import SIPaKMeDLoader
from data.preprocessing import Preprocessor
from data.audit import DatasetAuditor
from contracts.report import ValidationReport


class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: Path, dry_run: bool = False):
        """Initialize pipeline.
        
        Args:
            config_path: Path to pipeline configuration
            dry_run: Run with dummy data only
        """
        self.dry_run = dry_run
        self.config = self._load_config(config_path)
        
        # Setup logging
        log_level = self.config['pipeline']['log_level']
        self.logger = setup_logger('Pipeline', level=log_level)
        
        # Setup paths
        output_dir = Path(self.config['pipeline']['output_dir'])
        dataset_root = Path(self.config['dataset']['root'])
        self.path_manager = PathManager(dataset_root, output_dir)
        
        # Set random seed
        seed = self.config['random_seed']
        self._set_seed(seed)
        
        self.logger.info(f"Pipeline initialized (dry_run={dry_run})")
        
    def _load_config(self, config_path: Path) -> dict:
        """Load pipeline configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _run_dry(self):
        """Execute pipeline with dummy data."""
        self.logger.info("=" * 60)
        self.logger.info("DRY RUN MODE - Using dummy data")
        self.logger.info("=" * 60)
        
        classes = self.config['dataset']['classes']
        target_size = tuple(self.config['preprocessing']['target_size'])
        
        # Create dummy data
        self.logger.info("Stage 1: Data Ingestion (dummy)")
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.logger.info(f"  Created dummy image: shape={dummy_image.shape}")
        
        # Test preprocessor
        self.logger.info("Stage 2: Preprocessing Validation (dummy)")
        preprocessor = Preprocessor(
            target_size=target_size,
            normalize_mean=tuple(self.config['preprocessing']['normalize_mean']),
            normalize_std=tuple(self.config['preprocessing']['normalize_std']),
            seed=self.config['random_seed']
        )
        
        tensor = preprocessor(dummy_image)
        self.logger.info(f"  Preprocessed tensor: shape={tensor.shape}")
        self.logger.info(f"  Config hash: {preprocessor.config_hash}")
        
        # Create dummy report
        report = ValidationReport.create(
            config_hash=preprocessor.config_hash,
            total_samples=100,
            class_dist={cls: 20 for cls in classes},
            prep_params=preprocessor.get_config(),
            checks={'deterministic': True, 'structure_valid': True}
        )
        
        output_path = self.path_manager.get_output_path('validation')
        report_path = output_path / 'dry_run_report.json'
        
        with open(report_path, 'w') as f:
            f.write(report.to_json())
        
        self.logger.info(f"  Saved report: {report_path}")
        self.logger.info("=" * 60)
        self.logger.info("DRY RUN COMPLETE")
        self.logger.info("=" * 60)
    
    def run(self):
        """Execute pipeline stages."""
        if self.dry_run:
            self._run_dry()
            return
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION START")
        self.logger.info("=" * 60)
        
        # Stage 1: Data Ingestion
        self.logger.info("Stage 1: Data Ingestion")
        classes = self.config['dataset']['classes']
        
        # Validate dataset structure
        if not self.path_manager.validate_dataset_structure(classes):
            self.logger.error("Dataset structure validation failed")
            return
        
        # Create preprocessor
        preprocessor = Preprocessor(
            target_size=tuple(self.config['preprocessing']['target_size']),
            normalize_mean=tuple(self.config['preprocessing']['normalize_mean']),
            normalize_std=tuple(self.config['preprocessing']['normalize_std']),
            interpolation=self.config['preprocessing']['interpolation'],
            augment=self.config['augmentation']['enabled'],
            seed=self.config['preprocessing']['seed']
        )
        
        # Load dataset
        dataset = SIPaKMeDLoader(
            root=self.path_manager.root,
            classes=classes,
            split='train',
            transform=None,  # Apply preprocessor separately
            seed=self.config['random_seed']
        )
        
        self.logger.info(f"  Loaded {len(dataset)} samples")
        
        # Stage 2: Preprocessing Validation
        self.logger.info("Stage 2: Preprocessing Validation")
        
        output_dir = self.path_manager.get_output_path('validation')
        auditor = DatasetAuditor(dataset, preprocessor, output_dir)
        
        audit_report = auditor.generate_report()
        
        # Create validation report
        report = ValidationReport.create(
            config_hash=preprocessor.config_hash,
            total_samples=audit_report['total_samples'],
            class_dist=audit_report['class_distribution'],
            prep_params=audit_report['preprocessing_config'],
            checks=audit_report['checks']
        )
        
        # Save report
        report_path = output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            f.write(report.to_json())
        
        self.logger.info(f"  Validation report: {report_path}")
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION COMPLETE")
        self.logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Cervical Cytology AI Pipeline')
    parser.add_argument('--config', type=str, 
                       default='config/pipeline.yaml',
                       help='Path to pipeline configuration')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run with dummy data only')
    
    args = parser.parse_args()
    
    pipeline = Pipeline(
        config_path=Path(args.config),
        dry_run=args.dry_run
    )
    pipeline.run()


if __name__ == '__main__':
    main()