"""Deterministic preprocessing pipeline."""
import hashlib
import json
from typing import Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Preprocessor:
    """Deterministic preprocessing for train and inference parity."""
    
    def __init__(self, target_size: Tuple[int, int],
                 normalize_mean: Tuple[float, float, float],
                 normalize_std: Tuple[float, float, float],
                 interpolation: str = "bilinear",
                 augment: bool = False,
                 seed: int = 42):
        """Initialize preprocessor.
        
        Args:
            target_size: Target image dimensions (H, W)
            normalize_mean: Mean for normalization (RGB)
            normalize_std: Std for normalization (RGB)
            interpolation: Interpolation mode
            augment: Enable augmentation (OFF by default)
            seed: Random seed
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.interpolation = self._get_interpolation(interpolation)
        self.augment = augment
        self.seed = seed
        
        # Build transform pipeline
        self.transform = self._build_transform()
        
        # Compute config hash for reproducibility
        self.config_hash = self._compute_config_hash()
        
    def _get_interpolation(self, mode: str):
        """Map interpolation string to torchvision mode."""
        modes = {
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
            "nearest": T.InterpolationMode.NEAREST,
        }
        return modes.get(mode, T.InterpolationMode.BILINEAR)
    
    def _build_transform(self):
        """Build deterministic transform pipeline."""
        transforms = []
        
        # Resize (deterministic)
        transforms.append(T.Resize(self.target_size, 
                                   interpolation=self.interpolation))
        
        # Convert to tensor
        transforms.append(T.ToTensor())
        
        # Normalize (deterministic)
        transforms.append(T.Normalize(mean=self.normalize_mean,
                                      std=self.normalize_std))
        
        return T.Compose(transforms)
    
    def _compute_config_hash(self) -> str:
        """Compute hash of preprocessing configuration."""
        config = {
            'target_size': self.target_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'interpolation': str(self.interpolation),
            'augment': self.augment,
            'seed': self.seed
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply preprocessing to image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transform
        tensor = self.transform(image)
        return tensor
    
    def get_config(self) -> Dict:
        """Get preprocessing configuration."""
        return {
            'target_size': self.target_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'interpolation': str(self.interpolation),
            'augment': self.augment,
            'seed': self.seed,
            'config_hash': self.config_hash
        }