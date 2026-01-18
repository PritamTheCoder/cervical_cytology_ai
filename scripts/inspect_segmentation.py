import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random
from src.config import SegmentationConfig

def visualize_random_sample(n=3):
    cfg = SegmentationConfig()
    overlay_dir = cfg.OUTPUT_DIR / "overlays"
    overlays = list(overlay_dir.glob("*.jpg"))
    
    if not overlays:
        print("No overlays found.Run segment.py first.")
        return
    
    selected = random.sample(overlays, min(n, len(overlays)))
    
    plt.figure(figsize=(15, 5))
    for i, path in enumerate(selected):
        img = cv2.imread(str(path))
        if img is None:
            print("No Image found.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, n,i+1)
        plt.imshow(img)
        plt.title(path.name)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    visualize_random_sample()