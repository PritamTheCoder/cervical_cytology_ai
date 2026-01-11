import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader

# ADD PROJECT ROOT TO PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.config import CONFIG
from src.data.dataset import SIPaKMeDDataset
from src.data.transforms import get_transforms, inverse_normalize

def run_audit():
    print(f"--- STARTING PHASE 1 AUDIT ---")
    print(f"Target Data:{CONFIG.raw_data_dir}")
    print(f"Output Audit: {CONFIG.output_dir}")
    
    # Initialize Dataset (Infer Mode)
    dataset = SIPaKMeDDataset(CONFIG.raw_data_dir, transform=get_transforms("infer"))
    
    if len(dataset) == 0:
        print("[FATAL] No images found. Check path and directory structure.")
        return
    
    # Class balance Check
    print("\n[AUDIT 1] Checking Class balance...")
    labels = [s[1] for s in dataset.samples]
    counts = Counter(labels)
    
    total_images = len(dataset)
    print(f"Total Images: {total_images}")
    print(f"{'Class name':<25} | {'Count':<10} | {'Ratio':<10}")
    
    class_imbalance_detected = False
    for idx, class_name in enumerate(CONFIG.classes):
        count = counts.get(idx, 0)
        ratio = count / total_images
        print(f"{class_name:<25} | {count:<10} | {ratio:.2%}")
        
        if ratio < 0.10:    # If any class is < 10% of data
            class_imbalance_detected = True
            
    if class_imbalance_detected:
        print("\n[WARNING] Significant class imbalance detected. Consider weighted loss in Phase 2.")

    # 3. Reproducibility & Integrity Test
    print("\n[AUDIT 2] Checking Reproducibility...")
    idx_to_test = 0
    item1, _, _ = dataset[idx_to_test]
    item2, _, _ = dataset[idx_to_test]
    
    # Assert tensor equality
    if torch.equal(item1, item2):
        print("[OK] Deterministic loading passed (Tensors are identical).")
    else:
        print("[ERROR] DETERMINISTIC LOADING FAILED.")
        return

    # 4. Visual Inspection Grid
    print("\n[AUDIT 3] Generating Visual Inspection Grid...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, labels, paths = next(iter(loader))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Phase 1 Audit: Preprocessed Samples (Size: {CONFIG.img_size}x{CONFIG.img_size})")
    
    for i in range(8):
        ax = axes[i // 4, i % 4]
        # Reverse normalization for display
        img_disp = inverse_normalize(images[i])
        img_disp = np.clip(img_disp.permute(1, 2, 0).numpy(), 0, 1)
        
        ax.imshow(img_disp)
        ax.set_title(f"{CONFIG.classes[labels[i]]}")
        ax.axis("off")
    
    save_path = CONFIG.output_dir / "audit_samples.png"
    plt.savefig(save_path)
    print(f"[OK] Visual samples saved to {save_path}")
    plt.close()

    # 5. Preprocessing Stats Check
    print("\n[AUDIT 4] Checking Tensor Statistics (Normalization Sanity)...")
    mean_val = images.mean()
    std_val = images.std()
    print(f"Batch Mean (should be close to 0): {mean_val:.4f}")
    print(f"Batch Std  (should be close to 1): {std_val:.4f}")
    
    if abs(mean_val) > 0.5 or abs(std_val - 1.0) > 0.5:
        print("[ERROR] Normalization stats look suspicious. Check Mean/Std configs.")
    else:
        print("[OK] Normalization looks correct.")

    print("\n--- PHASE 1 COMPLETE: READY FOR PHASE 2 ---")

if __name__ == "__main__":
    run_audit()