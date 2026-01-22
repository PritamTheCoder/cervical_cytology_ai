import cv2
import numpy as np
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = PROJECT_ROOT / "tools" / "stitch.py"
TEMP_DIR = PROJECT_ROOT / "tests_temp"
OUTPUT_IMG = TEMP_DIR / "result_wsi.png"

def setup_environment():
    """Creates a temporary directory for test artifacts."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[TEST] Environment created at: {TEMP_DIR}")

def generate_synthetic_data():
    """
    Generates a large 'ground truth' image with random texture (simulating cells)
    and slices it into overlapping tiles and a video.
    
    Why texture? OpenCV's feature matchers (ORB/SIFT) fail on blank images.
    We need high-frequency noise to simulate cellular material.
    """
    print("[TEST] Generating synthetic cell data (overlapping crops)...")
    
    # 1. Create a large "Ground Truth" Slide (800x800)
    # We use random noise to ensure plenty of "features" for the stitcher
    ground_truth = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    
    # Add some large "cells" (circles) to verify geometry visually later
    cv2.circle(ground_truth, (200, 400), 50, (0, 255, 0), -1) # Green cell left
    cv2.circle(ground_truth, (400, 400), 50, (0, 0, 255), -1) # Red cell center
    cv2.circle(ground_truth, (600, 400), 50, (255, 0, 0), -1) # Blue cell right

    # 2. Create Overlapping Crops (simulating a microscope panning right)
    # Crop size: 400x400
    # Overlap: 50%
    crops = []
    # Crop 1: 0-400 (Contains Green)
    crops.append(ground_truth[200:600, 0:400])
    # Crop 2: 200-600 (Contains Green edge + Red)
    crops.append(ground_truth[200:600, 200:600])
    # Crop 3: 400-800 (Contains Red edge + Blue)
    crops.append(ground_truth[200:600, 400:800])

    # 3. Save as Image Tiles
    tiles_dir = TEMP_DIR / "tiles"
    tiles_dir.mkdir()
    for i, crop in enumerate(crops):
        cv2.imwrite(str(tiles_dir / f"frame_{i:03d}.png"), crop)
    
    print(f"[TEST] Saved {len(crops)} overlapping tiles to {tiles_dir}")

    # 4. Save as Video
    video_path = TEMP_DIR / "test_slide_scan.avi"
    height, width, _ = crops[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(video_path), fourcc, 5, (width, height))
    
    # Write frames multiple times to simulate a slow scan
    for crop in crops:
        for _ in range(5): 
            out.write(crop)
    out.release()
    print(f"[TEST] Saved dummy video to {video_path}")
    
    return tiles_dir, video_path

def run_stitcher_tool(arg_list):
    """Executes the tool script as a subprocess."""
    cmd = [sys.executable, str(TOOL_PATH)] + arg_list
    print(f"[TEST] Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[TEST] STITCHING FAILED!")
        print("--- STDERR ---")
        print(result.stderr)
        return False
    
    print("[TEST] Tool execution successful.")
    return True

def validate_output():
    """Checks if the output WSI exists and has reasonable dimensions."""
    if not OUTPUT_IMG.exists():
        print("[TEST] FAILURE: Output file was not created.")
        return False
    
    img = cv2.imread(str(OUTPUT_IMG))
    if img is None:
        print("[TEST] FAILURE: Output file is not a valid image.")
        return False
    
    h, w, _ = img.shape
    print(f"[TEST] Output WSI Dimensions: {w}x{h}")
    
    # We expect the width to be larger than a single tile (400) 
    # but slightly less than sum of widths (1200) due to overlap
    if w > 500 and h > 300:
        print("[TEST] SUCCESS: Dimensions indicate successful stitching.")
        return True
    else:
        print("[TEST] FAILURE: Dimensions look like a single tile, not a stitch.")
        return False

def main():
    print("=== STARTING STITCHING PIPELINE TEST ===")
    
    if not TOOL_PATH.exists():
        print(f"[ERROR] Could not find tool at {TOOL_PATH}")
        sys.exit(1)

    setup_environment()
    tiles_dir, video_path = generate_synthetic_data()
    
    # --- TEST CASE 1: Directory Mode ---
    print("\n--- TEST CASE 1: Stitching from Directory of Images ---")
    output_1 = TEMP_DIR / "wsi_from_dir.png"
    success_1 = run_stitcher_tool([
        "--dir", str(tiles_dir),
        "--output", str(output_1)
    ])
    
    # Update global output for validation
    global OUTPUT_IMG
    OUTPUT_IMG = output_1
    if not success_1 or not validate_output():
        print("[TEST] Directory mode failed.")
        sys.exit(1)

    # --- TEST CASE 2: Video Mode ---
    print("\n--- TEST CASE 2: Stitching from Video ---")
    output_2 = TEMP_DIR / "wsi_from_video.png"
    success_2 = run_stitcher_tool([
        "--video", str(video_path),
        "--n_frames", "3", # We know we created 3 distinct views
        "--output", str(output_2)
    ])
    
    OUTPUT_IMG = output_2
    if not success_2 or not validate_output():
        print("[TEST] Video mode failed.")
        sys.exit(1)

    print("\n=== ALL TESTS PASSED ===")
    print(f"Artifacts remain in {TEMP_DIR} for manual inspection.")

if __name__ == "__main__":
    main()