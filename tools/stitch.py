import cv2
import argparse
import os
import sys
import glob
import numpy as np
from pathlib import Path
# Use: python tools/stitch.py --video data/raw/slide_scan_01.mp4 --n_frames 15 --output data/wsi/slide_01.png | python tools/stitch.py --dir data/tiles/slide_01/ --output data/wsi/slide_01.png
if cv2.__version__.split('.')[0] < '4':
    print("CRITICAL WARNING: OpenCV 4.x+ is required for clinical-grade 'SCANS' stitching.")
    print("Your version:", cv2.__version__)
    sys.exit(1)

def extract_frames_from_video(video_path, n_frames, output_temp_dir):
    """
    Intelligently extracts N frames spaced evenly across the video.
    """
    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total video frames: {total_frames}")

    if total_frames < n_frames:
        print("[WARN] Video has fewer frames than requested. Using all frames.")
        step = 1
        n_frames = total_frames
    else:
        step = total_frames // n_frames

    frames_data = []
    count = 0
    saved_count = 0

    # Ensure temp dir exists
    os.makedirs(output_temp_dir, exist_ok=True)

    while cap.isOpened() and saved_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            # Save strictly for verification/debugging
            filename = os.path.join(output_temp_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(filename, frame)
            frames_data.append(frame)
            saved_count += 1
        
        count += 1

    cap.release()
    print(f"[INFO] Extracted {len(frames_data)} frames for stitching.")
    return frames_data

def run_stitching(images, output_path):
    """
    Performs the stitching using OpenCV SCANS mode.
    """
    print("[INFO] Initializing Stitcher in SCANS mode (Planar preservation)...")
    
    # Mode=1 is SCANS (Affine). Mode=0 is PANORAMA (Spherical).
    # We strictly use SCANS for microscopy.
    stitcher = cv2.Stitcher.create(mode=1) 

    print(f"[INFO] Stitching {len(images)} images. This may take memory...")
    status, result = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print(f"[SUCCESS] Stitching complete. Saving to {output_path}...")
        # Use PNG to prevent JPEG compression artifacts which look like chromatin texture
        cv2.imwrite(output_path, result)
        print(f"[DONE] WSI saved: {result.shape[1]}x{result.shape[0]} px")
        return True
    else:
        # Error mapping for debugging
        error_map = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images (insufficient overlap)",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed (features not found)",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera params adjust fail"
        }
        err_msg = error_map.get(status, f"Unknown error code {status}")
        print(f"[FAILURE] Stitching failed: {err_msg}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Microscopy Video/Image Stitcher (WSI Generator)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='Path to microscopy video file (e.g., .mp4, .avi)')
    group.add_argument('--dir', type=str, help='Path to directory containing frame images')

    parser.add_argument('--n_frames', type=int, default=10, help='Number of frames to use (for video input)')
    parser.add_argument('--output', type=str, default='generated_wsi.png', help='Output WSI path (use .png or .tiff)')
    
    args = parser.parse_args()

    images = []

    # Input Handling
    if args.video:
        # Create a temp dir to store debug frames
        temp_debug_dir = os.path.join(os.path.dirname(args.output), "debug_frames")
        images = extract_frames_from_video(args.video, args.n_frames, temp_debug_dir)
    elif args.dir:
        print(f"[INFO] Loading images from {args.dir}...")
        # Sort to ensure sequential overlap usually helps, though feature matching is robust
        file_list = sorted(glob.glob(os.path.join(args.dir, "*")))
        # Filter for image extensions
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        file_list = [f for f in file_list if f.lower().endswith(valid_exts)]
        
        if len(file_list) > args.n_frames:
             print(f"[INFO] Subsampling directory: {len(file_list)} -> {args.n_frames} images")
             step = len(file_list) // args.n_frames
             file_list = file_list[::step][:args.n_frames]

        for f in file_list:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
    
    # Validation
    if len(images) < 2:
        print("[ERROR] Need at least 2 images to stitch.")
        sys.exit(1)

    # Execution
    success = run_stitching(images, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()