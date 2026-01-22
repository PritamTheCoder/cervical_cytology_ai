import cv2
import numpy as np
import json
import math
from pathlib import Path
from tqdm import tqdm

class WSISimulator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.wsi_dir = self.output_dir / "pseudo_wsi"
        self.tiles_dir = self.output_dir / "tiles"
        self.meta_dir = self.output_dir / "metadata"
        
        # Ensure directories exist
        for d in [self.wsi_dir, self.tiles_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_images(self, n_images=None):
        """Loads images sorted to ensure the 'slide' looks the same every time."""
        valid_exts = {".bmp", ".jpg", ".png", ".jpeg"}
        files = sorted([
            f for f in self.input_dir.iterdir() 
            if f.suffix.lower() in valid_exts
        ])
        
        if not files:
            raise ValueError(f"No images found in {self.input_dir}")
            
        if n_images and len(files) > n_images:
            files = files[:n_images]
        
        images = []
        names = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"[WARN] Failed to load image: {f.name}")
                continue
            images.append(img)
            names.append(f.name)
            
        print(f"[INFO] Loaded {len(images)} images for WSI generation.")
        return images, names

    def create_pseudo_wsi(self, grid_size: int = 5):
        """
        Creates a 'Mosaic' WSI by placing images in a fixed grid.
        This preserves medical morphology (no warping).
        """
        # We need grid_size * grid_size images
        required_count = grid_size * grid_size
        images, names = self.load_images(n_images=required_count)
        
        # If we don't have enough images, shrink the grid
        if len(images) < required_count:
            new_grid_size = math.ceil(math.sqrt(len(images)))
            print(f"[WARN] Not enough images for {grid_size}x{grid_size}. Adjusting to {new_grid_size}x{new_grid_size}.")
            grid_size = new_grid_size

        if not images:
             raise ValueError("No images loaded. check input directory.")

        # Get dimensions of the first image (Assuming all are roughly same size)
        base_h, base_w, c = images[0].shape
        
        canvas_h = base_h * grid_size
        canvas_w = base_w * grid_size
        
        # Create blank white canvas
        canvas = np.ones((canvas_h, canvas_w, c), dtype=np.uint8) * 255
        
        metadata = {"original_images": [], "wsi_dims": [canvas_w, canvas_h]}

        print(f"[INFO] Stitching {grid_size}x{grid_size} Mosaic WSI...")
        for idx, img in enumerate(tqdm(images)):
            # Resize checks
            if img.shape[:2] != (base_h, base_w):
                img = cv2.resize(img, (base_w, base_h))

            row = idx // grid_size
            col = idx % grid_size
            
            y_off = row * base_h
            x_off = col * base_w
            
            # Place image directly (No Blending/Warping to preserve cell shapes)
            canvas[y_off:y_off+base_h, x_off:x_off+base_w] = img
            
            metadata["original_images"].append({
                "filename": names[idx],
                "grid_pos": (row, col),
                "global_bbox": [x_off, y_off, x_off+base_w, y_off+base_h] # x1, y1, x2, y2
            })

        wsi_path = self.wsi_dir / "simulated_wsi.png"
        cv2.imwrite(str(wsi_path), canvas)
        
        with open(self.meta_dir / "wsi_structure.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[SUCCESS] WSI saved to {wsi_path} ({canvas_w}x{canvas_h})")
        return canvas

    def tile_wsi(self, wsi_image, tile_size: int = 1024, overlap: int = 128):
        """
        Slices the WSI into overlapping tiles for inference.
        """
        h, w, _ = wsi_image.shape
        step = tile_size - overlap
        
        tile_metadata = []
        
        # Calculate steps
        y_steps = range(0, h, step)
        x_steps = range(0, w, step)
        
        print(f"[INFO] Tiling WSI into {tile_size}x{tile_size} patches (Overlap: {overlap})...")
        
        count = 0
        for y in tqdm(y_steps):
            for x in x_steps:
                # Calculate coordinates
                y_start = y
                x_start = x
                y_end = y + tile_size
                x_end = x + tile_size
                
                # Boundary Check: Shift back if we go off the edge
                if y_end > h:
                    y_end = h
                    y_start = h - tile_size
                if x_end > w:
                    x_end = w
                    x_start = w - tile_size
                    
                # Sanity check for very small WSIs
                if y_start < 0: y_start = 0
                if x_start < 0: x_start = 0
                
                tile = wsi_image[y_start:y_end, x_start:x_end]
                
                # Save tile
                tile_filename = f"tile_{y_start}_{x_start}.png"
                tile_path = self.tiles_dir / tile_filename
                cv2.imwrite(str(tile_path), tile)
                
                tile_metadata.append({
                    "tile_id": count,
                    "filename": tile_filename,
                    "abs_coords": [x_start, y_start], # Global coordinates
                    "width": tile_size,
                    "height": tile_size
                })
                count += 1

        # Save the map
        with open(self.meta_dir / "tile_map.json", "w") as f:
            json.dump(tile_metadata, f, indent=2)
            
        print(f"[SUCCESS] Generated {count} tiles in {self.tiles_dir}")