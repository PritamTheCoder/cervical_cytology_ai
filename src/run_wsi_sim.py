import argparse
from wsi_simulator.wsi_core import WSISimulator

def main():
    parser = argparse.ArgumentParser(description="Pseudo-WSI Generation & Tiling")
    parser.add_argument("--input_dir", type=str, default="./data/SIPAKMED/im_Dyskeratotic", help="Path to raw cell images (e.g. SIPAKMED classes)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save WSI and tiles.")
    parser.add_argument("--grid_size", type=int, default=4, help="Grid size for stitching (e.g. 4 = 4x4 grid)")
    parser.add_argument("--tile_size", type=int, default=1024, help="Size of inference tiles")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles to prevent cutting cells.")
    
    args = parser.parse_args()
    
    print("--- STARTING WSI GENERATION ---")
    
    sim = WSISimulator(args.input_dir, args.output_dir)
    
    wsi_img = sim.create_pseudo_wsi(grid_size=args.grid_size)
    
    sim.tile_wsi(wsi_img, tile_size=args.tile_size, overlap=args.overlap)
    
    print("--- WSI Generation Complete ---")
    print(f"Artifacts generated in: {args.output_dir}")
    
if __name__ == "__main__":
    main()