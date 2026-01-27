import requests
import os
import json
from pathlib import Path

# --- Configuration ---
API_URL = "http://localhost:8000/analyze-slide/"
# Pick a few sample images from your SIPAKMED folder
SAMPLE_IMAGE_DIR = Path("test/")
TEST_OUTPUT_DIR = Path("scripts/test_results")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_analyze_slide_json():
    """Test 1: Upload images and get JSON metadata + PDF Link"""
    print("\n--- Test 1: JSON Response & Metadata ---")
    
    # Select first 3 bmp files
    image_files = [f for f in SAMPLE_IMAGE_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.bmp', '.jpeg']]
    image_files = image_files[:3]
    
    if not image_files:
        print("[ERROR] No test images found! Check SAMPLE_IMAGE_DIR.")
        return

    # Prepare the multipart/form-data payload
    files = [('files', (img.name, open(img, 'rb'), 'image/jpeg')) for img in image_files]

    print(f"[>>] Sending {len(image_files)} frames to AI Pipeline...")
    response = requests.post(f"{API_URL}?return_pdf_directly=false", files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"[OK] Success! Slide ID: {result['slide_id']}")
        print(f" Risk Flag: {result['risk_flag']}")
        print(f" Total Cells Found: {result['total_cells']}")
        print(f" PDF URL: {result['pdf_url']}")
        
        # Save JSON result locally
        json_file = TEST_OUTPUT_DIR / f"{result['slide_id']}_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f" Metadata saved to: {json_file}")
    else:
        print(f" Failed: {response.status_code}")
        print(f"Detail: {response.text}")

def test_analyze_slide_pdf_direct():
    """Test 2: Upload images and get the PDF file directly"""
    print("\n--- Test 2: Direct PDF Download ---")
    
    image_files = list(SAMPLE_IMAGE_DIR.glob("*.bmp"))[3:6] # Pick different ones
    files = [('files', (img.name, open(img, 'rb'), 'image/bmp')) for img in image_files]

    print(f"[OK] Requesting direct PDF for {len(image_files)} frames...")
    # Trigger the 'direct' mode we added to main.py
    response = requests.post(f"{API_URL}?return_pdf_directly=true", files=files)

    if response.status_code == 200 and response.headers['content-type'] == 'application/pdf':
        pdf_path = TEST_OUTPUT_DIR / "direct_download_test.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"[OK] Success! PDF downloaded directly to: {pdf_path}")
    else:
        print(f"[ERROR] Failed to get PDF. Status: {response.status_code}")

if __name__ == "__main__":
    print(" Starting AI API Integration Tests...")
    test_analyze_slide_json()
    test_analyze_slide_pdf_direct()
    print("\n Tests Complete.")