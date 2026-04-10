# Segmentation (`src/segment.py`)

Handles extracting individual cell crops from raw microscope images.

## Architecture

We rely on **Cellpose**, an advanced instance segmentation framework that generalizes well to new image modalities without extensive fine-tuning.

### Primary Functions

- Converts images to grayscale/normalization if required.
- Invokes `cyto2` model to handle overlapping structures and cytoplasmic artifacts common in Papanicolaou-stained smears.
- Iterates over generated segmentation masks to slice bounding-box cropped images for classification.

### Challenges Solved

Cell clumps present a significant hurdle in cytology. Cellpose's flow-prediction algorithm excels at distinguishing closely packed or overlapping boundaries where traditional watershed or thresholding algorithms fail.
