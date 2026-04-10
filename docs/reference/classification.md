# Classification (`src/infer.py`, `src/model.py`)

## MobileViT Engine (`src/model.py`)

-   **Model**: MobileViT-S (~5.6M parameters).
-   **Why**: Combines the local inductive bias of CNNs with the global receptive field of Vision Transformers. This allows it to capture subtle chromatin textural changes in small regions at near-real-time speeds on edge devices.

## Inference Loops (`src/infer.py`)

Prepares cell-level bounding boxes and feeds them directly through the MobileViT model.

Contains the logic for batch inferencing, evaluating precision/recall/F1-score across all 5 clinical categories, and appending raw softmax probabilities onto the `Cell` data contracts.

## Categories Assessed

1. `Superficial-Intermediate` (Normal)
2. `Parabasal` (Normal)
3. `Metaplastic` (Benign/Reactive)
4. `Koilocytotic` (Low Grade Lesion)
5. `Dyskeratotic` (High Grade Lesion)
