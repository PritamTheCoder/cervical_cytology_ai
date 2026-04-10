# Getting Started

Follow these steps to set up the **Cervical Cytology AI** project on your local machine.

## 1. Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.9+** (preferably 3.10 to 3.13)
- **Git**
- A **CUDA-compatible GPU** (highly recommended for faster training and inference)

## 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/PritamTheCoder/cervical_cytology_ai.git
cd cervical_cytology_ai
pip install -r requirements.txt
```

> **Note**: For PDF report generation, we explicitly use `reportlab==4.1.0`.

## 3. Dataset Setup

This project uses the **SIPaKMeD Database** to train the cell classifier.

1. Download the dataset.
2. Extract the dataset into the `data/` directory.
3. Ensure the structure uses class-wise folders (no pre-split Train/Test directories inside the main folder):

```text
data/
  SIPAKMED/
    im_Dyskeratotic/
    im_Koilocytotic/
    im_Metaplastic/
    im_Parabasal/
    im_Superficial_Intermediate/
```

## 4. Downloading Model Weights

The classification portion requires trained `MobileViT-S` weights.

- **Option A**: Use pre-trained weights. Download the checkpoint and place it in the `weights/` folder. The expected file name is generally `mobilevit_s_sipakmed_stain_normalized.pth` (or similar, depending on your training run).
  *Check out our HuggingFace repository for pre-trained weights if available.*
- **Option B**: Train your own weights from scratch by running the training scripts from the [Usage Guide](usage.md).

## Next Steps

Once the environment is installed and data is available, move on to the [Usage Guide](usage.md) to run the API, start the training pipeline, or generate clinical reports.
