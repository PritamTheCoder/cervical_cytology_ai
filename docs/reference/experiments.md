# Experiments (`notebooks/`)

This directory holds the artifacts for creating, fine-tuning, and managing the initial experiments for `MobileViT-S` using the SIPaKMeD dataset.

## Training the Classifier (`train_mobilevit_s_on_SIPKAMED.ipynb`)

Allows execution and logging for all training runs directly in a browser environment (Jupyter).
This notebook documents:
- Splitting raw datasets.
- Tuning hyperparameters for learning rates.
- Generating epoch accuracy graphs.
- Evaluating the model on the test dataset.
- Saving weights out to `.pth` format for production.
