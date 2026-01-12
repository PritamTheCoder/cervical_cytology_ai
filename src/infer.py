import argparse
import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


from config import phase2_instance
from model import get_model, load_weights

def get_transforms():
    """Matches the validation transforms"""
    return transforms.Compose([
        transforms.Resize((phase2_instance.img_size, phase2_instance.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=phase2_instance.mean, std=phase2_instance.std),
    ])
    
def predict_single_image(model, image_path, device):
    """Run inference on a single image"""
    transform = get_transforms()
    
    # Open image (supports bmp)
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        score, pred_idx = torch.max(probs, 1)
        
    return phase2_instance.classes[pred_idx.item()], score.item()

def evaluate_directory(model, data_dir, device):
    """Calculate Macro Recall and Accuracy on a test set directory"""
    transform = get_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=phase2_instance.batch_size, shuffle=False, num_workers=2)
    
    # Verify class mapping alignment
    if dataset.classes != phase2_instance.classes:
        print(f"Warning: Dataset classes {dataset.classes} do not match Config {phase2_instance.classes}")
    
    print(f"Evaluating on {len(dataset)} images from {data_dir}...")
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- Phase 2 Acceptance Metrics ---")
    print(classification_report(all_labels, all_preds, target_names=phase2_instance.classes, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: MobileViT Inference")
    parser.add_argument("--weights", type=str, default="./weights/mobilevit_s_sipakmed_stain_normalized.pth", help="Path to .pth model weights")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--test_dir", type=str, help="Path to test directory (ImageFolder structure)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Device handling
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on: {device}")

    # Load Model
    model = get_model(num_classes=phase2_instance.NUM_CLASSES, pretrained=False)
    model = load_weights(model, args.weights, device)
    model.to(device)
    
    if args.image:
        label, score = predict_single_image(model, args.image, device)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {label} ({score:.2%})")
        
    elif args.test_dir:
        evaluate_directory(model, args.test_dir, device)
    else:
        print("Please provide either --image or --test_dir")
    