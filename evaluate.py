"""
STEP 5: EVALUATION
============================
Loads the best trained checkpoint and the held-out test split saved by
train.py, then reports accuracy, precision, recall, F1-score, and a
confusion matrix — exactly the metrics named in the project abstract.

Usage:
    python evaluate.py --data_dir data --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset import DeepfakeDataset
from model import MultimodalDeepfakeDetector


def load_test_split(path):
    samples = []
    with open(path) as f:
        for line in f:
            fpath, label = line.strip().rsplit(",", 1)
            samples.append((fpath, int(label)))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--test_split_file", default="checkpoints/test_split.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_samples = load_test_split(args.test_split_file)
    test_ds = DeepfakeDataset(args.data_dir, split_files=test_samples, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MultimodalDeepfakeDetector().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for rgb, freq, labels in test_loader:
            rgb, freq = rgb.to(device), freq.to(device)
            out = model(rgb, freq)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("=== Test Set Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix (rows=actual, cols=predicted) [real, fake]:")
    print(cm)


if __name__ == "__main__":
    main()
