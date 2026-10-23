"""
STEP 4: TRAINING
============================
Trains the multimodal model on the preprocessed face-crop dataset
(produced by preprocess.py), with an 80/10/10 train/val/test split.

Usage:
    python train.py --data_dir data --epochs 15 --batch_size 32 --lr 1e-4
"""

import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import DeepfakeDataset, get_all_samples
from model import MultimodalDeepfakeDetector


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for rgb, freq, labels in loader:
            rgb, freq, labels = rgb.to(device), freq.to(device), labels.to(device)
            out = model(rgb, freq)
            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Build splits
    all_samples = get_all_samples(args.data_dir)
    labels = [s[1] for s in all_samples]
    train_samples, temp_samples = train_test_split(
        all_samples, test_size=0.2, stratify=labels, random_state=42)
    temp_labels = [s[1] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, stratify=temp_labels, random_state=42)

    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    train_ds = DeepfakeDataset(args.data_dir, split_files=train_samples, train=True)
    val_ds = DeepfakeDataset(args.data_dir, split_files=val_samples, train=False)
    test_ds = DeepfakeDataset(args.data_dir, split_files=test_samples, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Save test split file list so evaluate.py uses the exact same held-out set
    with open(os.path.join(args.checkpoint_dir, "test_split.txt"), "w") as f:
        for path, label in test_samples:
            f.write(f"{path},{label}\n")

    model = MultimodalDeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for rgb, freq, labels_batch in pbar:
            rgb, freq, labels_batch = rgb.to(device), freq.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            out = model(rgb, freq)
            loss = criterion(out, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels_batch.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    # Final test evaluation using the best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_model.pth")))
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}  Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
