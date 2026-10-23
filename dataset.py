"""
STEP 2: DATASET CLASS
============================
This is the "multimodal" part of the pipeline. For every face image we
produce TWO views:

  1. RGB view      -> fed to a spatial CNN branch (learns texture/shape
                       artifacts typical of GAN/diffusion generated faces)
  2. Frequency view -> 2D FFT magnitude spectrum (log-scaled), fed to a
                       second small CNN branch (GAN-generated images
                       often leave periodic artifacts that show up
                       clearly in the frequency domain but are invisible
                       to the naked eye / a plain RGB CNN)

The two views come from the SAME image, so this is a form of
multimodal fusion at the feature level (spatial + frequency).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224


def compute_fft_image(gray_img):
    """Log-scaled magnitude spectrum of a grayscale image, resized/normalized."""
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude = cv2.resize(magnitude, (IMG_SIZE, IMG_SIZE))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split_files=None, train=True):
        """
        root_dir: folder containing real/ and fake/ subfolders of face crops
        split_files: optional list of specific file paths to use (for train/val/test split)
        """
        self.samples = []  # list of (filepath, label)

        if split_files is not None:
            self.samples = split_files
        else:
            for label, cls in enumerate(["real", "fake"]):  # real=0, fake=1
                cls_dir = os.path.join(root_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cls_dir, fname), label))

        if train:
            self.rgb_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path)
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        freq = compute_fft_image(gray)
        freq_tensor = torch.from_numpy(freq).unsqueeze(0)  # 1 x H x W

        rgb_tensor = self.rgb_transform(img_rgb)

        return rgb_tensor, freq_tensor, torch.tensor(label, dtype=torch.long)


def get_all_samples(data_dir):
    """Helper used by train.py to build train/val/test splits."""
    samples = []
    for label, cls in enumerate(["real", "fake"]):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples
