"""
STEP 1: DATA PREPROCESSING
============================
Takes raw videos (or images) for the two classes ("real" and "fake"),
detects faces frame-by-frame using MTCNN, crops them, and saves
fixed-size face images to disk. These cropped faces are what the
model will actually train on.

Expected raw input layout (you create this yourself from a dataset
such as FaceForensics++, Celeb-DF or DFDC — see README.md Step 1):

    raw_data/
        real/
            video1.mp4
            video2.mp4
            ...
        fake/
            video1.mp4
            video2.mp4
            ...

Output layout (created by this script):

    data/
        real/
            video1_frame0.jpg
            video1_frame5.jpg
            ...
        fake/
            ...

Usage:
    python preprocess.py --input_dir raw_data --output_dir data --frame_gap 10
"""

import os
import cv2
import argparse
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

IMG_SIZE = 224  # final saved face crop size


def extract_faces_from_video(video_path, out_dir, detector, frame_gap=10, max_faces=60):
    """Reads a video, runs face detection every `frame_gap` frames, saves crops."""
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved = 0

    while cap.isOpened() and saved < max_faces:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_gap == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(rgb)
            if boxes is not None:
                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                x1, y1 = max(0, x1), max(0, y1)
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    out_path = os.path.join(out_dir, f"{video_name}_frame{frame_idx}.jpg")
                    cv2.imwrite(out_path, face)
                    saved += 1
        frame_idx += 1

    cap.release()
    return saved


def process_images_folder(input_dir, output_dir, detector):
    """If input is already images (not videos), just detect + crop faces."""
    for fname in tqdm(os.listdir(input_dir), desc=f"Images in {input_dir}"):
        fpath = os.path.join(input_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb)
        if boxes is not None:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            x1, y1 = max(0, x1), max(0, y1)
            face = img[y1:y2, x1:x2]
            if face.size != 0:
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                out_name = os.path.splitext(fname)[0] + "_face.jpg"
                cv2.imwrite(os.path.join(output_dir, out_name), face)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="raw_data/ containing real/ and fake/")
    parser.add_argument("--output_dir", required=True, help="where cropped faces are written")
    parser.add_argument("--frame_gap", type=int, default=10, help="sample every Nth frame")
    parser.add_argument("--max_faces", type=int, default=60, help="max faces per video")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(keep_all=False, device=device)

    for label in ["real", "fake"]:
        in_dir = os.path.join(args.input_dir, label)
        out_dir = os.path.join(args.output_dir, label)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(in_dir):
            print(f"Skipping missing folder: {in_dir}")
            continue

        for fname in tqdm(os.listdir(in_dir), desc=f"{label} videos"):
            fpath = os.path.join(in_dir, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                n = extract_faces_from_video(fpath, out_dir, detector, args.frame_gap, args.max_faces)
                print(f"{fname}: saved {n} faces")
            elif ext in [".jpg", ".jpeg", ".png"]:
                process_images_folder(in_dir, out_dir, detector)
                break  # process_images_folder already loops the whole dir

    print("Done. Cropped faces saved under:", args.output_dir)


if __name__ == "__main__":
    main()
