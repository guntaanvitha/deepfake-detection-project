"""
STEP 6: WEB APPLICATION
============================
A Flask app where a user uploads an image or short video clip and gets
back a Real/Fake prediction with a confidence score. For videos, it
samples several frames, runs the model on each, and averages the
fake-probability across frames (majority-vote style aggregation).

Usage:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify
from facenet_pytorch import MTCNN

from model import MultimodalDeepfakeDetector
from dataset import compute_fft_image
from torchvision import transforms

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 224
CHECKPOINT_PATH = os.path.join(APP_ROOT, "checkpoints", "best_model.pth")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = MTCNN(keep_all=False, device=device)

model = MultimodalDeepfakeDetector().to(device)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("Loaded trained checkpoint.")
else:
    print("WARNING: no checkpoint found at", CHECKPOINT_PATH,
          "- predictions will be random until you train the model (see README Step 4).")
model.eval()

rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_face_crop(face_bgr):
    """Runs the model on a single BGR face crop, returns fake-probability."""
    face_bgr = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    rgb_tensor = rgb_transform(face_rgb).unsqueeze(0).to(device)
    freq = compute_fft_image(gray)
    freq_tensor = torch.from_numpy(freq).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(rgb_tensor, freq_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs[1]  # probability of "fake"


def detect_and_crop_face(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, _ = detector.detect(rgb)
    if boxes is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    x1, y1 = max(0, x1), max(0, y1)
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face


def analyze_image(path):
    frame = cv2.imread(path)
    face = detect_and_crop_face(frame)
    if face is None:
        return {"error": "No face detected in the uploaded image."}
    fake_prob = predict_face_crop(face)
    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = fake_prob if label == "FAKE" else 1 - fake_prob
    return {"label": label, "confidence": round(float(confidence) * 100, 2), "frames_analyzed": 1}


def analyze_video(path, num_samples=15):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return {"error": "Could not read the uploaded video."}

    sample_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames)).astype(int)
    fake_probs = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = detect_and_crop_face(frame)
        if face is not None:
            fake_probs.append(predict_face_crop(face))

    cap.release()

    if not fake_probs:
        return {"error": "No faces detected in the uploaded video."}

    avg_fake_prob = float(np.mean(fake_probs))
    label = "FAKE" if avg_fake_prob >= 0.5 else "REAL"
    confidence = avg_fake_prob if label == "FAKE" else 1 - avg_fake_prob
    return {"label": label, "confidence": round(confidence * 100, 2), "frames_analyzed": len(fake_probs)}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "media" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["media"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        result = analyze_image(save_path)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        result = analyze_video(save_path)
    else:
        return jsonify({"error": "Unsupported file type. Upload JPG/PNG or MP4/AVI/MOV."}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
