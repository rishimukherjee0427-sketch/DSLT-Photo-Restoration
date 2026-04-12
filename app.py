"""
Flask Backend for DSLT v3 — MPS-aware inference.
"""

import os
import sys
import uuid

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), ".torch_cache")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.filters import apply_filter, FILTERS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_DIR = os.path.join("static", "uploads")
RESTORED_DIR = os.path.join("static", "restored")
FILTERED_DIR = os.path.join("static", "filtered")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

# Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
encoder = None
decoder = None
dslt = None

MODEL_PATH = "results/best_model.pth"


def load_dslt_model():
    global encoder, decoder, dslt
    if not os.path.exists(MODEL_PATH):
        print("⚠️  No model at results/best_model.pth")
        return False

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt = MultiScaleDSLT().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    dslt.load_state_dict(checkpoint['dslt'])

    encoder.eval(); decoder.eval(); dslt.eval()

    print(f"✅ DSLT v3 loaded on {device} (epoch {checkpoint.get('epoch', '?')}, "
          f"PSNR={checkpoint.get('val_psnr', 0):.2f})")
    return True


def restore_image(img_rgb):
    if encoder is None:
        return img_rgb

    h, w = img_rgb.shape[:2]
    img = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        bottleneck, skips = encoder(tensor)
        bn_t, skips_t = dslt(bottleneck, skips)
        correction = decoder(bn_t, skips_t)
        output = torch.clamp(tensor + correction, 0, 1)

    result = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return cv2.resize(result, (w, h))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    original_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    file.save(original_path)

    img = cv2.imread(original_path)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    restored_rgb = restore_image(img_rgb)
    restored_path = os.path.join(RESTORED_DIR, f"{file_id}.jpg")
    cv2.imwrite(restored_path, cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return jsonify({
        "id": file_id,
        "original_url": f"/{original_path}",
        "restored_url": f"/{restored_path}",
    })


@app.route("/apply-filter", methods=["POST"])
def apply_filter_route():
    data = request.json
    file_id = data.get("id")
    filter_name = data.get("filter")
    if not file_id or not filter_name:
        return jsonify({"error": "Missing id or filter"}), 400

    restored_path = os.path.join(RESTORED_DIR, f"{file_id}.jpg")
    if not os.path.exists(restored_path):
        return jsonify({"error": "Image not found"}), 404

    img = cv2.imread(restored_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filtered_rgb = img_rgb if filter_name == "none" else apply_filter(img_rgb, filter_name)

    filtered_path = os.path.join(FILTERED_DIR, f"{file_id}_{filter_name}.jpg")
    cv2.imwrite(filtered_path, cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return jsonify({"filtered_url": f"/{filtered_path}"})


@app.route("/download/<file_id>")
def download(file_id):
    filter_name = request.args.get("filter", "none")
    if filter_name != "none":
        path = os.path.join(FILTERED_DIR, f"{file_id}_{filter_name}.jpg")
    else:
        path = os.path.join(RESTORED_DIR, f"{file_id}.jpg")
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(os.path.abspath(path), mimetype="image/jpeg",
                     as_attachment=True, download_name=f"restored_{file_id}.jpg")


if __name__ == "__main__":
    load_dslt_model()
    app.run(debug=False, host="0.0.0.0", port=8080)
