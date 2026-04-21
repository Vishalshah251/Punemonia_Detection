"""
app.py — Flask API for Pneumonia X-ray classification + GradCAM
Routes:
  POST /predict  — returns {label, confidence, raw}
  POST /gradcam  — returns {label, confidence, heatmap_b64 (PNG overlay)}
  GET  /health   — returns {status, model, size_mb}
"""

import io
import sys
import os
import base64
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.transfer_learning import build_transfer_model

app = Flask(__name__)
CORS(app)

# ── Load model & build GradCAM graph once ─────────────────────────────────────
print("Loading model...")
_weights = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "models", "pneumonia_mobilenetv2.keras")
)
_base_model = build_transfer_model(trainable_base=False)
_base_model.load_weights(_weights)
_ = _base_model(tf.zeros([1, 224, 224, 1]), training=False)

# TFLite interpreter for fast prediction
_tflite_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "models", "mobilenetv2_dynamic.tflite")
)
_interpreter = tf.lite.Interpreter(model_path=_tflite_path)
_interpreter.allocate_tensors()
_inp_detail  = _interpreter.get_input_details()[0]
_out_detail  = _interpreter.get_output_details()[0]

# Flat Keras model for GradCAM (RGB input → [conv_features, prediction])
_base    = _base_model.get_layer("mobilenetv2_1.00_224")
_gap     = _base_model.get_layer("gap")
_dense1  = _base_model.get_layer("dense_256")
_drop1   = _base_model.get_layer("drop_1")
_dense2  = _base_model.get_layer("dense_128")
_drop2   = _base_model.get_layer("drop_2")
_out_lyr = _base_model.get_layer("output")

_rgb_inp  = tf.keras.Input(shape=(224, 224, 3))
_base_out = _base(_rgb_inp, training=False)
_x = _gap(_base_out); _x = _dense1(_x); _x = _drop1(_x)
_x = _dense2(_x); _x = _drop2(_x); _x = _out_lyr(_x)
_gradcam_model = tf.keras.Model(_rgb_inp, [_base_out, _x])

print(f"TFLite model loaded: {os.path.getsize(_tflite_path)/1024**2:.1f} MB")
print("GradCAM model ready.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def preprocess_grayscale(image_bytes: bytes) -> np.ndarray:
    """bytes → (1, 224, 224, 1) float32 normalised 0-1."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, :, :, np.newaxis]


def preprocess_rgb(image_bytes: bytes) -> np.ndarray:
    """bytes → (1, 224, 224, 3) float32 — grayscale repeated across channels."""
    arr = preprocess_grayscale(image_bytes)          # (1,224,224,1)
    return np.repeat(arr, 3, axis=-1)               # (1,224,224,3)


def run_tflite(input_arr: np.ndarray) -> float:
    """Run compressed TFLite model, return raw sigmoid output."""
    _interpreter.set_tensor(_inp_detail["index"], input_arr)
    _interpreter.invoke()
    return float(_interpreter.get_tensor(_out_detail["index"])[0][0])


def compute_gradcam(rgb_arr: np.ndarray) -> np.ndarray:
    """
    rgb_arr: (1, 224, 224, 3)
    Returns: (224, 224, 3) uint8 overlay image (original blended with heatmap).
    """
    with tf.GradientTape() as tape:
        conv_out, preds = _gradcam_model(rgb_arr, training=False)
        loss = preds[:, 0]
    grads   = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam     = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam     = tf.nn.relu(cam).numpy()

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # Resize + smooth for clean heatmap
    cam_resized = cv2.resize(cam, (224, 224))
    cam_blurred = cv2.GaussianBlur(cam_resized, (11, 11), 0)

    # Apply JET colormap via OpenCV (better quality than matplotlib)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_blurred), cv2.COLORMAP_JET)

    # Original grayscale → BGR
    original     = (rgb_arr[0, :, :, 0] * 255).astype(np.uint8)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Blend: 55% original + 45% heatmap
    overlay     = cv2.addWeighted(original_bgr, 0.55, heatmap_bgr, 0.45, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def image_to_b64(arr: np.ndarray) -> str:
    """numpy (H,W,3) uint8 → base64-encoded PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    input_arr   = preprocess_grayscale(image_bytes)
    prediction  = run_tflite(input_arr)

    label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"

    return jsonify({"label": label})


@app.route("/gradcam", methods=["POST"])
def gradcam():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()

    # Prediction (TFLite — fast)
    input_gray = preprocess_grayscale(image_bytes)
    prediction = run_tflite(input_gray)
    label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"

    # GradCAM (Keras — for explainability)
    input_rgb   = preprocess_rgb(image_bytes)
    overlay_arr = compute_gradcam(input_rgb)
    heatmap_b64 = image_to_b64(overlay_arr)

    return jsonify({
        "label":       label,
        "heatmap_b64": heatmap_b64,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model":    "mobilenetv2_dynamic.tflite",
        "size_mb":  round(os.path.getsize(_tflite_path) / 1024 ** 2, 1),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
