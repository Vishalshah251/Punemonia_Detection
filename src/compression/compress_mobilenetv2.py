"""
compress_mobilenetv2.py
-----------------------
Compress the trained MobileNetV2 model using:
  1. Dynamic-range quantization  (weights → INT8, no calibration data needed)
  2. Full-integer quantization   (weights + activations → INT8, uses test samples)

Run from project root:
    python -m src.compression.compress_mobilenetv2
"""

import os
import time
import numpy as np
import tensorflow as tf

from src.data.tf_dataset import get_generators

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "pneumonia_mobilenetv2.keras")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")

DYNAMIC_OUT = os.path.join(OUTPUT_DIR, "mobilenetv2_dynamic.tflite")
INT8_OUT    = os.path.join(OUTPUT_DIR, "mobilenetv2_int8.tflite")


# ── Helpers ───────────────────────────────────────────────────────────────────

def file_mb(path):
    return os.path.getsize(path) / 1024 ** 2


def evaluate_tflite(tflite_path, test_gen, max_batches=50):
    """Run inference with TFLite interpreter, return (accuracy, ms_per_sample)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    test_gen.reset()
    correct, total, times = 0, 0, []

    for i, (images, labels) in enumerate(test_gen):
        if i >= max_batches:
            break
        for img, label in zip(images, labels):
            sample = img[np.newaxis].astype(inp["dtype"])
            t0 = time.perf_counter()
            interpreter.set_tensor(inp["index"], sample)
            interpreter.invoke()
            times.append((time.perf_counter() - t0) * 1000)
            pred = float(interpreter.get_tensor(out["index"])[0][0])
            correct += int((pred >= 0.5) == label)
            total += 1

    return correct / total, np.mean(times)


def evaluate_keras(model, test_gen, max_batches=50):
    """Run inference with Keras model, return (accuracy, ms_per_sample)."""
    test_gen.reset()
    correct, total, times = 0, 0, []

    for i, (images, labels) in enumerate(test_gen):
        if i >= max_batches:
            break
        t0 = time.perf_counter()
        preds = model.predict(images, verbose=0).ravel()
        ms = (time.perf_counter() - t0) * 1000 / len(images)
        times.extend([ms] * len(images))
        correct += int(np.sum((preds >= 0.5) == labels))
        total += len(labels)

    return correct / total, np.mean(times)


# ── Quantization ──────────────────────────────────────────────────────────────

def dynamic_quantize(model):
    """Quantize weights to INT8, activations stay float32 at runtime."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    with open(DYNAMIC_OUT, "wb") as f:
        f.write(tflite)
    print(f"  Dynamic quant saved → {DYNAMIC_OUT}  ({file_mb(DYNAMIC_OUT):.1f} MB)")


def full_int8_quantize(model, test_gen):
    """Quantize weights + activations to INT8 using test samples for calibration."""
    # Collect 100 samples for calibration
    test_gen.reset()
    samples = []
    for images, _ in test_gen:
        samples.append(images)
        if sum(x.shape[0] for x in samples) >= 100:
            break
    data = np.concatenate(samples)[:100].astype(np.float32)

    def rep_dataset():
        for img in data:
            yield [img[np.newaxis]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite = converter.convert()
    with open(INT8_OUT, "wb") as f:
        f.write(tflite)
    print(f"  Full INT8 saved     → {INT8_OUT}  ({file_mb(INT8_OUT):.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    train_gen, val_gen, test_gen = get_generators()

    print("\nLoading MobileNetV2 model …")
    model = tf.keras.models.load_model(os.path.abspath(MODEL_PATH))

    print("\nApplying compression …")
    dynamic_quantize(model)
    full_int8_quantize(model, test_gen)

    # Save baseline size for comparison
    tmp_path = os.path.join(OUTPUT_DIR, "_tmp_baseline.keras")
    model.save(tmp_path)
    baseline_mb = file_mb(tmp_path)
    os.remove(tmp_path)

    print("\nEvaluating all variants (up to 50 batches each) …")
    base_acc,    base_ms    = evaluate_keras(model, test_gen)
    dynamic_acc, dynamic_ms = evaluate_tflite(DYNAMIC_OUT, test_gen)
    int8_acc,    int8_ms    = evaluate_tflite(INT8_OUT,    test_gen)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"{'Model':<30} {'Size':>7} {'Saved':>7} {'Accuracy':>9} {'ms/img':>7}")
    print("─" * 65)

    rows = [
        ("Baseline (FP32 Keras)",    baseline_mb,         base_acc,    base_ms),
        ("Dynamic-range quant",      file_mb(DYNAMIC_OUT), dynamic_acc, dynamic_ms),
        ("Full INT8 quant",          file_mb(INT8_OUT),    int8_acc,    int8_ms),
    ]
    for name, mb, acc, ms in rows:
        saved = (1 - mb / baseline_mb) * 100
        print(f"{name:<30} {mb:>6.1f}M {saved:>6.1f}% {acc*100:>8.2f}% {ms:>6.2f}ms")

    print("─" * 65)


if __name__ == "__main__":
    main()
