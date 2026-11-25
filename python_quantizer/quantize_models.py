import os
import json
import time
import requests
import numpy as np
import tensorflow as tf

# Output directory jahan .tflite models aur labels save honge
OUTPUT_DIR = "exported_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_imagenet_labels(save_path):
    """Downloads exact 1000 ImageNet class names matching MobileNetV2 indices (0-999)."""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    print("Downloading ImageNet 1000 class labels...")
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    class_idx = response.json()

    labels = [class_idx[str(i)][1].replace("_", " ") for i in range(1000)]
    with open(save_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")
    print(f"Saved 1000 labels to {save_path}\n")


def get_file_size_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)


def verify_tflite_model(tflite_path, sample_input):
    """Runs a quick sanity check on the exported .tflite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], sample_input)
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    top_index = int(np.argmax(output))
    return top_index, float(output[top_index]), elapsed_ms


def main():
    print("Loading pretrained MobileNetV2 (224x224, ImageNet)...")
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights="imagenet",
    )

    # Save labels.txt
    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    download_imagenet_labels(labels_path)

    # Dummy normalized image tensor [-1, 1] for sanity check
    rng = np.random.default_rng(42)
    sample_input = rng.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)

    configs = [
        ("FP32 (Baseline)", "mobilenet_v2_fp32.tflite", None, None),
        ("FP16 (Half-Precision)", "mobilenet_v2_fp16.tflite", [tf.lite.Optimize.DEFAULT], [tf.float16]),
        ("INT8 (Dynamic Range)", "mobilenet_v2_int8.tflite", [tf.lite.Optimize.DEFAULT], None),
    ]

    results = []

    for label, filename, optimizations, supported_types in configs:
        print(f"Converting {label} -> {filename} ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if optimizations is not None:
            converter.optimizations = optimizations
        if supported_types is not None:
            converter.target_spec.supported_types = supported_types

        tflite_model = converter.convert()
        out_path = os.path.join(OUTPUT_DIR, filename)

        with open(out_path, "wb") as f:
            f.write(tflite_model)

        size_mb = get_file_size_mb(out_path)
        top_idx, top_conf, cpu_ms = verify_tflite_model(out_path, sample_input)
        results.append((label, filename, size_mb, top_idx, cpu_ms))

    # Print Summary Table
    baseline_size = results[0][2]
    print("\n" + "=" * 75)
    print(f"{'Model Variant':<24} | {'File Size':<10} | {'Reduction':<10} | {'Laptop CPU':<10}")
    print("=" * 75)
    for label, filename, size_mb, top_idx, cpu_ms in results:
        reduction = (1 - (size_mb / baseline_size)) * 100
        print(f"{label:<24} | {size_mb:>7.2f} MB | {reduction:>8.1f}% | {cpu_ms:>7.2f} ms")
    print("=" * 75)
    print(f"\nAll files saved inside: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()