#!/usr/bin/env python3

import json
import shutil
from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "animal_model.keras"
CLASS_PATH = BASE_DIR / "models" / "classes.json"
TARGET_DIR = BASE_DIR / "model" / "web_model"


def load_model_from_keras():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def verify_classes_json():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Classes file not found: {CLASS_PATH}")
    with open(CLASS_PATH, "r", encoding="utf-8") as handle:
        classes = json.load(handle)
    if not isinstance(classes, list) or len(classes) != 15:
        raise ValueError("classes.json must contain exactly 15 class names.")
    return classes


def find_model_shape(model_json_data):
    stack = [model_json_data]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if "batch_input_shape" in current:
                return current["batch_input_shape"]
            if "batch_shape" in current:
                return current["batch_shape"]
            if "config" in current and isinstance(current["config"], dict):
                config = current["config"]
                if "batch_input_shape" in config:
                    return config["batch_input_shape"]
                if "batch_shape" in config:
                    return config["batch_shape"]
            for value in current.values():
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def find_output_units(model_json_data):
    stack = [model_json_data]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if "class_name" in current and current.get("class_name") == "Dense":
                config = current.get("config", {})
                units = config.get("units")
                if units is not None:
                    return int(units)
            for value in current.values():
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def convert_model(model):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from tensorflowjs.converters import save_keras_model

        inputs = tf.keras.Input(shape=(64, 64, 3), name="image")
        exported = inputs
        source_weights = {}
        for layer in model.layers:
            source_weights[layer.name] = layer.get_weights()
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            # Skip data augmentation and preprocessing layers
            if layer.__class__.__name__.startswith("Random"):
                continue
            if isinstance(layer, tf.keras.layers.Rescaling):
                continue
            exported = layer(exported, training=False)

        inference_model = tf.keras.Model(inputs, exported, name="animal_classifier")
        for layer in inference_model.layers:
            weights = source_weights.get(layer.name)
            if weights:
                layer.set_weights(weights)

        save_keras_model(inference_model, str(TARGET_DIR))
        print("Converted model to TensorFlow.js format.")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"TensorFlow.js conversion failed: {exc}") from exc


def verify_artifacts():
    model_json = TARGET_DIR / "model.json"
    if not model_json.exists():
        raise FileNotFoundError(f"Missing TensorFlow.js model file: {model_json}")

    shard_files = sorted(TARGET_DIR.glob("*.bin"))
    if not shard_files:
        raise FileNotFoundError(f"No TensorFlow.js weight files found in: {TARGET_DIR}")

    classes_file = TARGET_DIR / "classes.json"
    if not classes_file.exists():
        raise FileNotFoundError(f"Missing web classes file: {classes_file}")

    with open(model_json, "r", encoding="utf-8") as handle:
        model_data = json.load(handle)

    input_shape = find_model_shape(model_data)
    if input_shape is None:
        raise ValueError("Could not verify model input shape from model.json.")
    if list(input_shape) != [None, 64, 64, 3]:
        raise ValueError(f"Unexpected input shape: {input_shape}. Expected [None, 64, 64, 3].")

    output_units = find_output_units(model_data)
    if output_units is None:
        raise ValueError("Could not verify model output class count from model.json.")
    if output_units != 15:
        raise ValueError(f"Unexpected output units: {output_units}. Expected 15 classes.")

    print(f"Verified model.json input shape: {input_shape}")
    print(f"Verified output classes: {output_units}")


def main():
    model = load_model_from_keras()
    classes = verify_classes_json()

    convert_model(model)

    target_classes = TARGET_DIR / "classes.json"
    shutil.copy2(CLASS_PATH, target_classes)

    verify_artifacts()

    with open(target_classes, "r", encoding="utf-8") as handle:
        saved_classes = json.load(handle)
    if saved_classes != classes:
        raise ValueError("Saved web classes do not match the trained classes order.")

    print(f"Model conversion complete. Output directory: {TARGET_DIR}")


if __name__ == "__main__":
    main()
