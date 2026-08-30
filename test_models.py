#!/usr/bin/env python3
"""Test all model versions to find best baseline."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"
DATASET_DIR = BASE_DIR / "dataset"

CANONICAL_CLASS_ORDER = [
    "bear", "cat", "deer", "dog", "elephant", "fox", "giraffe", "horse",
    "lion", "monkey", "panda", "rabbit", "tiger", "wolf", "zebra", "owl",
    "penguin", "shark", "dolphin", "snake"
]


def load_model_and_classes(version):
    """Load a model version and its classes."""
    if version == "current":
        model_path = MODELS_DIR / "animal_model.keras"
        classes_path = MODELS_DIR / "classes.json"
    else:
        model_path = MODEL_VERSIONS_DIR / version / "animal_model.keras"
        classes_path = MODEL_VERSIONS_DIR / version / "classes.json"
    
    if not model_path.exists() or not classes_path.exists():
        return None, None
    
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        with open(classes_path, "r") as f:
            classes = json.load(f)
        return model, classes
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None, None


def evaluate_model(version):
    """Evaluate a model on a sample image from each class."""
    print(f"\n{'='*60}")
    print(f"Testing model: {version}")
    print('='*60)
    
    model, classes = load_model_and_classes(version)
    if model is None or classes is None:
        print("  FAILED to load model")
        return None
    
    print(f"  Classes: {classes}")
    print(f"  Model input shape: {model.input_shape}")
    
    correct = 0
    total = 0
    per_class_accuracy = {}
    
    for class_idx, class_name in enumerate(CANONICAL_CLASS_ORDER):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"  {class_name}: MISSING dataset directory")
            continue
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            print(f"  {class_name}: NO images found")
            continue
        
        # Test on first 5 images from the class
        test_images = images[:5]
        class_correct = 0
        
        for img_path in test_images:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((64, 64))
                img_array = np.asarray(img, dtype=np.float32)[None, ...]
                
                pred = model.predict(img_array, verbose=0)[0]
                top_idx = np.argmax(pred)
                
                if classes[top_idx] == class_name:
                    class_correct += 1
                    correct += 1
                total += 1
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
        
        accuracy = (class_correct / len(test_images)) * 100
        per_class_accuracy[class_name] = accuracy
        print(f"  {class_name:12} {class_correct:2}/5 correct ({accuracy:5.1f}%)")
    
    if total > 0:
        overall_acc = (correct / total) * 100
        print(f"\n  Overall accuracy: {overall_acc:.1f}% ({correct}/{total})")
        return overall_acc
    else:
        print("  No images tested")
        return None


def main():
    print("Testing all model versions...")
    
    versions = ["current"]
    if MODEL_VERSIONS_DIR.exists():
        versions.extend(sorted(
            path.name for path in MODEL_VERSIONS_DIR.iterdir()
            if path.is_dir()
        ))
    
    results = {}
    for version in versions:
        acc = evaluate_model(version)
        if acc is not None:
            results[version] = acc
    
    print(f"\n{'='*60}")
    print("SUMMARY - Model Accuracy Ranking")
    print('='*60)
    for version, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {version:45} {acc:6.1f}%")


if __name__ == "__main__":
    main()
