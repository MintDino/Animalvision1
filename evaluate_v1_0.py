#!/usr/bin/env python3
"""Establish v1.0 baseline metrics."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"

CANONICAL_CLASS_ORDER = [
    "bear", "cat", "deer", "dog", "elephant", "fox", "giraffe", "horse",
    "lion", "monkey", "panda", "rabbit", "tiger", "wolf", "zebra", "owl",
    "penguin", "shark", "dolphin", "snake"
]


def load_v1_0_model():
    """Load v1.0 (current) model."""
    model_path = MODELS_DIR / "animal_model.keras"
    classes_path = MODELS_DIR / "classes.json"
    
    if not model_path.exists() or not classes_path.exists():
        raise FileNotFoundError("v1.0 model not found")
    
    model = tf.keras.models.load_model(str(model_path), compile=False)
    with open(classes_path, "r") as f:
        classes = json.load(f)
    
    return model, classes


def get_validation_samples(test_split=0.2):
    """Get images for validation evaluation (20% split for validation)."""
    validation_images = []
    validation_labels = []
    
    for class_idx, class_name in enumerate(CANONICAL_CLASS_ORDER):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_name} directory not found")
            continue
        
        images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        if not images:
            print(f"WARNING: No images in {class_name}")
            continue
        
        # Use last 20% for validation (matching train.py's validation_split)
        split_idx = int(len(images) * (1 - test_split))
        val_images = images[split_idx:]
        
        for img_path in val_images:
            validation_images.append(img_path)
            validation_labels.append(class_idx)
    
    return validation_images, validation_labels


def evaluate_model(model, classes, validation_images, validation_labels):
    """Evaluate model on validation set."""
    predictions = []
    correct_labels = []
    
    print(f"\nEvaluating v1.0 on {len(validation_images)} validation images...")
    
    for idx, img_path in enumerate(validation_images):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(validation_images)}")
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64))
            img_array = np.asarray(img, dtype=np.float32)[None, ...]
            
            pred = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(pred)
            predictions.append(top_idx)
            correct_labels.append(validation_labels[idx])
        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
    
    return np.array(predictions), np.array(correct_labels)


def print_evaluation_results(predictions, correct_labels, classes):
    """Print detailed evaluation results."""
    accuracy = accuracy_score(correct_labels, predictions)
    
    print("\n" + "="*70)
    print("AnimalVision v1.0 BASELINE EVALUATION")
    print("="*70)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Samples evaluated: {len(correct_labels)}\n")
    
    print("Per-Class Accuracy:")
    print("-" * 70)
    class_accuracies = {}
    for idx, class_name in enumerate(classes):
        mask = correct_labels == idx
        if mask.sum() > 0:
            class_acc = (predictions[mask] == idx).sum() / mask.sum()
            class_accuracies[class_name] = class_acc
            print(f"  {class_name:12} {class_acc * 100:6.2f}% ({mask.sum():3} samples)")
    
    print("\n" + "="*70)
    print("Classification Report:")
    print("="*70)
    print(classification_report(
        correct_labels, predictions,
        target_names=classes,
        digits=3
    ))
    
    # Save baseline results
    baseline_results = {
        "version": "v1.0",
        "accuracy": float(accuracy),
        "samples": len(correct_labels),
        "per_class_accuracy": {classes[i]: float(class_accuracies.get(classes[i], 0)) 
                              for i in range(len(classes))},
        "date": "2026-08-30"
    }
    
    with open(MODELS_DIR / "v1.0_baseline.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    print("\nBaseline results saved to models/v1.0_baseline.json")
    return baseline_results


def main():
    print("="*70)
    print("Establishing v1.0 (Current) Baseline Metrics")
    print("="*70)
    
    # Load model
    print("\nLoading v1.0 model...")
    model, classes = load_v1_0_model()
    print(f"✓ Model loaded (input shape: {model.input_shape})")
    print(f"✓ Classes: {classes}")
    
    # Get validation data
    print("\nPreparing validation data...")
    val_images, val_labels = get_validation_samples(test_split=0.2)
    print(f"✓ Found {len(val_images)} validation samples")
    
    # Evaluate
    predictions, correct_labels = evaluate_model(model, classes, val_images, val_labels)
    
    # Print results
    baseline_results = print_evaluation_results(predictions, correct_labels, classes)
    
    print("\n" + "="*70)
    print("Baseline established! Ready for v1.1 fine-tuning.")
    print("="*70)


if __name__ == "__main__":
    main()
