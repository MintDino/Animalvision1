#!/usr/bin/env python3
"""Evaluate v1.1 and compare to v1.0 baseline."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"
DATASET_DIR = BASE_DIR / "dataset"

CANONICAL_CLASS_ORDER = [
    "bear", "cat", "deer", "dog", "elephant", "fox", "giraffe", "horse",
    "lion", "monkey", "panda", "rabbit", "tiger", "wolf", "zebra", "owl",
    "penguin", "shark", "dolphin", "snake"
]


def load_model(model_path, classes_path):
    """Load model and classes."""
    if not model_path.exists() or not classes_path.exists():
        return None, None
    
    model = tf.keras.models.load_model(str(model_path), compile=False)
    with open(classes_path, "r") as f:
        classes = json.load(f)
    
    return model, classes


def get_validation_samples(test_split=0.2):
    """Get validation images (same split as v1.0 baseline)."""
    validation_images = []
    validation_labels = []
    
    for class_idx, class_name in enumerate(CANONICAL_CLASS_ORDER):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            continue
        
        images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        if not images:
            continue
        
        # Use last 20% for validation (matching train.py's validation_split)
        split_idx = int(len(images) * (1 - test_split))
        val_images = images[split_idx:]
        
        for img_path in val_images:
            validation_images.append(img_path)
            validation_labels.append(class_idx)
    
    return validation_images, validation_labels


def evaluate_model(model, classes, validation_images, validation_labels, model_name):
    """Evaluate model on validation set."""
    predictions = []
    correct_labels = []
    
    print(f"\nEvaluating {model_name} on {len(validation_images)} validation images...")
    
    for idx, img_path in enumerate(validation_images):
        if (idx + 1) % 200 == 0:
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
    
    accuracy = accuracy_score(correct_labels, predictions)
    return np.array(predictions), np.array(correct_labels), accuracy


def main():
    print("="*70)
    print("v1.1 EVALUATION & COMPARISON TO v1.0")
    print("="*70)
    
    # Load v1.1 model
    print("\nLoading v1.1 model...")
    v1_1_model_path = MODEL_VERSIONS_DIR / "v1.1_finetuned" / "animal_model.keras"
    v1_1_classes_path = MODEL_VERSIONS_DIR / "v1.1_finetuned" / "classes.json"
    v1_1_model, v1_1_classes = load_model(v1_1_model_path, v1_1_classes_path)
    
    if v1_1_model is None:
        print("ERROR: v1.1 model not found")
        return
    
    print("✓ v1.1 model loaded")
    
    # Get validation data
    print("\nPreparing validation data...")
    val_images, val_labels = get_validation_samples(test_split=0.2)
    print(f"✓ Found {len(val_images)} validation samples")
    
    # Evaluate v1.1
    v1_1_predictions, v1_1_labels, v1_1_accuracy = evaluate_model(
        v1_1_model, v1_1_classes, val_images, val_labels, "v1.1"
    )
    
    # Load v1.0 baseline results
    with open(MODELS_DIR / "v1.0_baseline.json", "r") as f:
        v1_0_results = json.load(f)
    
    print("\n" + "="*70)
    print("COMPARISON: v1.0 vs v1.1")
    print("="*70)
    print(f"\nv1.0 Baseline (current/default):")
    print(f"  Overall Accuracy: {v1_0_results['accuracy']*100:.2f}%")
    
    print(f"\nv1.1 Fine-tuned (experimental):")
    print(f"  Overall Accuracy: {v1_1_accuracy*100:.2f}%")
    
    improvement = v1_1_accuracy - v1_0_results['accuracy']
    print(f"\nChange: {improvement:+.2%}")
    
    if improvement > 0.01:
        print("✓ v1.1 IMPROVED - recommend as new default")
    elif improvement < -0.01:
        print("✗ v1.1 WORSE - keep v1.0 as default, v1.1 as experimental")
    else:
        print("≈ v1.1 COMPARABLE - keep v1.0 as default")
    
    print("\n" + "="*70)
    print("Per-Class Performance Comparison:")
    print("="*70)
    print(f"{'Class':12} {'v1.0 %':>8} {'v1.1 %':>8} {'Change':>8}")
    print("-" * 70)
    
    class_accuracies_v1_1 = {}
    for idx, class_name in enumerate(v1_1_classes):
        mask = v1_1_labels == idx
        if mask.sum() > 0:
            class_acc = (v1_1_predictions[mask] == idx).sum() / mask.sum()
            class_accuracies_v1_1[class_name] = class_acc
            v1_0_acc = v1_0_results['per_class_accuracy'].get(class_name, 0)
            change = (class_acc - v1_0_acc) * 100
            print(f"{class_name:12} {v1_0_acc*100:7.1f}% {class_acc*100:7.1f}% {change:+7.1f}%")
    
    # Save comparison
    comparison_results = {
        "v1.0_accuracy": v1_0_results['accuracy'],
        "v1.1_accuracy": float(v1_1_accuracy),
        "improvement": float(improvement),
        "recommendation": "keep_v1_0_as_default" if improvement < 0 else "use_v1_1",
        "v1_1_per_class": {k: float(v) for k, v in class_accuracies_v1_1.items()},
    }
    
    with open(MODELS_DIR / "v1.1_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to models/v1.1_comparison.json")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
