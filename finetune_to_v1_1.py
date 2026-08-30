#!/usr/bin/env python3
"""Fine-tune v1.0 model to create v1.1."""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"

CANONICAL_CLASS_ORDER = [
    "bear", "cat", "deer", "dog", "elephant", "fox", "giraffe", "horse",
    "lion", "monkey", "panda", "rabbit", "tiger", "wolf", "zebra", "owl",
    "penguin", "shark", "dolphin", "snake"
]


def load_v1_0_model():
    """Load v1.0 model."""
    model_path = MODELS_DIR / "animal_model.keras"
    classes_path = MODELS_DIR / "classes.json"
    
    if not model_path.exists() or not classes_path.exists():
        raise FileNotFoundError("v1.0 model not found")
    
    model = tf.keras.models.load_model(str(model_path), compile=False)
    with open(classes_path, "r") as f:
        classes = json.load(f)
    
    return model, classes


def create_datasets():
    """Create training and validation datasets."""
    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=42,
        label_mode="int",
        class_names=CANONICAL_CLASS_ORDER,
        image_size=(64, 64),
        batch_size=32,
        color_mode="rgb",
    )

    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        validation_split=0.2,
        subset="validation",
        seed=42,
        label_mode="int",
        class_names=CANONICAL_CLASS_ORDER,
        image_size=(64, 64),
        batch_size=32,
        color_mode="rgb",
    )

    # Data augmentation
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),
    ])

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights():
    """Compute class weights for imbalanced dataset."""
    class_counts = {}
    for class_name in CANONICAL_CLASS_ORDER:
        class_dir = DATASET_DIR / class_name
        count = len([p for p in class_dir.iterdir() if p.is_file()])
        class_counts[class_name] = count

    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = {}
    for idx, class_name in enumerate(CANONICAL_CLASS_ORDER):
        count = class_counts[class_name]
        weights[idx] = total / (num_classes * max(count, 1))
    
    print("\nClass distribution:")
    for class_name in CANONICAL_CLASS_ORDER:
        print(f"  {class_name}: {class_counts[class_name]}")
    
    return weights


def fine_tune_model(model, train_ds, val_ds, class_weights):
    """Fine-tune v1.0 model."""
    print("\n" + "="*70)
    print("Fine-tuning v1.0 → v1.1")
    print("="*70)
    
    # Compile for fine-tuning with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Much lower than 1e-3
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "v1.1_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]
    
    # Train for multiple epochs with fine-tuning
    print("\nTraining with v1.0 baseline...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # More epochs than v1.0's 1
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def save_v1_1_model(model, classes, history):
    """Save v1.1 model and training history."""
    print("\n" + "="*70)
    print("Saving v1.1 model")
    print("="*70)
    
    # Create v1.1 version directory
    v1_1_dir = MODEL_VERSIONS_DIR / "v1.1_finetuned"
    v1_1_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = v1_1_dir / "animal_model.keras"
    model.save(str(model_path))
    print(f"✓ Saved model: {model_path}")
    
    # Save classes
    classes_path = v1_1_dir / "classes.json"
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"✓ Saved classes: {classes_path}")
    
    # Save training history
    history_data = {
        "version": "v1.1",
        "epochs": len(history.history["loss"]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
    
    history_path = v1_1_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"✓ Saved training history: {history_path}")
    
    # Also save full history
    full_history_path = v1_1_dir / "full_training_history.json"
    full_history = {key: [float(v) for v in values] 
                   for key, values in history.history.items()}
    with open(full_history_path, "w") as f:
        json.dump(full_history, f, indent=2)
    print(f"✓ Saved full training history: {full_history_path}")
    
    return v1_1_dir, history_data


def main():
    print("="*70)
    print("Fine-tuning v1.0 to create v1.1")
    print("="*70)
    
    # Load v1.0
    print("\nLoading v1.0 model...")
    model, classes = load_v1_0_model()
    print(f"✓ Model loaded")
    print(f"✓ Classes: {classes}")
    
    # Prepare data
    print("\nPreparing training datasets...")
    train_ds, val_ds = create_datasets()
    print("✓ Datasets created")
    
    # Compute class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights()
    
    # Fine-tune
    model, history = fine_tune_model(model, train_ds, val_ds, class_weights)
    
    # Save v1.1
    v1_1_dir, history_data = save_v1_1_model(model, classes, history)
    
    print("\n" + "="*70)
    print("v1.1 Fine-tuning Complete!")
    print("="*70)
    print(f"\nv1.1 Results:")
    print(f"  Final accuracy: {history_data['final_accuracy']*100:.2f}%")
    print(f"  Final val accuracy: {history_data['final_val_accuracy']*100:.2f}%")
    print(f"  Best val accuracy: {history_data['best_val_accuracy']*100:.2f}%")
    print(f"  Trained for {history_data['epochs']} epochs")
    print(f"\nSaved to: {v1_1_dir}")


if __name__ == "__main__":
    main()
