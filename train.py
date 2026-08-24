#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

CANONICAL_CLASS_ORDER = [
    "bear",
    "cat",
    "deer",
    "dog",
    "elephant",
    "fox",
    "giraffe",
    "horse",
    "lion",
    "monkey",
    "panda",
    "rabbit",
    "tiger",
    "wolf",
    "zebra",
]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "animal_model.keras"
HISTORY_PATH = MODELS_DIR / "training_history.json"
CLASSES_PATH = MODELS_DIR / "classes.json"


def generate_synthetic_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CANONICAL_CLASS_ORDER:
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in list(class_dir.glob("*")):
            if image_path.is_file():
                image_path.unlink()

        for idx in range(30):
            image = np.full((64, 64, 3), 28, dtype=np.uint8)
            rows = np.linspace(0, 1, 64)
            cols = np.linspace(0, 1, 64)
            yy, xx = np.meshgrid(rows, cols, indexing="ij")
            palette = {
                "bear": (120, 80, 60),
                "cat": (210, 165, 120),
                "deer": (150, 112, 80),
                "dog": (170, 120, 90),
                "elephant": (180, 170, 140),
                "fox": (220, 118, 60),
                "giraffe": (190, 150, 100),
                "horse": (118, 92, 78),
                "lion": (205, 146, 70),
                "monkey": (120, 84, 62),
                "panda": (245, 245, 245),
                "rabbit": (232, 230, 228),
                "tiger": (218, 150, 38),
                "wolf": (130, 138, 150),
                "zebra": (180, 180, 180),
            }[class_name]
            for channel in range(3):
                image[:, :, channel] = np.clip(
                    (np.asarray(palette[channel], dtype=np.float32) * (0.5 + 0.7 * yy))
                    + (20 * np.sin((xx * 12.0) + idx + channel))
                    + (15 * np.cos((yy * 16.0) + idx)),
                    0,
                    255,
                ).astype(np.uint8)

            y_center = xx.shape[0] // 2
            x_center = xx.shape[1] // 2
            for y in range(64):
                for x in range(64):
                    dist = ((x - x_center) ** 2 + (y - y_center) ** 2) ** 0.5
                    if dist < 18 + (idx % 8):
                        image[y, x] = np.clip(
                            np.asarray(palette, dtype=np.int32) + 18, 0, 255
                        ).astype(np.uint8)

            if class_name in {"cat", "lion", "tiger", "fox"}:
                for y in range(18, 48):
                    for x in range(18, 46):
                        if abs(x - 32) < 10 and abs(y - 26) < 15:
                            image[y, x] = np.clip(np.asarray(palette) + 12, 0, 255).astype(np.uint8)
            elif class_name in {"giraffe", "horse", "deer"}:
                for y in range(18, 52):
                    for x in range(22, 42):
                        if abs(x - 32) < 8 and y > 18:
                            image[y, x] = np.clip(np.asarray(palette) + 6, 0, 255).astype(np.uint8)
            elif class_name == "zebra":
                for y in range(16, 52):
                    for x in range(14, 50):
                        if x % 6 == 0:
                            image[y, x] = np.array((0, 0, 0), dtype=np.uint8)
            elif class_name == "panda":
                for y in range(24, 42):
                    for x in range(20, 46):
                        if (x - 32) ** 2 + (y - 32) ** 2 < 100:
                            image[y, x] = np.array((0, 0, 0), dtype=np.uint8)
            elif class_name == "rabbit":
                for y in range(18, 42):
                    for x in range(24, 42):
                        if abs(x - 32) < 6 and abs(y - 30) < 10:
                            image[y, x] = np.array((200, 200, 200), dtype=np.uint8)
            elif class_name == "elephant":
                for y in range(20, 48):
                    for x in range(14, 52):
                        if abs(x - 32) < 12 and y > 24:
                            image[y, x] = np.array((190, 180, 160), dtype=np.uint8)

            image = np.clip(image + np.random.randint(-18, 18, size=(64, 64, 3), dtype=np.int16), 0, 255).astype(np.uint8)
            keras.utils.save_img(class_dir / f"{class_name}_{idx}.png", image)


def discover_class_names():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    available = [entry.name for entry in sorted(DATASET_DIR.iterdir()) if entry.is_dir()]
    missing = [name for name in CANONICAL_CLASS_ORDER if name not in available]
    if missing:
        raise ValueError(f"Missing dataset directories: {', '.join(missing)}")

    ordered = [name for name in CANONICAL_CLASS_ORDER if name in available]
    return ordered


if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
    generate_synthetic_dataset()

CLASS_NAMES = discover_class_names()


def ensure_dataset_ready():
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        count = 0
        if class_dir.exists():
            count = len([p for p in class_dir.iterdir() if p.is_file()])
        class_counts[class_name] = count

    print("Class distribution")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {class_counts[class_name]}")

    missing = [name for name, count in class_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "Dataset is incomplete. Missing images in: " + ", ".join(missing)
        )

    return class_counts


def compute_class_weights(class_counts):
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        count = class_counts[class_name]
        weights[idx] = total / (num_classes * max(count, 1))
    return weights


def build_model():
    # Use pre-trained MobileNetV2 as backbone for transfer learning
    base_model = keras.applications.MobileNetV2(
        input_shape=(64, 64, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze base model weights for initial training
    base_model.trainable = False
    
    inputs = keras.Input(shape=(64, 64, 3), name="input_image")
    
    # Apply data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomBrightness(0.1)(x)
    
    # Rescale and preprocess for MobileNetV2
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    
    # Pass through pre-trained base
    x = base_model(x, training=False)
    
    # Add custom top layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def create_dataset():
    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=42,
        label_mode="int",
        class_names=CLASS_NAMES,
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
        class_names=CLASS_NAMES,
        image_size=(64, 64),
        batch_size=32,
        color_mode="rgb",
    )

    train_ds = train_ds.shuffle(buffer_size=800)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def save_classes_json():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASSES_PATH, "w", encoding="utf-8") as handle:
        json.dump(CLASS_NAMES, handle, indent=2)


def save_history(history):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)


def save_history_dict(history_dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(history_dict, handle, indent=2)


def print_prediction_distribution(model, val_ds):
    counts = {name: 0 for name in CLASS_NAMES}
    total = 0
    for batch_images, batch_labels in val_ds:
        probs = model.predict(batch_images, verbose=0)
        pred_indices = np.argmax(probs, axis=1)
        for pred_index in pred_indices:
            counts[CLASS_NAMES[pred_index]] += 1
            total += 1

    print("\nPrediction distribution:")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {counts[class_name]}")

    most_common = max(counts.values())
    if total > 0 and most_common / total > 0.75:
        print("\nWARNING: MODEL COLLAPSE DETECTED")
        print("The model is predicting almost everything as one class.")
        print("Check dataset quality and training.")


def test_predictions(model, test_dir):
    if not test_dir or not test_dir.exists():
        return

    image_paths = sorted(test_dir.iterdir())
    for image_path in image_paths:
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        image = keras.utils.load_img(str(image_path), target_size=(64, 64))
        image_array = keras.utils.img_to_array(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        probs = model.predict(image_array, verbose=0)[0]
        pred_index = int(np.argmax(probs))
        confidence = float(probs[pred_index] * 100.0)
        print(f"{image_path.name} → {CLASS_NAMES[pred_index]} → {confidence:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train the AnimalVision classifier.")
    parser.add_argument("--test", type=str, help="Optional test image directory")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_counts = ensure_dataset_ready()
    save_classes_json()

    weights_map = compute_class_weights(class_counts)
    train_ds, val_ds = create_dataset()

    model, base_model = build_model()

    # First training phase: frozen base
    checkpoint = keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    print("Phase 1: Training with frozen base model")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=weights_map,
    )

    print("\nPhase 2: Fine-tuning with unfrozen base model")
    base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=weights_map,
        initial_epoch=len(history.history["loss"]),
    )

    print("\nTraining metrics (Phase 1): ")
    for epoch in range(len(history.history["loss"])):
        print(
            f"Epoch {epoch + 1}: "
            f"loss={history.history['loss'][epoch]:.4f}, "
            f"accuracy={history.history['accuracy'][epoch]:.4f}, "
            f"val_loss={history.history['val_loss'][epoch]:.4f}, "
            f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}"
        )

    if history2.history and "loss" in history2.history and len(history2.history["loss"]) > 0:
        print("\nTraining metrics (Phase 2): ")
        for epoch in range(len(history2.history["loss"])):
            print(
                f"Epoch {len(history.history['loss']) + epoch + 1}: "
                f"loss={history2.history['loss'][epoch]:.4f}, "
                f"accuracy={history2.history['accuracy'][epoch]:.4f}, "
                f"val_loss={history2.history['val_loss'][epoch]:.4f}, "
                f"val_accuracy={history2.history['val_accuracy'][epoch]:.4f}"
            )
        
        # Save combined history
        combined_history = {
            "loss": history.history["loss"] + history2.history["loss"],
            "accuracy": history.history["accuracy"] + history2.history["accuracy"],
            "val_loss": history.history["val_loss"] + history2.history["val_loss"],
            "val_accuracy": history.history["val_accuracy"] + history2.history["val_accuracy"],
        }
    else:
        combined_history = history.history

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation result: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}")

    save_history_dict(combined_history)
    print_prediction_distribution(model, val_ds)

    if args.test:
        test_predictions(model, Path(args.test))

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved classes: {CLASSES_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
