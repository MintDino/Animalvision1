#!/usr/bin/env python3

import base64
import io
import json
import os
from pathlib import Path
from urllib import error as urllib_error
import urllib.request as urllib_request

import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory, request as flask_request
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" / "web_model"
KERAS_MODEL_PATH = BASE_DIR / "models" / "animal_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"
MODEL_VERSIONS_DIR = BASE_DIR / "models" / "versions"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
LOCAL_MODEL_VERSION = os.environ.get("LOCAL_MODEL_VERSION", "current")

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))
local_models = {}
local_classes = {}


def normalize_tfjs_topology(value):
    if isinstance(value, list):
        return [normalize_tfjs_topology(item) for item in value]
    if not isinstance(value, dict):
        return value

    args = value.get("args")
    if isinstance(args, list) and args:
        tensors = args[0]
        kwargs = normalize_tfjs_topology(value.get("kwargs", {}))
        if isinstance(tensors, list):
            return [
                [
                    tensor["config"]["keras_history"][0],
                    tensor["config"]["keras_history"][1],
                    tensor["config"]["keras_history"][2],
                    kwargs,
                ]
                for tensor in tensors
            ]
        if isinstance(tensors, dict) and "config" in tensors:
            history = tensors["config"]["keras_history"]
            return [history[0], history[1], history[2], kwargs]

    normalized = {}
    for key, child in value.items():
        if key == "batch_shape":
            normalized["batchInputShape"] = normalize_tfjs_topology(child)
        elif key == "inbound_nodes":
            normalized["inboundNodes"] = normalize_tfjs_topology(child)
        elif key != "optional":
            normalized[key] = normalize_tfjs_topology(child)
    return normalized


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/openrouter")
def openrouter_page():
    return render_template("openrouter.html")


@app.route("/debug")
def debug_page():
    return render_template("debug.html")


@app.route("/health")
def health():
    model_json = MODEL_DIR / "model.json"
    bin_files = list(MODEL_DIR.glob("*.bin"))
    classes_file = MODEL_DIR / "classes.json"
    return jsonify(
        {
            "server": True,
            "model_json": model_json.exists(),
            "model_bin": len(bin_files) > 0,
            "classes": classes_file.exists(),
            "openrouter": bool(OPENROUTER_API_KEY),
        }
    )


@app.route("/api/debug")
def debug_info():
    model_json = MODEL_DIR / "model.json"
    bin_files = list(MODEL_DIR.glob("*.bin"))
    return jsonify(
        {
            "server": {"status": "online", "debug": app.debug},
            "routes": ["/", "/openrouter", "/debug", "/health"],
            "local_model": {
                "keras": KERAS_MODEL_PATH.exists(),
                "web_model_json": model_json.exists(),
                "web_model_bins": len(bin_files),
                "classes": CLASSES_PATH.exists(),
            },
            "openrouter": {
                "configured": bool(OPENROUTER_API_KEY),
                "model": OPENROUTER_MODEL,
            },
            "cache": {"loaded_versions": list(local_models)},
        }
    )


@app.route("/api/model-versions")
def model_versions():
    versions = ["current"]
    if MODEL_VERSIONS_DIR.exists():
        versions.extend(
            sorted(
                path.name
                for path in MODEL_VERSIONS_DIR.iterdir()
                if path.is_dir()
                and (path / "animal_model.keras").exists()
                and (path / "classes.json").exists()
            )
        )
    return jsonify({"versions": versions})


def get_local_model(version):
    if version not in local_models:
        import tensorflow as tf

        if version == "current":
            model_path = KERAS_MODEL_PATH
            classes_path = CLASSES_PATH
        else:
            version_dir = MODEL_VERSIONS_DIR / version
            model_path = version_dir / "animal_model.keras"
            classes_path = version_dir / "classes.json"
        if not model_path.exists() or not classes_path.exists():
            raise FileNotFoundError(f"Local model version not found: {version}")
        local_models[version] = tf.keras.models.load_model(str(model_path), compile=False)
        with classes_path.open("r", encoding="utf-8") as handle:
            local_classes[version] = json.load(handle)
    return local_models[version], local_classes[version]


@app.route("/api/local-predict", methods=["POST"])
def local_predict():
    image_file = flask_request.files.get("image")
    if image_file is None:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        version = flask_request.form.get("version", "current")
        if version != "current" and Path(version).name != version:
            return jsonify({"error": "Invalid local model version."}), 400
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        image = image.resize((64, 64))
        image_array = np.asarray(image, dtype=np.float32)[None, ...]
        model, classes = get_local_model(version)
        probabilities = model.predict(image_array, verbose=0)[0]
        predictions = [
            {"label": classes[index], "confidence": float(score * 100)}
            for index, score in enumerate(probabilities)
        ]
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        return jsonify({"predictions": predictions})
    except Exception as exc:  # pragma: no cover
        app.logger.exception("Local model prediction failed")
        return jsonify({"error": f"Local model prediction failed: {exc}"}), 500


@app.route("/model/web_model/<path:filename>")
def model_files(filename):
    if filename == "model.json":
        model_json_path = MODEL_DIR / filename
        with model_json_path.open("r", encoding="utf-8") as handle:
            model_data = json.load(handle)
        model_data["modelTopology"] = normalize_tfjs_topology(model_data["modelTopology"])
        return jsonify(model_data)
    return send_from_directory(str(MODEL_DIR), filename)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/api/openrouter", methods=["POST"])
def openrouter():
    if not OPENROUTER_API_KEY:
        return jsonify(
            {
                "error": "OpenRouter is not configured. Set OPENROUTER_API_KEY in the environment.",
                "configured": False,
            }
        ), 503

    image_file = flask_request.files.get("image")
    if image_file is None:
        return jsonify({"error": "No image uploaded."}), 400

    file_data = image_file.read()
    if not file_data:
        return jsonify({"error": "Invalid image file."}), 400

    mimetype = image_file.mimetype or "image/jpeg"
    if not mimetype.startswith("image/"):
        return jsonify({"error": "Invalid image file."}), 400

    encoded = base64.b64encode(file_data).decode("utf-8")
    image_url = f"data:{mimetype};base64,{encoded}"

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Classify the main animal in the image. Return only valid JSON with keys: label and confidence. Use one of these exact animal labels: bear, cat, deer, dog, elephant, fox, giraffe, horse, lion, monkey, panda, rabbit, tiger, wolf, zebra, owl, penguin, shark, dolphin, snake. Confidence must be a number between 0 and 100.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    request_body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=request_body,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AnimalVision",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": f"OpenRouter API error: {body}"}), 500
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"OpenRouter request failed: {exc}"}), 500

    try:
        message = response_data["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = message[0].get("text", "")
        data = json.loads(message)
        label = str(data.get("label", "unknown")).strip().lower()
        confidence = float(data.get("confidence", 0.0))
        if label not in {
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
            "owl",
            "penguin",
            "shark",
            "dolphin",
            "snake",
        }:
            raise ValueError("Invalid animal label returned by OpenRouter.")
        return jsonify({"animal": label, "confidence": max(0.0, min(confidence, 100.0))})
    except Exception:
        return jsonify({"error": "OpenRouter returned an invalid prediction."}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
