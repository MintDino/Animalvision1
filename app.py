#!/usr/bin/env python3

import base64
import json
import os
from pathlib import Path
from urllib import error as urllib_error
import urllib.request as urllib_request

from flask import Flask, jsonify, render_template, send_from_directory, request as flask_request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" / "web_model"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))


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
                        "text": "Classify the main animal in the image. Return only valid JSON with keys: label and confidence. Use one of these exact animal labels: bear, cat, deer, dog, elephant, fox, giraffe, horse, lion, monkey, panda, rabbit, tiger, wolf, zebra. Confidence must be a number between 0 and 100.",
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
        }:
            raise ValueError("Invalid animal label returned by OpenRouter.")
        return jsonify({"animal": label, "confidence": max(0.0, min(confidence, 100.0))})
    except Exception:
        return jsonify({"error": "OpenRouter returned an invalid prediction."}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
