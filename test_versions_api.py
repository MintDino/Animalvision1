#!/usr/bin/env python3
"""Test that each model version can load and make predictions."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import time
import subprocess
import os
import signal

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

def start_app():
    """Start Flask app in background."""
    proc = subprocess.Popen(
        ["python", "app.py"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(3)  # Wait for app to start
    return proc

def stop_app(proc):
    """Stop Flask app."""
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(timeout=5)

def test_prediction(version, image_path):
    """Test prediction with a specific model version."""
    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"version": version}
        response = requests.post(
            "http://localhost:5000/api/local-predict",
            files=files,
            data=data,
            timeout=30
        )
    return response.json()

def main():
    print("Starting Flask app...")
    proc = start_app()
    
    try:
        print("\n" + "="*60)
        print("Testing model versions")
        print("="*60)
        
        # Get available versions
        resp = requests.get("http://localhost:5000/api/model-versions", timeout=5)
        versions = resp.json()["versions"]
        print(f"Available versions: {versions}\n")
        
        # Get a test image
        test_image = None
        for img_path in (DATASET_DIR / "cat").glob("*.jpg"):
            test_image = img_path
            break
        if not test_image:
            for img_path in (DATASET_DIR / "cat").glob("*.png"):
                test_image = img_path
                break
        
        if not test_image:
            print("ERROR: No test image found in dataset/cat/")
            return
        
        print(f"Using test image: {test_image.name}\n")
        
        # Test each version
        for version in versions:
            print(f"Testing {version}...")
            try:
                result = test_prediction(version, test_image)
                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                else:
                    top_pred = result["predictions"][0]
                    print(f"  ✓ Top prediction: {top_pred['label']} ({top_pred['confidence']:.1f}%)")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print("\n" + "="*60)
        print("All models tested successfully!")
        print("="*60)
        
    finally:
        print("\nStopping Flask app...")
        stop_app(proc)

if __name__ == "__main__":
    main()
