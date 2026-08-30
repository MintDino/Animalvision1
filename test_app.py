#!/usr/bin/env python3
"""Comprehensive end-to-end testing of AnimalVision."""

import requests
import time
import subprocess
import os
import signal
import sys
from pathlib import Path

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

def test_endpoints():
    """Test all major endpoints."""
    tests_passed = 0
    tests_total = 0
    
    print("="*70)
    print("ENDPOINT TESTS")
    print("="*70)
    
    # Test /health
    print("\n[1] Testing /health")
    tests_total += 1
    try:
        resp = requests.get("http://localhost:5000/health", timeout=5)
        data = resp.json()
        if data.get("server") and data.get("model_json") and data.get("model_bin"):
            print("  ✓ /health OK")
            tests_passed += 1
        else:
            print(f"  ✗ /health returned unexpected data: {data}")
    except Exception as e:
        print(f"  ✗ /health failed: {e}")
    
    # Test /api/model-versions
    print("\n[2] Testing /api/model-versions")
    tests_total += 1
    try:
        resp = requests.get("http://localhost:5000/api/model-versions", timeout=5)
        data = resp.json()
        versions = data.get("versions", [])
        expected = {"current", "v0.0_initial", "v0.1_improved", "v0.2_checkpoint", "v1.1_finetuned"}
        if set(versions) == expected:
            print(f"  ✓ /api/model-versions OK ({len(versions)} versions found)")
            tests_passed += 1
        else:
            print(f"  ✗ /api/model-versions returned unexpected versions: {versions}")
    except Exception as e:
        print(f"  ✗ /api/model-versions failed: {e}")
    
    # Test /api/debug
    print("\n[3] Testing /api/debug")
    tests_total += 1
    try:
        resp = requests.get("http://localhost:5000/api/debug", timeout=5)
        data = resp.json()
        if data.get("server", {}).get("status") == "online":
            print("  ✓ /api/debug OK")
            tests_passed += 1
        else:
            print(f"  ✗ /api/debug returned unexpected data: {data}")
    except Exception as e:
        print(f"  ✗ /api/debug failed: {e}")
    
    # Test /api/local-predict with current model
    print("\n[4] Testing /api/local-predict (current model)")
    tests_total += 1
    try:
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
            print("  ✗ No test image found")
        else:
            with open(test_image, "rb") as f:
                files = {"image": f}
                data = {"version": "current"}
                resp = requests.post(
                    "http://localhost:5000/api/local-predict",
                    files=files,
                    data=data,
                    timeout=30
                )
            result = resp.json()
            if "predictions" in result and len(result["predictions"]) > 0:
                top_pred = result["predictions"][0]
                print(f"  ✓ /api/local-predict OK (top: {top_pred['label']} {top_pred['confidence']:.1f}%)")
                tests_passed += 1
            else:
                print(f"  ✗ /api/local-predict returned unexpected result: {result}")
    except Exception as e:
        print(f"  ✗ /api/local-predict failed: {e}")
    
    # Test /api/local-predict with v0.0_initial model
    print("\n[5] Testing /api/local-predict (v0.0_initial model)")
    tests_total += 1
    try:
        with open(test_image, "rb") as f:
            files = {"image": f}
            data = {"version": "v0.0_initial"}
            resp = requests.post(
                "http://localhost:5000/api/local-predict",
                files=files,
                data=data,
                timeout=30
            )
        result = resp.json()
        if "predictions" in result and len(result["predictions"]) > 0:
            print(f"  ✓ /api/local-predict (v0.0_initial) OK")
            tests_passed += 1
        else:
            print(f"  ✗ /api/local-predict (v0.0_initial) returned unexpected result: {result}")
    except Exception as e:
        print(f"  ✗ /api/local-predict (v0.0_initial) failed: {e}")
    
    # Test /api/local-predict with v1.1_finetuned model
    print("\n[6] Testing /api/local-predict (v1.1_finetuned model)")
    tests_total += 1
    try:
        with open(test_image, "rb") as f:
            files = {"image": f}
            data = {"version": "v1.1_finetuned"}
            resp = requests.post(
                "http://localhost:5000/api/local-predict",
                files=files,
                data=data,
                timeout=30
            )
        result = resp.json()
        if "predictions" in result and len(result["predictions"]) > 0:
            print(f"  ✓ /api/local-predict (v1.1_finetuned) OK")
            tests_passed += 1
        else:
            print(f"  ✗ /api/local-predict (v1.1_finetuned) returned unexpected result: {result}")
    except Exception as e:
        print(f"  ✗ /api/local-predict (v1.1_finetuned) failed: {e}")
    
    # Test /health endpoint for web model components
    print("\n[7] Testing web model files")
    tests_total += 1
    try:
        resp = requests.get("http://localhost:5000/health", timeout=5)
        data = resp.json()
        has_json = data.get("model_json", False)
        has_bins = data.get("model_bin", False)
        if has_json and has_bins:
            print(f"  ✓ Web model files OK (model.json, bin shards)")
            tests_passed += 1
        else:
            print(f"  ✗ Web model files missing: json={has_json}, bins={has_bins}")
    except Exception as e:
        print(f"  ✗ Web model check failed: {e}")
    
    print("\n" + "="*70)
    print(f"TESTS PASSED: {tests_passed}/{tests_total}")
    print("="*70)
    
    return tests_passed == tests_total

def main():
    print("Starting Flask app...")
    proc = start_app()
    
    try:
        success = test_endpoints()
        sys.exit(0 if success else 1)
    finally:
        print("\nStopping Flask app...")
        stop_app(proc)

if __name__ == "__main__":
    main()
