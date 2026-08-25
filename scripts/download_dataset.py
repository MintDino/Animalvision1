#!/usr/bin/env python3

"""Download a balanced AnimalVision dataset from the iNaturalist API."""

import argparse
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from urllib import error, parse, request

from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
API_URL = "https://api.inaturalist.org/v1"
TARGET_TOTAL = 5000
MIN_WIDTH = 64
MIN_HEIGHT = 64
CLASS_NAMES = [
    "bear", "cat", "deer", "dog", "elephant", "fox", "giraffe", "horse",
    "lion", "monkey", "panda", "rabbit", "tiger", "wolf", "zebra",
    "owl", "penguin", "shark", "dolphin", "snake",
]
PER_CLASS = TARGET_TOTAL // len(CLASS_NAMES)
EXTRA_IMAGES = TARGET_TOTAL % len(CLASS_NAMES)

# Scientific taxa avoid ambiguous common-name matches in the public API.
TAXON_QUERIES = {
    "bear": "Ursidae", "cat": "Felis catus", "deer": "Cervidae",
    "dog": "Canis lupus familiaris", "elephant": "Elephantidae", "fox": "Vulpes",
    "giraffe": "Giraffa camelopardalis", "horse": "Equus caballus",
    "lion": "Panthera leo", "monkey": "Simiiformes",
    "panda": "Ailuropoda melanoleuca", "rabbit": "Leporidae",
    "tiger": "Panthera tigris", "wolf": "Canis lupus", "zebra": "Equus quagga",
    "owl": "Strigiformes", "penguin": "Sphenisciformes", "shark": "Selachimorpha",
    "dolphin": "Delphinidae", "snake": "Serpentes",
}


def target_for(class_name):
    return PER_CLASS + (1 if CLASS_NAMES.index(class_name) < EXTRA_IMAGES else 0)


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return value[:100] or "image"


def http_json(path, params=None, attempts=4):
    query = parse.urlencode(params or {})
    url = f"{API_URL}/{path}?{query}"
    for attempt in range(attempts):
        try:
            req = request.Request(url, headers={"User-Agent": "AnimalVision/1.0"})
            with request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 429 or 500 <= exc.code < 600:
                retry_after = exc.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                time.sleep(min(delay, 30))
                continue
            raise RuntimeError(f"iNaturalist API HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"iNaturalist API request failed: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("iNaturalist API request failed after retries")


def download_bytes(url, attempts=3):
    for attempt in range(attempts):
        try:
            req = request.Request(url, headers={"User-Agent": "AnimalVision/1.0"})
            with request.urlopen(req, timeout=45) as response:
                data = response.read()
            if not data:
                raise ValueError("empty response")
            return data
        except (error.HTTPError, error.URLError, TimeoutError, ValueError):
            if attempt == attempts - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def valid_images(folder):
    images = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                if image.format == "GIF" or image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
                    continue
                image.verify()
            images.append(path)
        except (OSError, UnidentifiedImageError):
            continue
    return images


def image_digest(image):
    normalized = image.convert("RGB")
    return hashlib.sha256(
        f"{normalized.width}x{normalized.height}:".encode("ascii") + normalized.tobytes()
    ).hexdigest()


def existing_hashes(folder):
    hashes = set()
    for path in valid_images(folder):
        try:
            with Image.open(path) as image:
                hashes.add(image_digest(image))
        except (OSError, UnidentifiedImageError):
            pass
    return hashes


def resolve_taxon(class_name):
    data = http_json("taxa", {"q": TAXON_QUERIES[class_name], "per_page": 10})
    taxa = data.get("results", [])
    query = TAXON_QUERIES[class_name].lower()
    for taxon in taxa:
        if str(taxon.get("name", "")).lower() == query:
            return taxon["id"]
    if taxa and taxa[0].get("id"):
        return taxa[0]["id"]
    raise RuntimeError(f"Could not resolve iNaturalist taxon for {class_name}")


def photo_urls(observation):
    for photo in observation.get("photos", []):
        url = photo.get("url")
        if url:
            yield url.replace("/square.", "/original.").replace("/medium.", "/original.")


def progress(counts, active_class=None):
    print("\033[2J\033[HAnimalVision Dataset Downloader")
    for class_name in CLASS_NAMES:
        count = counts[class_name]
        width = 20
        filled = round(width * count / target_for(class_name))
        marker = "#" * min(filled, width) + "." * max(width - filled, 0)
        print(f"{class_name:<10} [{marker}] {count}/{target_for(class_name)}")
    if active_class:
        print(f"Downloading: {active_class}")


def download_class(class_name, counts):
    folder = DATASET_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)
    hashes = existing_hashes(folder)
    target = target_for(class_name)
    if counts[class_name] >= target:
        return None
    taxon_id = resolve_taxon(class_name)
    for page in range(1, 51):
        if counts[class_name] >= target:
            break
        try:
            data = http_json("observations", {
                "taxon_id": taxon_id, "photos": "true", "quality_grade": "research",
                "order": "votes", "order_by": "desc", "per_page": 100, "page": page,
            })
        except RuntimeError as exc:
            return str(exc)
        observations = data.get("results", [])
        if not observations:
            break
        for observation in observations:
            if counts[class_name] >= target:
                break
            for photo_number, url in enumerate(photo_urls(observation), start=1):
                raw = download_bytes(url)
                if not raw:
                    continue
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format == "GIF" or image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
                            continue
                        image.load()
                        digest = image_digest(image)
                        if digest in hashes:
                            continue
                        image = image.convert("RGB")
                        filename = safe_name(f"{observation.get('id', 'observation')}-{photo_number}.jpg")
                        image.save(folder / filename, "JPEG", quality=90, optimize=True)
                    hashes.add(digest)
                    counts[class_name] += 1
                    progress(counts, class_name)
                    break
                except (OSError, UnidentifiedImageError):
                    continue
            time.sleep(0.2)
        time.sleep(1)
    if counts[class_name] < target:
        return "not enough usable images found"
    return None


def verify(selected_classes):
    counts = {}
    for class_name in CLASS_NAMES:
        folder = DATASET_DIR / class_name
        folder.mkdir(parents=True, exist_ok=True)
        counts[class_name] = len(valid_images(folder))
    print("AnimalVision Dataset Verification\n")
    for class_name in selected_classes:
        print(f"{class_name:<10} {counts[class_name]}")
    print(f"\nTotal: {sum(counts.values())}")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Download AnimalVision images from iNaturalist.")
    parser.add_argument("--verify", action="store_true", help="scan without downloading")
    parser.add_argument("--resume", action="store_true", help="resume incomplete classes")
    parser.add_argument("--class", dest="class_name", choices=CLASS_NAMES, help="download one class")
    args = parser.parse_args()
    selected = [args.class_name] if args.class_name else CLASS_NAMES
    counts = verify(selected)
    if args.verify:
        return 0
    failures = {}
    for class_name in selected:
        try:
            reason = download_class(class_name, counts)
            if reason:
                failures[class_name] = reason
        except Exception as exc:
            failures[class_name] = str(exc)
            print(f"\n{class_name}: {exc}")
    counts = {name: len(valid_images(DATASET_DIR / name)) for name in CLASS_NAMES}
    print("\nDOWNLOAD COMPLETE\n")
    for class_name in selected:
        print(f"{class_name:<10} {counts[class_name]}")
    print(f"\nTotal: {sum(counts.values())}")
    incomplete = [name for name in selected if counts[name] < target_for(name)]
    if incomplete:
        print("\nWARNING: Dataset is incomplete.\n")
        for name in incomplete:
            print(f"{name}: {counts[name]}/{target_for(name)}")
            print(f"Reason: {failures.get(name, 'not enough usable images found')}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
