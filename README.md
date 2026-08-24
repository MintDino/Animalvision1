# AnimalVision

AnimalVision trains a small CNN to classify animal images from a local dataset and runs the model in the browser using TensorFlow.js.

## Dataset structure

The dataset must follow this folder layout:

```text
dataset/
  bear/
  cat/
  deer/
  dog/
  elephant/
  fox/
  giraffe/
  horse/
  lion/
  monkey/
  panda/
  rabbit/
  tiger/
  wolf/
  zebra/
```

Add images into each class folder. The script expects exactly the class names shown above, in that order.

## Download the dataset

The downloader uses the public iNaturalist API and saves validated JPEG images into the class folders. It keeps existing valid images and can resume safely:

```bash
python scripts/download_dataset.py
```

Optional modes:

```bash
python scripts/download_dataset.py --verify
python scripts/download_dataset.py --resume
python scripts/download_dataset.py --class cat
```

It skips corrupt, tiny, unsupported, duplicate, and GIF files.

## Train the model

```bash
python train.py
```

This script checks class counts, trains a CNN, saves the model to `models/animal_model.keras`, and writes `models/classes.json` and `models/training_history.json`.

Optional test prediction run:

```bash
python train.py --test test_images/
```

## Convert the model for the browser

```bash
python convert.py
```

This creates the TensorFlow.js model under `model/web_model/` and copies the classes file into that directory.

## Start the server

```bash
python app.py
```

Then open the app in the browser using your Acode/Termux browser or a local device browser at:

```text
http://localhost:5000/
```

## Configure OpenRouter

Set the environment variables before running the server:

```bash
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_MODEL="openai/gpt-4o-mini"
python app.py
```

If the key is missing, the UI shows a clear message instead of crashing.

## Add more training images

Place more labeled images in the relevant dataset directory and retrain.

## Retrain

```bash
python train.py
python convert.py
```

Then refresh the browser and use the app again.
