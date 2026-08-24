const state = {
  selectedFile: null,
  selectedImage: null,
  model: null,
  classes: [],
  mode: "local",
};

const elements = {
  dropZone: document.getElementById("dropZone"),
  input: document.getElementById("imageInput"),
  previewPanel: document.getElementById("previewPanel"),
  previewImage: document.getElementById("previewImage"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  removeBtn: document.getElementById("removeBtn"),
  messageBox: document.getElementById("messageBox"),
  animalName: document.getElementById("animalName"),
  confidenceValue: document.getElementById("confidenceValue"),
  confidenceBar: document.getElementById("confidenceBar"),
  topPredictions: document.getElementById("topPredictions"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
};

function showMessage(message) {
  elements.messageBox.textContent = message;
  elements.messageBox.classList.remove("hidden");
}

function clearMessage() {
  elements.messageBox.textContent = "";
  elements.messageBox.classList.add("hidden");
}

function setMode(mode) {
  state.mode = mode;
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  if (mode === "openrouter") {
    clearMessage();
  }
}

function readImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showMessage("Invalid image file.");
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    const result = event.target.result;
    state.selectedFile = file;
    state.selectedImage = result;
    elements.previewImage.src = result;
    elements.previewPanel.classList.remove("hidden");
    clearMessage();
  };
  reader.readAsDataURL(file);
}

function resetPreview() {
  state.selectedFile = null;
  state.selectedImage = null;
  elements.previewImage.removeAttribute("src");
  elements.previewPanel.classList.add("hidden");
  elements.animalName.textContent = "—";
  elements.confidenceValue.textContent = "—";
  elements.confidenceBar.style.width = "0%";
  elements.topPredictions.innerHTML = "";
}

async function loadLocalModel() {
  if (state.model) {
    return state.model;
  }

  try {
    const model = await tf.loadLayersModel("/model/web_model/model.json");
    state.model = model;
    const response = await fetch("/model/web_model/classes.json");
    state.classes = await response.json();
    return model;
  } catch (error) {
    console.error(error);
    throw new Error("The local model could not be loaded.");
  }
}

function preprocessImage(imageElement) {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;

  const context = canvas.getContext("2d");
  context.drawImage(imageElement, 0, 0, 64, 64);

  const imageData = context.getImageData(0, 0, 64, 64);
  const pixels = new Float32Array(64 * 64 * 3);
  let index = 0;
  // MobileNetV2 preprocessing: normalize to [-1, 1]
  for (let i = 0; i < imageData.data.length; i += 4) {
    pixels[index] = (imageData.data[i] / 127.5) - 1.0;
    pixels[index + 1] = (imageData.data[i + 1] / 127.5) - 1.0;
    pixels[index + 2] = (imageData.data[i + 2] / 127.5) - 1.0;
    index += 3;
  }

  return tf.tensor(pixels, [1, 64, 64, 3], "float32");
}

function renderPredictions(predictions) {
  const top = [...predictions]
    .map(([label, score]) => ({ label, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  elements.topPredictions.innerHTML = "";
  top.forEach(({ label, score }) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <span class="name">${label}</span>
      <span class="confidence">${score.toFixed(1)}%</span>
    `;
    elements.topPredictions.appendChild(item);
  });
}

async function classifyLocalModel() {
  if (!state.selectedImage) {
    showMessage("Please choose an image first.");
    return;
  }

  try {
    const image = new Image();
    image.onload = async () => {
      const tensor = preprocessImage(image);
      const model = await loadLocalModel();
      const result = model.predict(tensor);
      const probabilities = await result.data();
      const predictions = probabilities
        .map((score, idx) => [state.classes[idx] || `class_${idx}`, score * 100])
        .sort((a, b) => b[1] - a[1]);

      const [topLabel, topScore] = predictions[0];
      elements.animalName.textContent = topLabel;
      elements.confidenceValue.textContent = `${topScore.toFixed(1)}%`;
      elements.confidenceBar.style.width = `${Math.min(topScore, 100)}%`;
      renderPredictions(predictions);
      tensor.dispose();
      result.dispose();
    };
    image.src = state.selectedImage;
  } catch (error) {
    console.error(error);
    showMessage("Something went wrong.\nThe local model could not be loaded.");
  }
}

async function classifyOpenRouter() {
  if (!state.selectedFile) {
    showMessage("Please choose an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", state.selectedFile);

  try {
    const response = await fetch("/api/openrouter", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      showMessage(data.error || "OpenRouter is not configured.");
      return;
    }

    elements.animalName.textContent = data.animal;
    elements.confidenceValue.textContent = `${Number(data.confidence || 0).toFixed(1)}%`;
    elements.confidenceBar.style.width = `${Math.min(Number(data.confidence || 0), 100)}%`;
    elements.topPredictions.innerHTML = `
      <li><span class="name">${data.animal}</span><span class="confidence">${Number(data.confidence || 0).toFixed(1)}%</span></li>
    `;
    clearMessage();
  } catch (error) {
    console.error(error);
    showMessage("Something went wrong.\nThe local model could not be loaded.");
  }
}

async function analyzeImage() {
  clearMessage();
  if (state.mode === "openrouter") {
    await classifyOpenRouter();
  } else {
    await classifyLocalModel();
  }
}

function bindEvents() {
  elements.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  elements.input.addEventListener("change", (event) => {
    const [file] = event.target.files;
    readImageFile(file);
  });

  elements.analyzeBtn.addEventListener("click", analyzeImage);
  elements.removeBtn.addEventListener("click", () => {
    elements.input.value = "";
    resetPreview();
  });

  elements.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragover");
  });

  elements.dropZone.addEventListener("dragleave", () => {
    elements.dropZone.classList.remove("dragover");
  });

  elements.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragover");
    const [file] = event.dataTransfer.files;
    readImageFile(file);
  });

  elements.dropZone.addEventListener("click", () => elements.input.click());
  elements.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.input.click();
    }
  });
}

bindEvents();
