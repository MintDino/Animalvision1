const state = {
  selectedFile: null,
  selectedImage: null,
  model: null,
  classes: [],
  mode: "local",
  isAnalyzing: false,
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
    const modelResponse = await fetch("/model/web_model/model.json");
    if (!modelResponse.ok) {
      throw new Error(`Model manifest request failed (${modelResponse.status}).`);
    }

    const modelJson = await modelResponse.json();
    const modelTopology = normalizeTensorFlowJsTopology(modelJson.modelTopology);
    const weightSpecs = modelJson.weightsManifest.flatMap((group) => group.weights);
    const weightBuffers = await Promise.all(
      modelJson.weightsManifest[0].paths.map(async (path) => {
        const response = await fetch(`/model/web_model/${path}`);
        if (!response.ok) {
          throw new Error(`Model weights request failed (${response.status}).`);
        }
        return response.arrayBuffer();
      }),
    );
    const weightData = new Uint8Array(
      weightBuffers.reduce((total, buffer) => total + buffer.byteLength, 0),
    );
    let offset = 0;
    weightBuffers.forEach((buffer) => {
      weightData.set(new Uint8Array(buffer), offset);
      offset += buffer.byteLength;
    });

    const model = await Promise.race([
      tf.loadLayersModel({
        load: async () => ({
          modelTopology,
          weightSpecs,
          weightData,
        }),
      }),
      new Promise((_, reject) => {
        window.setTimeout(
          () => reject(new Error("The local model took too long to load.")),
          30000,
        );
      }),
    ]);
    state.model = model;
    const response = await fetch("/model/web_model/classes.json");
    state.classes = await response.json();
    return model;
  } catch (error) {
    console.error(error);
    throw new Error(`The local model could not be loaded: ${error.message}`);
  }
}

function normalizeTensorFlowJsTopology(value) {
  if (Array.isArray(value)) {
    return value.map(normalizeTensorFlowJsTopology);
  }
  if (!value || typeof value !== "object") {
    return value;
  }

  if (Array.isArray(value.args?.[0])) {
    return value.args[0].map((tensor) => {
      const history = tensor.config.keras_history;
      return [
        history[0],
        history[1],
        history[2],
        normalizeTensorFlowJsTopology(value.kwargs || {}),
      ];
    });
  }
  if (value.args?.[0]?.config?.keras_history) {
    const history = value.args[0].config.keras_history;
    return [
      history[0],
      history[1],
      history[2],
      normalizeTensorFlowJsTopology(value.kwargs || {}),
    ];
  }

  const normalized = {};
  Object.entries(value).forEach(([key, child]) => {
    if (key === "batch_shape") {
      normalized.batchInputShape = normalizeTensorFlowJsTopology(child);
    } else if (key === "inbound_nodes") {
      normalized.inboundNodes = normalizeTensorFlowJsTopology(child);
    } else if (key !== "optional") {
      normalized[key] = normalizeTensorFlowJsTopology(child);
    }
  });
  return normalized;
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
    showMessage("Loading model (first use only)...");
    elements.analyzeBtn.disabled = true;
    
    const image = new Image();
    image.onload = async () => {
      try {
        showMessage("Preprocessing image...");
        const tensor = preprocessImage(image);
        
        showMessage("Loading the local model...");
        const model = await loadLocalModel();
        showMessage("Running inference...");
        const result = model.predict(tensor);
        const probabilities = await Promise.race([
          result.data(),
          new Promise((_, reject) => {
            window.setTimeout(
              () => reject(new Error("Local inference took too long.")),
              30000,
            );
          }),
        ]);
        const predictions = probabilities
          .map((score, idx) => [state.classes[idx] || `class_${idx}`, score * 100])
          .sort((a, b) => b[1] - a[1]);

        const [topLabel, topScore] = predictions[0];
        elements.animalName.textContent = topLabel;
        elements.confidenceValue.textContent = `${topScore.toFixed(1)}%`;
        elements.confidenceBar.style.width = `${Math.min(topScore, 100)}%`;
        renderPredictions(predictions);
        clearMessage();
        tensor.dispose();
        result.dispose();
      } catch (error) {
        console.error(error);
        showMessage("Error during analysis: " + error.message);
      } finally {
        elements.analyzeBtn.disabled = false;
        state.isAnalyzing = false;
      }
    };
    image.src = state.selectedImage;
  } catch (error) {
    console.error(error);
    showMessage("Error: " + error.message);
    elements.analyzeBtn.disabled = false;
    state.isAnalyzing = false;
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
    showMessage("Sending to OpenRouter...");
    elements.analyzeBtn.disabled = true;
    
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
    showMessage("Error: " + error.message);
  } finally {
    elements.analyzeBtn.disabled = false;
    state.isAnalyzing = false;
  }
}

async function analyzeImage() {
  if (state.isAnalyzing) {
    showMessage("Analysis already in progress...");
    return;
  }
  
  state.isAnalyzing = true;
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
