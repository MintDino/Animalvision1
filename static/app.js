const state = {
  selectedFile: null,
  selectedImage: null,
  model: null,
  classes: [],
  mode: document.body.dataset.mode || "local",
  modelVersion: "current",
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
  modelVersion: document.getElementById("modelVersion"),
  versionTrigger: document.querySelector(".version-trigger"),
  versionLabel: document.querySelector(".version-label"),
  versionOptions: document.querySelector(".version-options"),
};

function showMessage(message) {
  elements.messageBox.textContent = message;
  elements.messageBox.classList.remove("hidden");
}

function clearMessage() {
  elements.messageBox.textContent = "";
  elements.messageBox.classList.add("hidden");
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
  if (!state.selectedFile) {
    throw new Error("Please choose an image first.");
  }

  const formData = new FormData();
  formData.append("image", state.selectedFile);
  formData.append("version", state.modelVersion);
  const response = await fetch("/api/local-predict", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "The local model could not be loaded.");
  }
  return data.predictions;
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
        showMessage("Loading the local model...");
        const predictions = await loadLocalModel();
        showMessage("Running inference...");
        const [{ label: topLabel, confidence: topScore }] = predictions;
        elements.animalName.textContent = topLabel;
        elements.confidenceValue.textContent = `${topScore.toFixed(1)}%`;
        elements.confidenceBar.style.width = `${Math.min(topScore, 100)}%`;
        renderPredictions(predictions.map(({ label, confidence }) => [label, confidence]));
        clearMessage();
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
  elements.versionTrigger.addEventListener("click", () => {
    const isOpen = elements.versionTrigger.getAttribute("aria-expanded") === "true";
    elements.versionTrigger.setAttribute("aria-expanded", String(!isOpen));
    elements.versionOptions.hidden = isOpen;
    if (!isOpen) {
      elements.versionOptions.querySelector(".version-option")?.focus();
    }
  });

  elements.versionOptions.addEventListener("click", (event) => {
    const option = event.target.closest("[data-version]");
    if (!option) return;
    state.modelVersion = option.dataset.version;
    elements.versionLabel.textContent = option.textContent;
    elements.versionTrigger.setAttribute("aria-expanded", "false");
    elements.versionOptions.hidden = true;
    resetPreview();
    clearMessage();
  });

  elements.versionTrigger.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "Enter", " "].includes(event.key)) return;
    event.preventDefault();
    elements.versionTrigger.click();
  });

  elements.versionOptions.addEventListener("keydown", (event) => {
    const options = [...elements.versionOptions.querySelectorAll(".version-option")];
    const currentIndex = options.indexOf(document.activeElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const offset = event.key === "ArrowDown" ? 1 : -1;
      options[(currentIndex + offset + options.length) % options.length]?.focus();
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      document.activeElement?.click();
    } else if (event.key === "Escape") {
      event.preventDefault();
      elements.versionTrigger.setAttribute("aria-expanded", "false");
      elements.versionOptions.hidden = true;
      elements.versionTrigger.focus();
    }
  });

  document.addEventListener("click", (event) => {
    if (!elements.modelVersion.contains(event.target)) {
      elements.versionTrigger.setAttribute("aria-expanded", "false");
      elements.versionOptions.hidden = true;
    }
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

async function loadModelVersions() {
  try {
    const response = await fetch("/api/model-versions");
    if (!response.ok) return;
    const data = await response.json();
    elements.versionOptions.innerHTML = "";
    data.versions.forEach((version) => {
      const option = document.createElement("button");
      option.type = "button";
      option.setAttribute("role", "option");
      option.dataset.version = version;
      option.className = "version-option";
      option.textContent = version === "current" ? "Current model" : version;
      elements.versionOptions.appendChild(option);
    });
  } catch (error) {
    console.error("Could not load model versions", error);
  }
}

bindEvents();
loadModelVersions();
