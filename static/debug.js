const output = document.getElementById("consoleOutput");
const status = document.getElementById("debugStatus");
const systemStatus = document.getElementById("systemStatus");

function writeLog(level, message) {
  const entry = document.createElement("div");
  entry.className = `console-entry ${level}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${level.toUpperCase()}  ${message}`;
  output.appendChild(entry);
  output.scrollTop = output.scrollHeight;
}

["log", "info", "warn", "error"].forEach((level) => {
  const original = console[level];
  console[level] = (...args) => {
    writeLog(level, args.map((value) => typeof value === "string" ? value : JSON.stringify(value)).join(" "));
    original.apply(console, args);
  };
});

function renderStatus(data) {
  const rows = [
    ["Server", data.server.status],
    ["Flask debug", data.server.debug ? "enabled" : "disabled"],
    ["Keras model", data.local_model.keras ? "ready" : "missing"],
    ["Web model", data.local_model.web_model_json && data.local_model.web_model_bins > 0 ? "ready" : "incomplete"],
    ["Model classes", data.local_model.classes ? "ready" : "missing"],
    ["OpenRouter", data.openrouter.configured ? "configured" : "not configured"],
    ["OpenRouter model", data.openrouter.model],
    ["Loaded versions", data.cache.loaded_versions.length ? data.cache.loaded_versions.join(", ") : "none"],
  ];
  systemStatus.innerHTML = rows.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
  status.textContent = "Online";
  status.className = "status-dot online";
}

async function refreshStatus() {
  status.textContent = "Checking";
  try {
    const response = await fetch("/api/debug");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderStatus(await response.json());
    writeLog("info", "Diagnostics refreshed.");
  } catch (error) {
    status.textContent = "Offline";
    status.className = "status-dot offline";
    writeLog("error", `Diagnostics request failed: ${error.message}`);
  }
}

document.getElementById("refreshDebug").addEventListener("click", refreshStatus);
document.getElementById("clearConsole").addEventListener("click", () => { output.innerHTML = ""; });
writeLog("info", "Developer console initialized.");
refreshStatus();
