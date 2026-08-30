const output = document.getElementById("consoleOutput");
const status = document.getElementById("debugStatus");
const systemStatus = document.getElementById("systemStatus");
const routeList = document.getElementById("routeList");

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
    ["Updated at", data.server.timestamp || "unknown"],
    ["Python", data.server.python || "unknown"],
    ["Keras model", data.local_model.keras ? "ready" : "missing"],
    ["Web model", data.local_model.web_model_json && data.local_model.web_model_bins > 0 ? "ready" : "incomplete"],
    ["Model classes", data.local_model.classes ? "ready" : "missing"],
    ["Default model version", data.local_model.default_version || "current"],
    ["OpenRouter", data.openrouter.configured ? "configured" : "not configured"],
    ["OpenRouter model", data.openrouter.model],
    ["Loaded versions", data.cache.loaded_versions.length ? data.cache.loaded_versions.join(", ") : "none"],
  ];
  systemStatus.innerHTML = rows.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
  const routeRows = (data.routes || [])
    .map((route) => {
      if (typeof route === "string") {
        return `<li><code>GET</code><span>${route}</span></li>`;
      }
      const methods = (route.methods || []).join(", ");
      return `<li><code>${methods || "GET"}</code><span>${route.path || ""}</span></li>`;
    })
    .join("");
  routeList.innerHTML = routeRows || "<li>No routes reported.</li>";
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
