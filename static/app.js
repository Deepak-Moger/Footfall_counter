const state = {
  metrics: null,
  lastOccupancy: 0,
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindControls();
  if (window.lucide) {
    window.lucide.createIcons();
  }
  resizeHeatCanvas();
  window.addEventListener("resize", resizeHeatCanvas);
  fetchMetrics();
  window.setInterval(fetchMetrics, 1400);
});

function cacheElements() {
  [
    "sourceSelect",
    "startBtn",
    "stopBtn",
    "resetBtn",
    "sourceName",
    "streamStatus",
    "confidenceValue",
    "trackingValue",
    "occupancyValue",
    "peakValue",
    "occupancyTrend",
    "entriesValue",
    "exitsValue",
    "conversionValue",
    "dwellValue",
    "forecastBars",
    "recommendationList",
    "zoneList",
    "eventList",
    "alertList",
    "heatCanvas",
    "videoFeed",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindControls() {
  els.startBtn.addEventListener("click", async () => {
    await postJson("/api/start", {
      source: els.sourceSelect.value,
      loop: true,
    });
    reloadVideo();
    fetchMetrics();
  });

  els.stopBtn.addEventListener("click", async () => {
    await postJson("/api/stop", {});
    fetchMetrics();
  });

  els.resetBtn.addEventListener("click", async () => {
    await postJson("/api/reset", {});
    fetchMetrics();
  });
}

async function fetchMetrics() {
  try {
    const response = await fetch("/api/metrics", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Metrics failed: ${response.status}`);
    }
    const metrics = await response.json();
    render(metrics);
  } catch (error) {
    renderError(error);
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function render(metrics) {
  const previous = state.metrics || metrics;
  state.metrics = metrics;

  setNumber(els.occupancyValue, metrics.occupancy);
  setNumber(els.entriesValue, metrics.entries);
  setNumber(els.exitsValue, metrics.exits);
  setNumber(els.peakValue, metrics.peakOccupancy);
  els.trackingValue.textContent = metrics.activeTracks;
  els.confidenceValue.textContent = `${Math.round((metrics.confidence || 0) * 100)}%`;
  els.conversionValue.textContent = `${Math.round((metrics.conversionRate || 0) * 100)}%`;
  els.dwellValue.textContent = formatDuration(metrics.averageDwellSeconds || 0);
  els.sourceName.textContent = titleCase(metrics.source || "Synthetic showroom");

  const delta = metrics.occupancy - previous.occupancy;
  els.occupancyTrend.textContent = delta > 0 ? "Rising" : delta < 0 ? "Easing" : "Stable";

  renderStreamStatus(metrics);
  renderForecast(metrics.forecast || []);
  renderRecommendations(metrics.aiRecommendations || []);
  renderZones(metrics.zones || []);
  renderEvents(metrics.events || []);
  renderAlerts(metrics.alerts || []);
  drawHeatmap(metrics.heatmap || []);
}

function renderStreamStatus(metrics) {
  const running = metrics.stream && metrics.stream.running;
  els.streamStatus.classList.toggle("stopped", !running && metrics.mode !== "demo");
  if (running) {
    els.streamStatus.innerHTML = '<span class="status-dot"></span>Live';
    return;
  }
  if (metrics.mode === "demo") {
    els.streamStatus.innerHTML = '<span class="status-dot"></span>Demo';
    return;
  }
  els.streamStatus.innerHTML = '<span class="status-dot"></span>Paused';
}

function renderForecast(items) {
  const max = Math.max(1, ...items.map((item) => Number(item.occupancy) || 0));
  els.forecastBars.innerHTML = items
    .map((item) => {
      const value = Number(item.occupancy) || 0;
      const width = Math.min(100, Math.max(4, (value / max) * 100));
      return `
        <div class="forecast-row">
          <span>${escapeHtml(item.label)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${value}</strong>
        </div>
      `;
    })
    .join("");
}

function renderRecommendations(items) {
  els.recommendationList.innerHTML = items
    .slice(0, 3)
    .map(
      (item, index) => `
        <div class="recommendation">
          <span>${index + 1}</span>
          <p>${escapeHtml(item)}</p>
        </div>
      `,
    )
    .join("");
}

function renderZones(items) {
  els.zoneList.innerHTML = items
    .map((zone) => {
      const pressure = Math.round((Number(zone.pressure) || 0) * 100);
      return `
        <div class="zone-item">
          <div class="zone-topline">
            <strong>${escapeHtml(zone.name)}</strong>
            <span>${zone.occupancy} people / ${pressure}%</span>
          </div>
          <div class="zone-meter"><span style="width:${Math.max(4, pressure)}%"></span></div>
        </div>
      `;
    })
    .join("");
}

function renderEvents(items) {
  const fallback = [{ time: "now", type: "Ready", message: "Waiting for the next crossing" }];
  els.eventList.innerHTML = (items.length ? items : fallback)
    .slice(0, 4)
    .map(
      (event) => `
        <div class="event">
          <div>
            <strong>${escapeHtml(event.type || "Event")}</strong>
            <span>${escapeHtml(event.message || "")}</span>
          </div>
          <time>${escapeHtml(event.time || "")}</time>
        </div>
      `,
    )
    .join("");
}

function renderAlerts(items) {
  els.alertList.innerHTML = items
    .slice(0, 3)
    .map(
      (alert) => `
        <div class="alert ${escapeHtml(alert.level || "info")}">
          <div>
            <strong>${escapeHtml(alert.title || "System")}</strong>
            <span>${escapeHtml(alert.message || "")}</span>
          </div>
        </div>
      `,
    )
    .join("");
}

function resizeHeatCanvas() {
  const rect = els.videoFeed.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.heatCanvas.width = Math.max(1, Math.round(rect.width * dpr));
  els.heatCanvas.height = Math.max(1, Math.round(rect.height * dpr));
  drawHeatmap(state.metrics ? state.metrics.heatmap || [] : []);
}

function drawHeatmap(points) {
  const canvas = els.heatCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!points.length) {
    return;
  }

  points.forEach((point) => {
    const x = point.x * width;
    const y = point.y * height;
    const radius = Math.max(28, Math.min(width, height) * 0.055) * (point.value || 0.7);
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, "rgba(0,229,255,0.34)");
    gradient.addColorStop(0.42, "rgba(53,242,163,0.16)");
    gradient.addColorStop(1, "rgba(255,93,115,0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

function renderError(error) {
  els.streamStatus.classList.add("stopped");
  els.streamStatus.innerHTML = '<span class="status-dot"></span>Error';
  console.error(error);
}

function reloadVideo() {
  const src = "/video?ts=" + Date.now();
  els.videoFeed.setAttribute("src", src);
}

function setNumber(element, value) {
  element.textContent = new Intl.NumberFormat("en").format(Number(value) || 0);
}

function formatDuration(seconds) {
  const value = Number(seconds) || 0;
  if (value < 60) {
    return `${Math.round(value)}s`;
  }
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

function titleCase(value) {
  return String(value)
    .replace(/[_-]/g, " ")
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
