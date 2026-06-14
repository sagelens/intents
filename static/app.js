const state = {
  workspace: null,
  selectedCluster: "all",
  transform: { scale: 1, x: 0, y: 0 },
  dragging: false,
  dragStart: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function clusterColor(clusterId) {
  const hue = (clusterId * 137.508) % 360;
  return `hsl(${hue.toFixed(1)} 68% 43%)`;
}

function clusterBackground(clusterId) {
  const hue = (clusterId * 137.508) % 360;
  return `hsl(${hue.toFixed(1)} 70% 96%)`;
}

function formatMs(value) {
  if (value == null) return "Not run";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 2600);
}

function setMessage(selector, message = "", type = "") {
  const element = $(selector);
  element.textContent = message;
  element.className = `message ${type}`.trim();
}

function parseEditorIntents() {
  return $("#intentInput").value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function updateEditorCount() {
  const count = parseEditorIntents().length;
  $("#editorCount").textContent = `${count} item${count === 1 ? "" : "s"}`;
}

function setActiveView(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  $$(".view").forEach((view) => view.classList.remove("active"));
  $(`#${name}View`).classList.add("active");
  if (name === "map") renderMap();
}

function updateWorkspace(payload) {
  state.workspace = payload;
  $("#intentCount").textContent = payload.intents.length;
  $("#clusterCount").textContent = payload.clusters.length;
  $("#buildTime").textContent = formatMs(payload.build_ms);
  $("#threshold").value = payload.threshold;
  $("#thresholdValue").value = Number(payload.threshold).toFixed(2);
  renderClusterControls();
  renderMap();
}

function renderClusterControls() {
  const filter = $("#clusterFilter");
  filter.innerHTML = '<option value="all">All clusters</option>';
  state.workspace.clusters.forEach((cluster) => {
    const option = document.createElement("option");
    option.value = String(cluster.id);
    option.textContent = `Cluster ${cluster.id} · ${cluster.center}`;
    filter.appendChild(option);
  });
  renderClusterList();
}

function renderClusterList() {
  const selected = String(state.selectedCluster);
  $("#clusterList").innerHTML = state.workspace.clusters.map((cluster) => {
    const color = clusterColor(cluster.id);
    const background = clusterBackground(cluster.id);
    const active = selected === String(cluster.id) ? "active" : "";
    return `
      <article class="cluster-card ${active}" data-cluster="${cluster.id}" style="--cluster-color:${color};--cluster-bg:${background}">
        <div class="cluster-card-head">
          <h3>Cluster ${cluster.id}</h3>
          <code>${cluster.size} intents</code>
        </div>
        <p>Center</p>
        <code>${escapeHtml(cluster.center)}</code>
        <p>${cluster.representatives.map(escapeHtml).join(" · ")}</p>
      </article>
    `;
  }).join("");

  $$(".cluster-card").forEach((card) => {
    card.addEventListener("click", () => {
      const clusterId = card.dataset.cluster;
      state.selectedCluster = state.selectedCluster === clusterId ? "all" : clusterId;
      $("#clusterFilter").value = state.selectedCluster;
      renderClusterList();
      renderMap();
    });
  });
}

function scalePoints(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return points.map((point) => ({
    ...point,
    sx: 70 + ((point.x - minX) / Math.max(maxX - minX, 1)) * 1060,
    sy: 60 + ((point.y - minY) / Math.max(maxY - minY, 1)) * 600,
  }));
}

function renderMap() {
  if (!state.workspace) return;
  const svg = $("#clusterMap");
  const search = $("#mapSearch").value.trim().toLowerCase();
  const points = scalePoints(state.workspace.points);
  const transform = state.transform;
  const grid = `
    <defs>
      <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
        <path d="M 60 0 L 0 0 0 60" fill="none" stroke="rgba(16,33,30,.055)" stroke-width="1"/>
      </pattern>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <rect width="1200" height="720" fill="url(#grid)"/>
  `;
  const nodes = points.map((point) => {
    const selected = state.selectedCluster === "all" || String(point.cluster_id) === String(state.selectedCluster);
    const matches = !search || point.intent.toLowerCase().includes(search);
    const opacity = selected && matches ? 1 : selected ? .16 : .07;
    const radius = point.is_center ? 11 : matches && search ? 8 : 5;
    const color = clusterColor(point.cluster_id);
    const label = point.is_center || (search && matches)
      ? `<text x="${point.sx + 13}" y="${point.sy + 4}" class="map-label">${escapeHtml(point.intent)}</text>`
      : "";
    return `
      <g class="map-node" data-intent="${escapeHtml(point.intent)}" data-cluster="${point.cluster_id}" opacity="${opacity}">
        <circle cx="${point.sx}" cy="${point.sy}" r="${radius + (point.is_center ? 7 : 0)}" fill="${color}" opacity=".12"/>
        <circle cx="${point.sx}" cy="${point.sy}" r="${radius}" fill="${color}" stroke="#fffdf8" stroke-width="${point.is_center ? 3 : 1.5}" ${point.is_center ? 'filter="url(#glow)"' : ""}/>
        ${label}
      </g>
    `;
  }).join("");
  svg.innerHTML = `${grid}<g id="mapViewport" transform="translate(${transform.x} ${transform.y}) scale(${transform.scale})">${nodes}</g>`;

  svg.querySelectorAll(".map-node").forEach((node) => {
    node.addEventListener("mouseenter", (event) => {
      const clusterId = Number(node.dataset.cluster);
      const cluster = state.workspace.clusters.find((item) => item.id === clusterId);
      const tooltip = $("#mapTooltip");
      tooltip.innerHTML = `<strong>${escapeHtml(node.dataset.intent)}</strong><span>Cluster ${clusterId} · center ${escapeHtml(cluster.center)}</span>`;
      tooltip.classList.remove("hidden");
      positionTooltip(event);
    });
    node.addEventListener("mousemove", positionTooltip);
    node.addEventListener("mouseleave", () => $("#mapTooltip").classList.add("hidden"));
    node.addEventListener("click", () => {
      state.selectedCluster = node.dataset.cluster;
      $("#clusterFilter").value = state.selectedCluster;
      renderClusterList();
      renderMap();
    });
  });
}

function positionTooltip(event) {
  const stage = $(".map-stage").getBoundingClientRect();
  const tooltip = $("#mapTooltip");
  tooltip.style.left = `${event.clientX - stage.left + 14}px`;
  tooltip.style.top = `${event.clientY - stage.top + 14}px`;
}

function rankingHtml(results) {
  const maxScore = Math.max(...results.map((item) => Math.max(item.score, 0)), .001);
  return `
    <div class="ranking-list">
      ${results.map((item, index) => {
        const score = item.score;
        const width = Math.max(2, (Math.max(score, 0) / maxScore) * 100);
        return `
          <div class="rank-row">
            <span class="rank-number">${String(index + 1).padStart(2, "0")}</span>
            <div class="rank-body">
              <div class="rank-label"><span>${escapeHtml(item.intent)}</span></div>
              <div class="score-bar"><span style="width:${width}%"></span></div>
            </div>
            <span class="score-value">${score.toFixed(3)}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderVectorResult(payload) {
  $("#vectorLatency").textContent = formatMs(payload.latency_ms);
  $("#vectorStage").className = "";
  $("#vectorStage").innerHTML = rankingHtml(payload.results);
}

async function loadState() {
  const payload = await api("/api/state");
  if (payload.ready) {
    $("#intentInput").value = payload.intents.join("\n");
    updateEditorCount();
    updateWorkspace(payload);
  }
}

async function buildWorkspace() {
  const button = $("#buildButton");
  const progress = $("#buildProgress");
  setMessage("#catalogMessage");
  button.disabled = true;
  progress.classList.remove("hidden");
  try {
    const payload = await api("/api/build", {
      method: "POST",
      body: JSON.stringify({
        intents: parseEditorIntents(),
        threshold: Number($("#threshold").value),
      }),
    });
    updateWorkspace(payload);
    setMessage("#catalogMessage", `Built ${payload.clusters.length} natural clusters from ${payload.intents.length} intents.`, "success");
    showToast("Semantic workspace rebuilt");
    setActiveView("map");
  } catch (error) {
    setMessage("#catalogMessage", error.message, "error");
  } finally {
    button.disabled = false;
    progress.classList.add("hidden");
  }
}

async function runQuery() {
  const query = $("#queryInput").value.trim();
  if (!query) {
    setMessage("#queryMessage", "Enter a query first.", "error");
    return;
  }
  const button = $("#runQuery");
  button.disabled = true;
  setMessage("#queryMessage", "Searching...");
  $("#vectorLatency").textContent = "Running";
  try {
    const result = await api("/api/query/vector", {
      method: "POST",
      body: JSON.stringify({
        query,
        limit: Number($("#vectorLimit").value),
      }),
    });
    renderVectorResult(result);
    setMessage("#queryMessage", "Search complete.", "success");
  } catch (error) {
    $("#vectorLatency").textContent = "Failed";
    $("#vectorStage").className = "empty-state";
    $("#vectorStage").textContent = error.message;
    setMessage("#queryMessage", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

$$(".tab").forEach((tab) => tab.addEventListener("click", () => setActiveView(tab.dataset.view)));
$("#intentInput").addEventListener("input", updateEditorCount);
$("#threshold").addEventListener("input", () => $("#thresholdValue").value = Number($("#threshold").value).toFixed(2));
$("#formatIntents").addEventListener("click", () => {
  const cleaned = [...new Set(parseEditorIntents().map((item) => item.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")))].filter(Boolean);
  $("#intentInput").value = cleaned.join("\n");
  updateEditorCount();
});
$("#clearIntents").addEventListener("click", () => {
  $("#intentInput").value = "";
  updateEditorCount();
});
$("#buildButton").addEventListener("click", buildWorkspace);
$("#runQuery").addEventListener("click", runQuery);
$("#queryInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") runQuery();
});
$("#mapSearch").addEventListener("input", renderMap);
$("#clusterFilter").addEventListener("change", (event) => {
  state.selectedCluster = event.target.value;
  renderClusterList();
  renderMap();
});
$("#resetMap").addEventListener("click", () => {
  state.selectedCluster = "all";
  state.transform = { scale: 1, x: 0, y: 0 };
  $("#clusterFilter").value = "all";
  $("#mapSearch").value = "";
  renderClusterList();
  renderMap();
});
$("#zoomIn").addEventListener("click", () => {
  state.transform.scale = Math.min(3, state.transform.scale * 1.2);
  renderMap();
});
$("#zoomOut").addEventListener("click", () => {
  state.transform.scale = Math.max(.6, state.transform.scale / 1.2);
  renderMap();
});

$("#clusterMap").addEventListener("wheel", (event) => {
  event.preventDefault();
  state.transform.scale = Math.max(.6, Math.min(3, state.transform.scale * (event.deltaY < 0 ? 1.08 : .92)));
  renderMap();
}, { passive: false });
$("#clusterMap").addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.dragStart = { x: event.clientX - state.transform.x, y: event.clientY - state.transform.y };
  event.currentTarget.setPointerCapture(event.pointerId);
});
$("#clusterMap").addEventListener("pointermove", (event) => {
  if (!state.dragging) return;
  state.transform.x = event.clientX - state.dragStart.x;
  state.transform.y = event.clientY - state.dragStart.y;
  renderMap();
});
$("#clusterMap").addEventListener("pointerup", () => state.dragging = false);

loadState().catch((error) => showToast(error.message));
