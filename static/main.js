/* ═══════════════════════════════════════════════════════════════════
   SkyTrace — Flight Route Optimizer — main.js
   Handles: star field, airspace loading, canvas route drawing,
            route risk analysis, training charts
   ═══════════════════════════════════════════════════════════════════ */

// ── STAR FIELD ────────────────────────────────────────────────────────────────
(function buildStars() {
  const field = document.getElementById("starField");
  if (!field) return;
  for (let i = 0; i < 120; i++) {
    const star = document.createElement("div");
    star.className = "star";
    const size = Math.random() * 2.5 + 0.5;
    star.style.cssText = `
      width:${size}px; height:${size}px;
      top:${Math.random() * 100}%;
      left:${Math.random() * 100}%;
      --dur:${(Math.random() * 4 + 2).toFixed(1)}s;
      --delay:-${(Math.random() * 5).toFixed(1)}s;
    `;
    field.appendChild(star);
  }
})();

// ── NAVBAR SCROLL EFFECT ──────────────────────────────────────────────────────
window.addEventListener("scroll", () => {
  document.getElementById("navbar")?.classList.toggle("scrolled", window.scrollY > 40);
});

// ── SMOOTH SCROLL FOR HERO CTA ────────────────────────────────────────────────
document.getElementById("tryBtn")?.addEventListener("click", e => {
  e.preventDefault();
  document.getElementById("optimizer-section").scrollIntoView({ behavior: "smooth" });
});

// ── LOAD MODEL STATUS ─────────────────────────────────────────────────────────
async function loadStats() {
  const badge = document.getElementById("modelStatus");
  try {
    const res  = await fetch("/stats");
    const data = await res.json();
    badge.textContent = "⬤ Model Ready";
    badge.classList.remove("not-ready");

    const panel = document.getElementById("modelInfoPanel");
    if (panel) {
      panel.innerHTML = [
        `encoder    : ${data.encoder}`,
        `backbone   : ${data.architecture}`,
        `input      : ${data.in_channels}-channel temporal`,
        `best_dice  : ${data.best_dice.toFixed(4)}`,
        `epoch      : ${data.best_epoch}`,
        `threshold  : ${data.threshold}`,
        `device     : ${data.device}`,
      ].map(l => `<div>${l}</div>`).join("");
    }
  } catch {
    badge.textContent = "⬤ Model Offline";
    badge.classList.add("not-ready");
  }
}
loadStats();


// ══════════════════════════════════════════════════════════════════════════════
// ROUTE OPTIMIZER
// ══════════════════════════════════════════════════════════════════════════════

let probCanvas = null;     // hidden canvas holding the grayscale probability map
let probCtx    = null;     // 2d context for reading pixel values
let pointA     = null;     // departure point  { x, y (image 0-255), cx, cy (canvas px) }
let pointB     = null;     // arrival point
let canvasScale = 1;       // ratio: displayed canvas size / 256

// ── FILE UPLOAD & DRAG/DROP ZONE ─────────────────────────────────────────────
const uploadZone = document.getElementById("uploadZone");
const customFile = document.getElementById("customFile");

if (uploadZone && customFile) {
  // Click to browse
  uploadZone.addEventListener("click", (e) => {
    if (e.target !== customFile) {
      customFile.click();
    }
  });

  customFile.addEventListener("change", e => {
    if (e.target.files[0]) uploadCustom(e.target.files[0]);
  });

  // Drag and drop visual cues & drop handler
  uploadZone.addEventListener("dragover", e => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadCustom(file);
  });
}

// ── RESET & BACK BUTTONS ─────────────────────────────────────────────────────
document.getElementById("resetBtn")?.addEventListener("click", resetRoute);
document.getElementById("backBtn")?.addEventListener("click", goBack);


// ── SHOW / HIDE LOADING ──────────────────────────────────────────────────────
function showLoading() {
  document.getElementById("stepSelect").style.display  = "none";
  document.getElementById("scanLoading").style.display = "block";
}

// ── HIDE LOADING ─────────────────────────────────────────────────────────────
function hideLoading() {
  document.getElementById("scanLoading").style.display = "none";
}


// ── UPLOAD CUSTOM IMAGE ──────────────────────────────────────────────────────
async function uploadCustom(file) {
  const allowed = ["image/png", "image/jpeg", "image/jpg"];
  if (!allowed.includes(file.type)) {
    alert("Please upload a PNG or JPG image.");
    return;
  }
  showLoading();
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/predict", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    setupWorkspace(data);
  } catch (e) {
    alert(`Upload failed: ${e.message}`);
    goBack();
  }
}


// ── SETUP WORKSPACE ──────────────────────────────────────────────────────────
function setupWorkspace(data) {
  // Display the overlay image (satellite + heatmap + contours)
  const scanImg = document.getElementById("scanImage");
  scanImg.src = `data:image/png;base64,${data.overlay}`;

  // Load grayscale probability map into a hidden canvas
  // (used later for client-side route risk sampling — no server round-trip!)
  probCanvas = document.createElement("canvas");
  probCanvas.width  = 256;
  probCanvas.height = 256;
  probCtx = probCanvas.getContext("2d", { willReadFrequently: true });

  const pImg = new Image();
  pImg.onload = () => probCtx.drawImage(pImg, 0, 0, 256, 256);
  pImg.src = `data:image/png;base64,${data.probmap}`;

  // Populate scan metrics
  const m = data.metrics;
  animateCounter("sCoverage",    m.coverage_pct,          1, "%");
  animateCounter("sConfidence",  m.mean_confidence * 100, 1, "%");
  animateCounter("sTime",        m.processing_ms,         0, " ms");
  animateCounter("sPixels",      m.contrail_pixels,       0, " px");

  // Reset any previous route
  resetRoute();

  // Show workspace
  hideLoading();
  document.getElementById("stepDraw").style.display = "block";

  // Size the canvas once the image has rendered
  scanImg.onload = () => setupCanvas();
  if (scanImg.complete) setupCanvas();
}


// ── SETUP DRAWING CANVAS ─────────────────────────────────────────────────────
function setupCanvas() {
  const img    = document.getElementById("scanImage");
  const canvas = document.getElementById("drawCanvas");

  const w = img.clientWidth;
  const h = img.clientHeight;

  canvas.width  = w;
  canvas.height = h;
  canvasScale   = w / 256;

  // Attach click handler (remove old one first to avoid duplicates)
  canvas.removeEventListener("click", handleCanvasClick);
  canvas.addEventListener("click", handleCanvasClick);
}


// ── CANVAS CLICK HANDLER ─────────────────────────────────────────────────────
function handleCanvasClick(e) {
  if (pointA && pointB) return;   // both points already set

  const canvas = document.getElementById("drawCanvas");
  const rect   = canvas.getBoundingClientRect();
  const cx     = e.clientX - rect.left;
  const cy     = e.clientY - rect.top;

  // Map display coordinates to 256×256 probability map coordinates
  const ix = Math.min(Math.round(cx / canvasScale), 255);
  const iy = Math.min(Math.round(cy / canvasScale), 255);

  if (!pointA) {
    // Set departure
    pointA = { x: ix, y: iy, cx, cy };
    drawPoint(cx, cy, "#10b981", "A");
    document.getElementById("drawInstruction").innerHTML =
      'Click to set <strong>Arrival (B)</strong>';
  } else {
    // Set arrival → draw route → analyze
    pointB = { x: ix, y: iy, cx, cy };
    drawPoint(cx, cy, "#ef4444", "B");
    drawRouteLine();
    analyzeRoute();
  }
}


// ── DRAW A LABELED POINT ─────────────────────────────────────────────────────
function drawPoint(x, y, color, label) {
  const ctx = document.getElementById("drawCanvas").getContext("2d");

  // Outer glow
  ctx.beginPath();
  ctx.arc(x, y, 16, 0, Math.PI * 2);
  ctx.fillStyle = color + "30";
  ctx.fill();

  // Inner circle
  ctx.beginPath();
  ctx.arc(x, y, 9, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Label text
  ctx.fillStyle = "#fff";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y);
}


// ── DRAW ROUTE LINE + HOTSPOT MARKERS ────────────────────────────────────────
function drawRouteLine() {
  const ctx = document.getElementById("drawCanvas").getContext("2d");

  // Yellow dashed route line
  ctx.beginPath();
  ctx.setLineDash([8, 5]);
  ctx.moveTo(pointA.cx, pointA.cy);
  ctx.lineTo(pointB.cx, pointB.cy);
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.setLineDash([]);

  // Mark hotspot points along the route (red dots where contrail probability is high)
  if (!probCtx) return;
  const steps = 60;
  for (let i = 0; i <= steps; i++) {
    const t  = i / steps;
    const sx = Math.min(Math.round(pointA.x + (pointB.x - pointA.x) * t), 255);
    const sy = Math.min(Math.round(pointA.y + (pointB.y - pointA.y) * t), 255);
    const pixel = probCtx.getImageData(sx, sy, 1, 1).data;
    const prob  = pixel[0] / 255;

    if (prob > 0.35) {
      const dx = pointA.cx + (pointB.cx - pointA.cx) * t;
      const dy = pointA.cy + (pointB.cy - pointA.cy) * t;
      ctx.beginPath();
      ctx.arc(dx, dy, 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(239, 68, 68, ${Math.min(prob * 1.5, 0.9)})`;
      ctx.fill();
    }
  }
}


// ── ANALYZE ROUTE RISK ───────────────────────────────────────────────────────
function analyzeRoute() {
  if (!probCtx || !pointA || !pointB) return;

  const SAMPLES = 100;
  let totalRisk = 0, maxRisk = 0, hotspots = 0;

  for (let i = 0; i <= SAMPLES; i++) {
    const t = i / SAMPLES;
    const x = Math.min(Math.round(pointA.x + (pointB.x - pointA.x) * t), 255);
    const y = Math.min(Math.round(pointA.y + (pointB.y - pointA.y) * t), 255);

    const pixel = probCtx.getImageData(x, y, 1, 1).data;
    const prob  = pixel[0] / 255;

    totalRisk += prob;
    if (prob > maxRisk) maxRisk = prob;
    if (prob > 0.35) hotspots++;
  }

  const avgRisk    = totalRisk / (SAMPLES + 1);
  const hotspotPct = Math.round((hotspots / (SAMPLES + 1)) * 100);

  // ── Populate stats
  document.getElementById("rRisk").textContent     = `${(avgRisk * 100).toFixed(1)}%`;
  document.getElementById("rPeak").textContent      = `${(maxRisk * 100).toFixed(1)}%`;
  document.getElementById("rHotspots").textContent  = `${hotspotPct}%`;
  document.getElementById("rSamples").textContent   = `${SAMPLES + 1}`;

  // Color-code the risk score
  const riskEl = document.getElementById("rRisk");
  if (avgRisk > 0.3)       riskEl.style.color = "var(--red)";
  else if (avgRisk > 0.15) riskEl.style.color = "var(--amber)";
  else                     riskEl.style.color = "var(--green)";

  // ── Decision banner
  const banner = document.getElementById("routeBanner");
  const icon   = document.getElementById("rBannerIcon");
  const title  = document.getElementById("rBannerTitle");
  const text   = document.getElementById("rBannerText");

  if (avgRisk > 0.25 || hotspotPct > 30) {
    banner.className = "route-banner banner-danger";
    icon.textContent = "🚨";
    title.textContent = "HIGH RISK — Reroute Recommended";
    text.textContent =
      `Route intersects ${hotspotPct}% contrail-forming zones (peak: ${(maxRisk * 100).toFixed(0)}%). ` +
      `Recommend altitude change of ±2,000 ft or lateral deviation to avoid persistent contrail formation. ` +
      `Contrails at this altitude will significantly increase radiative forcing.`;
  } else if (avgRisk > 0.1 || hotspotPct > 10) {
    banner.className = "route-banner banner-warning";
    icon.textContent = "⚠️";
    title.textContent = "MODERATE RISK — Monitor Conditions";
    text.textContent =
      `${hotspotPct}% of route shows potential contrail formation. ` +
      `Consider minor altitude adjustment (±1,000 ft). ` +
      `Monitor atmospheric humidity and temperature en route.`;
  } else {
    banner.className = "route-banner banner-safe";
    icon.textContent = "✅";
    title.textContent = "OPTIMAL ROUTE — Cleared for Flight";
    text.textContent =
      `Only ${hotspotPct}% of route shows contrail probability above threshold. ` +
      `Current altitude and heading are optimal for minimal climate impact. ` +
      `Estimated contrail warming contribution: MINIMAL.`;
  }

  // Show the analysis panel
  document.getElementById("routeAnalysis").style.display = "block";
  document.getElementById("routeAnalysis").scrollIntoView({ behavior: "smooth", block: "nearest" });
  document.getElementById("drawInstruction").innerHTML =
    'Route analyzed! Click <strong>↻ Reset Route</strong> to try another.';
}


// ── RESET ROUTE (keep airspace, clear drawing) ───────────────────────────────
function resetRoute() {
  pointA = null;
  pointB = null;

  const canvas = document.getElementById("drawCanvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  document.getElementById("routeAnalysis").style.display = "none";
  document.getElementById("drawInstruction").innerHTML =
    'Click on the scan to set <strong>Departure (A)</strong>';
}


// ── GO BACK TO AIRSPACE SELECTION ────────────────────────────────────────────
function goBack() {
  document.getElementById("stepDraw").style.display    = "none";
  document.getElementById("scanLoading").style.display = "none";
  document.getElementById("stepSelect").style.display  = "block";
  resetRoute();
}


// ── METRIC COUNTER ANIMATION ──────────────────────────────────────────────────
function animateCounter(elId, target, decimals, suffix) {
  const el = document.getElementById(elId);
  if (!el) return;
  const dur = 900;
  let t0 = null;

  function step(ts) {
    if (!t0) t0 = ts;
    const progress = Math.min((ts - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - progress, 3);    // ease-out-cubic
    el.textContent = (target * ease).toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}


// ── TRAINING HISTORY CHARTS ───────────────────────────────────────────────────
async function renderCharts() {
  try {
    const res  = await fetch("/history");
    const data = await res.json();
    const hist = data.history;
    const epochs = hist.train_loss.map((_, i) => `${i + 1}`);

    const chartDefaults = {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#8892b0", font: { family: "Inter", size: 12 } } },
        tooltip: {
          backgroundColor: "#0f1325",
          borderColor: "rgba(100,130,255,0.2)",
          borderWidth: 1,
          titleColor: "#f0f4ff",
          bodyColor: "#8892b0",
          padding: 12,
        }
      },
      scales: {
        x: {
          ticks: { color: "#4a5280", font: { size: 11 } },
          grid:  { color: "rgba(100,130,255,0.06)" },
        },
        y: {
          ticks: { color: "#4a5280", font: { size: 11 } },
          grid:  { color: "rgba(100,130,255,0.06)" },
        }
      }
    };

    // Loss chart
    new Chart(document.getElementById("lossChart"), {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Train Loss",
            data: hist.train_loss,
            borderColor: "#00e5ff",
            backgroundColor: "rgba(0,229,255,0.08)",
            borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true,
          },
          {
            label: "Val Loss",
            data: hist.val_loss,
            borderColor: "#7c3aed",
            backgroundColor: "rgba(124,58,237,0.08)",
            borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true,
          },
        ]
      },
      options: chartDefaults,
    });

    // Dice chart
    const bestDice = data.best_dice;
    new Chart(document.getElementById("diceChart"), {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Train Dice",
            data: hist.train_dice,
            borderColor: "#00e5ff",
            backgroundColor: "rgba(0,229,255,0.08)",
            borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true,
          },
          {
            label: "Val Dice",
            data: hist.val_dice,
            borderColor: "#10b981",
            backgroundColor: "rgba(16,185,129,0.08)",
            borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true,
          },
          {
            label: `Best (${bestDice.toFixed(4)})`,
            data: epochs.map(() => bestDice),
            borderColor: "#f59e0b",
            borderWidth: 1.5, borderDash: [6, 4],
            pointRadius: 0, fill: false,
          },
        ]
      },
      options: chartDefaults,
    });
  } catch (err) {
    console.warn("Could not load training history:", err.message);
  }
}
renderCharts();
