let currentFile = null;
let lastData = null;
let selectedObjectIndex = null;
let camStream = null;
let _voiceRec = null;

function updateClock() {
  const n = new Date();
  const e = document.getElementById("clockTime");
  if (e) e.textContent = `${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
}
updateClock();
setInterval(updateClock, 10000);

function handleFileSelect(e) {
  const f = e.target.files[0];
  if (!f) return;
  currentFile = f;
  selectedObjectIndex = null;
  showPreview(URL.createObjectURL(f));
}

function handleViewfinderClick() {
  if (!currentFile && !camStream) document.getElementById("fileInput").click();
}

function showPreview(src) {
  const img = document.getElementById("previewImg");
  img.src = src;
  img.style.display = "block";
  document.getElementById("uploadHint").style.display = "none";
  document.getElementById("vfGuides").style.display = "block";
  stopCamera();
}

const vf = document.getElementById("viewfinder");
vf.addEventListener("dragover", (e) => {
  e.preventDefault();
  vf.style.opacity = ".85";
});
vf.addEventListener("dragleave", () => {
  vf.style.opacity = "1";
});
vf.addEventListener("drop", (e) => {
  e.preventDefault();
  vf.style.opacity = "1";
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) {
    currentFile = f;
    selectedObjectIndex = null;
    showPreview(URL.createObjectURL(f));
  }
});

async function toggleCamera() {
  if (camStream) {
    stopCamera();
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const v = document.getElementById("camVideo");
    v.srcObject = camStream;
    v.style.display = "block";
    document.getElementById("previewImg").style.display = "none";
    document.getElementById("uploadHint").style.display = "none";
    document.getElementById("vfGuides").style.display = "block";
  } catch (err) {
    toast("Camera access denied", "warn");
  }
}

function stopCamera() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  const v = document.getElementById("camVideo");
  if (v) v.style.display = "none";
}

function handleShutter() {
  if (camStream) capturePhoto();
  else document.getElementById("fileInput").click();
}

function capturePhoto() {
  const v = document.getElementById("camVideo");
  const c = document.getElementById("camCanvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(
    (b) => {
      currentFile = new File([b], "capture.jpg", { type: "image/jpeg" });
      selectedObjectIndex = null;
      showPreview(URL.createObjectURL(b));
    },
    "image/jpeg",
    0.95
  );
}

function setQuery(q) {
  document.getElementById("userQuery").value = q;
}

function toggleVoiceInput() {
  const btn = document.getElementById("micBtn");
  if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
    toast("Voice not supported", "warn");
    return;
  }
  if (_voiceRec) {
    _voiceRec.stop();
    _voiceRec = null;
    if (btn) btn.classList.remove("listening");
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  _voiceRec = new SR();
  _voiceRec.lang = "en-US";
  _voiceRec.interimResults = false;
  if (btn) btn.classList.add("listening");
  _voiceRec.onresult = (e) => {
    const t = e.results[0][0].transcript || "";
    document.getElementById("userQuery").value = t;
  };
  _voiceRec.onend = () => {
    _voiceRec = null;
    if (btn) btn.classList.remove("listening");
  };
  _voiceRec.onerror = () => {
    _voiceRec = null;
    if (btn) btn.classList.remove("listening");
    toast("Voice error", "warn");
  };
  _voiceRec.start();
}

async function triggerAnalyze() {
  if (!currentFile) {
    // Allow re-analyze from History / refreshed session using last image_path.
    try {
      const p = lastData && lastData.image_path ? String(lastData.image_path) : "";
      if (p) {
        const r = await fetch(p, { cache: "no-store" });
        if (r.ok) {
          const b = await r.blob();
          const ext = (b.type && b.type.includes("png")) ? "png" : "jpg";
          currentFile = new File([b], `history.${ext}`, { type: b.type || "image/jpeg" });
        }
      }
    } catch {
      // ignore
    }
    if (!currentFile) {
      toast("Upload or capture a photo first", "warn");
      return;
    }
  }

  const bar = document.getElementById("progressBar");
  bar.style.animation = "none";
  bar.offsetHeight;
  bar.style.animation = "";
  document.getElementById("screenAnalyzing").classList.remove("hidden");

  const steps = ["Detecting objects…", "Reading text…", "Reasoning…", "Finalizing…"];
  let si = 0;
  const st = setInterval(() => {
    si++;
    if (si < steps.length) {
      document.getElementById("analyzingStep").textContent = steps[si];
      ["dot1", "dot2", "dot3", "dot4"].forEach((id, i) => document.getElementById(id).classList.toggle("active", i === si));
    }
  }, 700);

  const queryInput = (document.getElementById("userQuery").value || "").trim();
  const fd = new FormData();
  fd.append("image", currentFile);

  let finalQuery = queryInput;
  if (selectedObjectIndex !== null && lastData && lastData.objects && lastData.objects[selectedObjectIndex]) {
    const obj = lastData.objects[selectedObjectIndex];
    if (queryInput) finalQuery = `For "${obj.label}": ${queryInput}`;
  }
  fd.append("query", finalQuery);

  try {
    const res = await fetch("/analyze", { method: "POST", body: fd });
    const data = await res.json();
    clearInterval(st);
    if (data.error) throw new Error(data.error);
    lastData = data;
    render(data);
  } catch (err) {
    clearInterval(st);
    document.getElementById("screenAnalyzing").classList.add("hidden");
    toast("❌ " + (err.message || "Something went wrong"), "error");
  }
}

function render(d) {
  document.getElementById("screenAnalyzing").classList.add("hidden");

  const objects = Array.isArray(d.objects) ? d.objects : [];
  const top = objects[0] || null;
  const ai = d.ai || {};
  const intent = (ai.type || "general object").toLowerCase();

  const titleEl = document.getElementById("sheetTitle");
  const confEl = document.getElementById("sheetConf");

  function pickHeaderObject(list) {
    if (!Array.isArray(list) || !list.length) return null;
    const sorted = [...list].sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0));
    const first = sorted[0] || null;
    const second = sorted[1] || null;
    if (!first) return null;
    // If "person" is top but a clear secondary object exists, prefer the object.
    if (String(first.label || "").toLowerCase() === "person" && second) {
      const c1 = Number(first.confidence || 0);
      const c2 = Number(second.confidence || 0);
      if (c2 >= c1 - 0.12) return second;
    }
    return first;
  }

  // Header: for Math, always show the solved result (not YOLO label confidence)
  if (intent === "math") {
    titleEl.textContent = (ai.summary && String(ai.summary).trim()) ? String(ai.summary).trim() : "Math";
    confEl.style.display = "none";
    confEl.textContent = "";
  } else {
    const headerObj = pickHeaderObject(objects);
    if (headerObj) {
      titleEl.textContent = headerObj.label || "Unknown";
      confEl.style.display = "";
      confEl.textContent = `${(Number(headerObj.confidence || 0) * 100).toFixed(0)}% confident`;
    } else if (intent === "document") {
      titleEl.textContent = "Document";
      confEl.style.display = "none";
      confEl.textContent = "";
    } else {
      titleEl.textContent = "Unknown";
      confEl.style.display = "none";
      confEl.textContent = "";
    }
  }

  drawBoxes(objects);

  const scroll = document.getElementById("sheetScroll");
  scroll.innerHTML = "";

  const badge = document.createElement("div");
  badge.className = "intent-badge " + (intent === "math" ? "intent-math" : intent === "product" ? "intent-product" : "intent-general");
  badge.textContent = intent === "math" ? "📐 Math" : intent === "product" ? "🛒 Product" : intent === "document" ? "📄 Document" : "🔍 Object";
  scroll.appendChild(badge);

  if (objects.length) {
    const body = `<div class="rcard-body">${objects
      .slice(0, 12)
      .map((o, i) => {
        const sel = i === selectedObjectIndex ? " (selected)" : "";
        return `<div style="padding:6px 0;border-bottom:1px solid rgba(230,169,109,.12)"><b>${o.label}</b> — ${(o.confidence * 100).toFixed(0)}%${sel}</div>`;
      })
      .join("")}</div>`;
    scroll.appendChild(makeCard("🎯", "Objects", body));
  }

  // ── Barcode / QR Code card (Google Lens-style) ──────────────────────────
  if (d.barcode && d.barcode.value) {
    const bc = d.barcode;
    const link = bc.link || {};
    const kindIcons = { url: "🔗", upi: "💳", email: "📧", phone: "📞", sms: "💬", wifi: "📶", geo: "📍", product: "🛒", text: "🔍" };
    const icon = kindIcons[link.kind] || "🔲";
    const typeBadge = `<span style="background:rgba(230,169,109,.15);color:#e6a96d;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;letter-spacing:.4px">${escapeHtml(bc.type)}</span>`;

    // Main value display
    let valueHtml = `<div style="font-family:monospace;font-size:13px;word-break:break-all;padding:8px 0;color:var(--text-primary)">${escapeHtml(bc.value)}</div>`;

    // Big open button — only if there's a URL to open
    let openBtn = "";
    if (link.url) {
      openBtn = `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer"
        style="display:flex;align-items:center;justify-content:center;gap:8px;
               margin:12px 0 4px;padding:13px 18px;border-radius:14px;
               background:linear-gradient(135deg,#e6a96d,#c97c3a);
               color:#fff;font-weight:700;font-size:15px;text-decoration:none;
               box-shadow:0 3px 12px rgba(230,169,109,.35);letter-spacing:.2px">
        ${icon} ${escapeHtml(link.label)}
      </a>`;
    }

    // WiFi special case — no URL, show instructions
    if (link.kind === "wifi") {
      const ssid = (bc.value.match(/S:([^;]+)/) || [])[1] || "";
      const pass = (bc.value.match(/P:([^;]+)/) || [])[1] || "";
      openBtn = `<div style="background:rgba(230,169,109,.1);border-radius:12px;padding:12px;margin:10px 0">
        <div style="font-size:13px;color:var(--text-secondary)">Network: <b style="color:var(--text-primary)">${escapeHtml(ssid)}</b></div>
        ${pass ? `<div style="font-size:13px;color:var(--text-secondary);margin-top:4px">Password: <b style="color:var(--text-primary)">${escapeHtml(pass)}</b></div>` : ""}
      </div>`;
    }

    // Display hint (domain / email / etc.)
    const displayHint = link.display && link.display !== bc.value
      ? `<div style="font-size:12px;color:var(--text-secondary);margin-top:2px">${escapeHtml(link.display)}</div>`
      : "";

    // Copy button
    const copyBtn = `<button type="button" class="chip chip-btn" style="margin-top:8px"
      onclick="copyText(${JSON.stringify(bc.value)})">Copy value</button>`;

    // Multiple codes
    let allCodesHtml = "";
    if (Array.isArray(bc.all) && bc.all.length > 1) {
      allCodesHtml = `<div class="rcard-body"><b>All detected (${bc.all.length})</b><div style="margin-top:6px">` +
        bc.all.map((item, i) => {
          const il = item.link || {};
          const ibtn = il.url
            ? `<a href="${escapeHtml(il.url)}" target="_blank" rel="noopener noreferrer" class="chip" style="text-decoration:none">${escapeHtml(il.label || "Open")}</a>`
            : "";
          return `<div style="padding:6px 0;border-bottom:1px solid rgba(230,169,109,.12)">
            <span style="font-size:12px;color:var(--text-secondary)">${escapeHtml(item.type)}</span>
            <div style="font-family:monospace;font-size:12px;word-break:break-all">${escapeHtml(item.value)}</div>
            ${ibtn}
          </div>`;
        }).join("") + "</div></div>";
    }

    scroll.appendChild(makeCard(
      "🔲",
      "Barcode / QR Code",
      `<div class="rcard-body">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">${typeBadge}</div>
        ${valueHtml}
        ${displayHint}
        ${openBtn}
        ${copyBtn}
      </div>${allCodesHtml}`
    ));
  }

  if (d.ocr_text) {
    scroll.appendChild(makeCard("📝", "OCR Text", `<div class="ocr-text">${escapeHtml(d.ocr_text)}</div>`));
  }
  if (d.translated_text) {
    scroll.appendChild(makeCard("🌍", "Translated (English)", `<div class="ocr-text">${escapeHtml(d.translated_text)}</div>`));
  }

  if (ai.summary || ai.details) {
    const actions = Array.isArray(ai.actions) ? ai.actions : [];
    const steps = ai.extra && Array.isArray(ai.extra.steps) ? ai.extra.steps : [];
    const products = ai.extra && Array.isArray(ai.extra.products) ? ai.extra.products : [];

    const selectedObj =
      selectedObjectIndex !== null && objects && objects[selectedObjectIndex] ? objects[selectedObjectIndex] : null;
    const baseLabel =
      intent === "math"
        ? ""
        : selectedObj && selectedObj.label
          ? String(selectedObj.label)
          : (Array.isArray(objects) && objects.length ? String(objects[0].label || "") : "");

    const wikiUrl = baseLabel ? `https://en.wikipedia.org/wiki/${encodeURIComponent(baseLabel.replaceAll(" ", "_"))}` : "";
    const ytUrl = baseLabel ? `https://www.youtube.com/results?search_query=${encodeURIComponent(baseLabel)}` : "";
    const googleShopUrl = baseLabel ? `https://www.google.com/search?tbm=shop&q=${encodeURIComponent(baseLabel)}` : "";
    const amazonUrl = baseLabel ? `https://www.amazon.in/s?k=${encodeURIComponent(baseLabel)}` : "";
    const flipkartUrl = baseLabel ? `https://www.flipkart.com/search?q=${encodeURIComponent(baseLabel)}` : "";

    const quickLinksHtml =
      baseLabel && intent !== "document"
        ? `<div class="rcard-body"><b>Quick links</b><div style="margin-top:6px">
            <a class="chip" target="_blank" href="${wikiUrl}">Wikipedia</a>
            <a class="chip" target="_blank" href="${ytUrl}">YouTube</a>
            <a class="chip" target="_blank" href="${googleShopUrl}">Shop</a>
          </div></div>`
        : "";

    const copyBar =
      `<div class="rcard-body"><b>Copy</b><div style="margin-top:6px">
        <button type="button" class="chip chip-btn" onclick="copyText(${JSON.stringify(String(ai.summary || ""))})">Copy summary</button>
        <button type="button" class="chip chip-btn" onclick="copyText(${JSON.stringify(String(ai.details || ""))})">Copy details</button>
        ${d.ocr_text ? `<button type="button" class="chip chip-btn" onclick="copyText(${JSON.stringify(String(d.ocr_text || ""))})">Copy OCR</button>` : ""}
      </div></div>`;

    // Make actions clickable: set query + run analyze again
    const actionsHtml = actions.length
      ? `<div class="rcard-body"><b>Actions</b><div style="margin-top:6px">${actions
          .map(
            (a) =>
              `<button type="button" class="chip chip-btn" data-action="${escapeHtml(String(a))}">${escapeHtml(a)}</button>`
          )
          .join(" ")}</div></div>`
      : "";
    const stepsHtml = steps.length ? `<div class="rcard-body"><b>Steps</b><div style="margin-top:6px;line-height:1.6">${steps.map((s) => `- ${escapeHtml(s)}`).join("<br>")}</div></div>` : "";
    // If backend didn't split products per object, render simple shopping cards per detected label.
    const labelList = Array.isArray(objects)
      ? [...new Set(objects.map((o) => String(o.label || "").trim()).filter(Boolean))].slice(0, 5)
      : [];

    const autoProductsHtml =
      labelList.length && intent !== "math" && intent !== "document"
        ? `<div class="rcard-body"><b>Shopping</b><div style="margin-top:6px;line-height:1.6">${labelList
            .map((name) => {
              const q = encodeURIComponent(name);
              return `<div style="padding:8px 0;border-bottom:1px solid rgba(230,169,109,.12)">
                <b>${escapeHtml(name)}</b>
                <div style="margin-top:6px">
                  <a class="chip" target="_blank" href="https://www.amazon.in/s?k=${q}">Amazon</a>
                  <a class="chip" target="_blank" href="https://www.flipkart.com/search?q=${q}">Flipkart</a>
                  <a class="chip" target="_blank" href="https://www.google.com/search?tbm=shop&q=${q}">Google</a>
                </div>
              </div>`;
            })
            .join("")}</div></div>`
        : "";

    const productsHtml = products.length
      ? `<div class="rcard-body"><b>Products</b><div style="margin-top:6px;line-height:1.6">${products
          .map((p) => {
            const links = Array.isArray(p.links) ? p.links : [];
            return `<div style="padding:8px 0;border-bottom:1px solid rgba(230,169,109,.12)"><b>${escapeHtml(p.name || "")}</b><div>${escapeHtml(
              p.price_range || ""
            )}</div><div style="margin-top:6px">${links.slice(0, 3).map((u) => `<a class="chip" target="_blank" href="${u}">Link</a>`).join(" ")}</div></div>`;
          })
          .join("")}</div></div>`
      : autoProductsHtml;

    scroll.appendChild(
      makeCard(
        "🤖",
        "AI Response",
        `<div class="rcard-body"><b>Summary</b><div style="margin-top:6px">${escapeHtml(ai.summary || "")}</div></div>
         <div class="rcard-body"><b>Details</b><div style="margin-top:6px;white-space:pre-wrap">${escapeHtml(ai.details || "")}</div></div>
         ${copyBar}${quickLinksHtml}${actionsHtml}${stepsHtml}${productsHtml}`
      )
    );
  }

  document.getElementById("resultsSheet").classList.add("open");
  document.getElementById("sheetActions").classList.remove("hidden");
}

function makeCard(ico, title, bodyHTML) {
  const d = document.createElement("div");
  d.className = "rcard";
  d.innerHTML = `<div class="rcard-hdr"><div class="rcard-ico">${ico}</div><span class="rcard-title">${title}</span></div>${bodyHTML}`;
  return d;
}

function drawBoxes(objects) {
  const img = document.getElementById("previewImg");
  const canvas = document.getElementById("detectionCanvas");
  if (!img || !canvas) return;

  if (!img.complete || !img.naturalWidth) {
    img.onload = () => drawBoxes(objects);
    return;
  }

  const vfr = document.getElementById("viewfinder").getBoundingClientRect();
  canvas.width = vfr.width;
  canvas.height = vfr.height;

  const rect = img.getBoundingClientRect();
  const sx = rect.width / img.naturalWidth;
  const sy = rect.height / img.naturalHeight;
  const ox = rect.left - vfr.left;
  const oy = rect.top - vfr.top;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const boxes = [];
  objects.forEach((o, i) => {
    if (!o.box) return;
    const [x1, y1, x2, y2] = o.box;
    const cx = ox + x1 * sx;
    const cy = oy + y1 * sy;
    const cw = (x2 - x1) * sx;
    const ch = (y2 - y1) * sy;
    boxes.push({ i, x: cx, y: cy, w: cw, h: ch, label: o.label, conf: o.confidence });
  });

  canvas._hitBoxes = boxes;

  boxes.forEach((b) => {
    const isSel = b.i === selectedObjectIndex;
    ctx.lineWidth = isSel ? 4 : 2.5;
    ctx.strokeStyle = isSel ? "#5CF0FC" : "#FF8A3D";
    ctx.fillStyle = isSel ? "rgba(92,240,252,0.10)" : "rgba(255,138,61,0.10)";
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    const tag = `${b.label} ${(b.conf * 100).toFixed(0)}%`;
    ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
    const tw = ctx.measureText(tag).width;
    const px = b.x;
    const py = b.y > 18 ? b.y - 18 : b.y + 6;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(px, py, tw + 10, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, px + 5, py + 12);
  });
}

document.getElementById("detectionCanvas").addEventListener("click", (e) => {
  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const boxes = canvas._hitBoxes || [];

  let hit = null;
  for (const b of boxes) {
    if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
      hit = b;
      break;
    }
  }
  if (!hit) return;

  selectedObjectIndex = hit.i === selectedObjectIndex ? null : hit.i;
  drawBoxes((lastData && lastData.objects) || []);
  toast(selectedObjectIndex === null ? "Selection cleared" : `Selected: ${hit.label}`);
});

function closeResults() {
  document.getElementById("resultsSheet").classList.remove("open");
  document.getElementById("sheetActions").classList.add("hidden");
}

function resetApp() {
  closeResults();
  stopCamera();
  currentFile = null;
  lastData = null;
  selectedObjectIndex = null;
  document.getElementById("previewImg").src = "";
  document.getElementById("previewImg").style.display = "none";
  document.getElementById("uploadHint").style.display = "";
  document.getElementById("vfGuides").style.display = "none";
  document.getElementById("fileInput").value = "";
  const ctx = document.getElementById("detectionCanvas").getContext("2d");
  ctx.clearRect(0, 0, 99999, 99999);
}

function toast(msg, type) {
  const t = document.createElement("div");
  t.className = "toast" + (type ? " " + type : "");
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 400);
  }, 2600);
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Expose functions used by HTML onclick handlers
window.handleFileSelect = handleFileSelect;
window.handleViewfinderClick = handleViewfinderClick;
window.toggleCamera = toggleCamera;
window.handleShutter = handleShutter;
window.triggerAnalyze = triggerAnalyze;
window.closeResults = closeResults;
window.resetApp = resetApp;
window.setQuery = setQuery;
window.toggleVoiceInput = toggleVoiceInput;

// Run an AI suggested action (chip) by re-analyzing.
window.runAction = function (actionText) {
  if (actionText) {
    document.getElementById("userQuery").value = String(actionText);
  }
  triggerAnalyze();
};

window.copyText = async function (text) {
  try {
    const t = String(text || "");
    if (!t.trim()) return toast("Nothing to copy", "warn");
    await navigator.clipboard.writeText(t);
    toast("Copied", "");
  } catch {
    toast("Copy failed", "warn");
  }
};

// Delegate Action chip clicks (more reliable than inline onclick).
document.addEventListener("click", (e) => {
  const btn = e.target.closest && e.target.closest("button[data-action]");
  if (!btn) return;
  const actionText = btn.getAttribute("data-action") || "";
  if (actionText) window.runAction(actionText);
});
