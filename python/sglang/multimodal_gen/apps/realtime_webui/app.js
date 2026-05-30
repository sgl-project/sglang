const $ = (id) => document.getElementById(id);

const presets = [
  { name: "Reference Dolly", tone: "green", size: "832x480", fps: 16, prompt: "A smooth dolly-in from the reference image, natural light, steady camera, cinematic realism.", actions: [["w"], ["w"], ["w"], ["w"], ["w"], ["w"], [], []] },
  { name: "Look Around", tone: "blue", size: "832x480", fps: 16, prompt: "A controlled camera look-around of the same scene, consistent geometry, subtle parallax.", actions: [["j"], ["j"], [], ["l"], ["l"], [], ["i"], ["k"]] },
  { name: "Scene Scout", tone: "accent", size: "832x480", fps: 16, prompt: "A scouting shot through a quiet interior, responsive camera movement, stable subject detail.", actions: [["w"], ["w"], ["d"], ["d"], ["w", "l"], ["w"], ["a"], []] },
  { name: "Surveillance", tone: "green", size: "832x480", fps: 16, prompt: "A fixed-lens surveillance view of a room, fluorescent light, understated motion, documentary feel.", actions: [[], [], ["j"], ["j"], [], ["l"], ["l"], []] },
];

let ws = null;
let pendingHeader = null;
let queue = [];
let frames = 0;
let bytes = 0;
let lastFrameAt = 0;
let fpsSamples = [];

const canvas = $("viewport");
const ctx = canvas.getContext("2d", { alpha: false });

function setStatus(text, kind = "") {
  $("statusText").textContent = text;
  $("statusDot").className = "dot" + (kind ? ` ${kind}` : "");
}

function addHistory(text) {
  const item = document.createElement("span");
  item.textContent = text;
  $("historyList").prepend(item);
  while ($("historyList").children.length > 8) $("historyList").lastChild.remove();
}

function drawIdle() {
  const w = canvas.width, h = canvas.height;
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, "#171a16");
  g.addColorStop(0.58, "#253628");
  g.addColorStop(1, "#8f4a37");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "rgba(238,241,236,.82)";
  ctx.font = "600 46px Avenir Next, sans-serif";
  ctx.fillText("SGLD Realtime", 56, 86);
  for (let i = 0; i < 18; i++) {
    ctx.fillStyle = `rgba(238,241,236,${0.05 + i * 0.015})`;
    ctx.fillRect(56 + i * 64, h - 86 - i * 7, 42, 42 + i * 4);
  }
}

function updateStats(header) {
  $("queueText").textContent = `queue ${queue.length}`;
  $("frameText").textContent = `frames ${frames}`;
  $("byteText").textContent = `${(bytes / 1048576).toFixed(1)} MB`;
  if (header) $("chunkText").textContent = `chunk ${header.chunk_index}`;
}

function rgbToImageData(header, payload) {
  const width = Number(header.width), height = Number(header.height);
  const channels = Number(header.channels), count = Number(header.num_frames);
  const frameBytes = Number(header.bytes_per_frame);
  const src = new Uint8Array(payload);
  const items = [];
  for (let f = 0; f < count; f++) {
    const img = ctx.createImageData(width, height);
    let s = f * frameBytes, d = 0;
    for (let p = 0; p < width * height; p++) {
      img.data[d++] = src[s++];
      img.data[d++] = src[s++];
      img.data[d++] = src[s++];
      if (channels > 3) s += channels - 3;
      img.data[d++] = 255;
    }
    items.push({ image: img, chunk: header.chunk_index });
  }
  return items;
}

function renderLoop(now) {
  const targetMs = 1000 / Math.max(1, Number($("fps").value || 16));
  if (queue.length && now - lastFrameAt >= targetMs) {
    const item = queue.shift();
    if (canvas.width !== item.image.width || canvas.height !== item.image.height) {
      canvas.width = item.image.width;
      canvas.height = item.image.height;
    }
    ctx.putImageData(item.image, 0, 0);
    lastFrameAt = now;
    fpsSamples.push(now);
    fpsSamples = fpsSamples.filter((t) => now - t < 1000);
    $("renderFps").textContent = String(fpsSamples.length);
    $("chunkText").textContent = `chunk ${item.chunk}`;
    updateStats();
  }
  requestAnimationFrame(renderLoop);
}

async function readFirstFrame() {
  const file = $("firstFrame").files[0];
  return file ? new Uint8Array(await file.arrayBuffer()) : undefined;
}

function drawReferencePreview(file) {
  const preview = $("referencePreview");
  const previewCtx = preview.getContext("2d", { alpha: false });
  previewCtx.fillStyle = "#e5e7df";
  previewCtx.fillRect(0, 0, preview.width, preview.height);
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    const scale = Math.min(preview.width / img.width, preview.height / img.height);
    const w = img.width * scale, h = img.height * scale;
    previewCtx.fillRect(0, 0, preview.width, preview.height);
    previewCtx.drawImage(img, (preview.width - w) / 2, (preview.height - h) / 2, w, h);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}

async function connect() {
  if (ws) ws.close();
  const firstFrame = await readFirstFrame();
  const init = compact({
    type: "init",
    model: $("model").value,
    prompt: $("prompt").value,
    size: $("size").value,
    seconds: Number($("seconds").value),
    fps: Number($("fps").value),
    seed: Number($("seed").value),
    num_inference_steps: Number($("steps").value),
    guidance_scale: Number($("guidance").value),
    first_frame: firstFrame,
  });
  ws = new WebSocket($("serverUrl").value);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { ws.send(pack(init)); setStatus("Live", "live"); addHistory("session started"); };
  ws.onclose = () => { ws = null; setStatus("Closed"); addHistory("session closed"); };
  ws.onerror = () => { setStatus("Socket error", "error"); addHistory("socket error"); };
  ws.onmessage = (event) => receive(event.data);
}

function receive(data) {
  if (!pendingHeader) {
    pendingHeader = unpack(new Uint8Array(data));
    if (pendingHeader.type === "error") {
      setStatus(pendingHeader.content || "Server error", "error");
      addHistory(pendingHeader.content || "server error");
      pendingHeader = null;
    }
    return;
  }
  const now = performance.now();
  queue.push(...rgbToImageData(pendingHeader, data));
  frames += Number(pendingHeader.num_frames || 0);
  bytes += data.byteLength;
  $("latencyText").textContent = `${Math.round(performance.now() - now)} ms`;
  updateStats(pendingHeader);
  pendingHeader = null;
}

function sendEvent(kind, payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(pack({ type: "event", kind, payload }));
  addHistory(`${kind} event sent`);
}

function repeatActions(actions) {
  const n = Math.max(1, Number($("eventFrames").value || 8));
  return Array.from({ length: n }, (_, i) => actions[i % actions.length] || []);
}

function applyPreset(preset) {
  $("prompt").value = preset.prompt;
  $("size").value = preset.size;
  $("fps").value = preset.fps;
  sendEvent("prompt", preset.prompt);
  sendEvent("camera_actions", repeatActions(preset.actions));
  addHistory(`preset ${preset.name}`);
}

function enhancePrompt() {
  const suffix = " high-fidelity temporal consistency, stable camera geometry, natural motion, clean lighting.";
  if (!$("prompt").value.includes("temporal consistency")) {
    $("prompt").value = `${$("prompt").value.trim()},${suffix}`;
  }
}

function compact(obj) {
  return Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== undefined && v !== ""));
}

function renderPresets() {
  $("presetList").innerHTML = "";
  presets.forEach((preset) => {
    const btn = document.createElement("button");
    btn.className = "preset";
    btn.dataset.tone = preset.tone;
    btn.innerHTML = `<b>${preset.name}</b><span>${preset.size} · ${preset.fps}fps</span>`;
    btn.onclick = () => applyPreset(preset);
    $("presetList").appendChild(btn);
  });
}

function pack(value) {
  const out = [];
  const bytes = (arr) => out.push(...arr);
  const str = (s) => new TextEncoder().encode(s);
  const u16 = (n) => [(n >> 8) & 255, n & 255];
  const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
  const write = (v) => {
    if (v === null) return out.push(0xc0);
    if (typeof v === "boolean") return out.push(v ? 0xc3 : 0xc2);
    if (typeof v === "number") {
      if (Number.isInteger(v) && v >= 0 && v < 128) return out.push(v);
      if (Number.isInteger(v) && v < 0 && v >= -32) return out.push(0xe0 | (v + 32));
      if (Number.isInteger(v) && v >= 0 && v < 256) return bytes([0xcc, v]);
      if (Number.isInteger(v) && v >= 0 && v < 65536) return bytes([0xcd, ...u16(v)]);
      const b = new ArrayBuffer(9), view = new DataView(b);
      view.setUint8(0, 0xcb); view.setFloat64(1, v);
      return bytes(new Uint8Array(b));
    }
    if (typeof v === "string") {
      const b = str(v), n = b.length;
      if (n < 32) bytes([0xa0 | n]); else if (n < 256) bytes([0xd9, n]); else bytes([0xda, ...u16(n)]);
      return bytes(b);
    }
    if (v instanceof Uint8Array) {
      if (v.length < 256) bytes([0xc4, v.length]); else if (v.length < 65536) bytes([0xc5, ...u16(v.length)]); else bytes([0xc6, ...u32(v.length)]);
      return bytes(v);
    }
    if (Array.isArray(v)) {
      v.length < 16 ? bytes([0x90 | v.length]) : bytes([0xdc, ...u16(v.length)]);
      return v.forEach(write);
    }
    const entries = Object.entries(v);
    entries.length < 16 ? bytes([0x80 | entries.length]) : bytes([0xde, ...u16(entries.length)]);
    entries.forEach(([k, val]) => { write(k); write(val); });
  };
  write(value);
  return new Uint8Array(out);
}

function unpack(buf) {
  let i = 0;
  const text = new TextDecoder();
  const read = () => {
    const b = buf[i++];
    if (b <= 0x7f) return b;
    if ((b & 0xe0) === 0xa0) return readStr(b & 0x1f);
    if ((b & 0xf0) === 0x80) return readMap(b & 0x0f);
    if ((b & 0xf0) === 0x90) return Array.from({ length: b & 0x0f }, read);
    if (b === 0xc0) return null;
    if (b === 0xc2 || b === 0xc3) return b === 0xc3;
    if (b === 0xcc) return buf[i++];
    if (b === 0xcd) return (buf[i++] << 8) | buf[i++];
    if (b === 0xce) return (buf[i++] * 16777216) + (buf[i++] << 16) + (buf[i++] << 8) + buf[i++];
    if (b === 0xd9) return readStr(buf[i++]);
    if (b === 0xda) return readStr((buf[i++] << 8) | buf[i++]);
    if (b === 0xde) return readMap((buf[i++] << 8) | buf[i++]);
    throw new Error(`Unsupported msgpack byte ${b}`);
  };
  const readStr = (n) => text.decode(buf.slice(i, i += n));
  const readMap = (n) => {
    const obj = {};
    for (let j = 0; j < n; j++) obj[read()] = read();
    return obj;
  };
  return read();
}

renderPresets();
drawIdle();
requestAnimationFrame(renderLoop);
$("connectBtn").onclick = connect;
$("stopBtn").onclick = () => ws && ws.close();
$("sendPromptBtn").onclick = () => sendEvent("prompt", $("prompt").value);
$("enhanceBtn").onclick = enhancePrompt;
$("firstFrame").onchange = () => drawReferencePreview($("firstFrame").files[0]);
document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.onclick = () => sendEvent("camera_actions", repeatActions([[btn.dataset.action]]));
});
