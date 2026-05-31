const $ = (id) => document.getElementById(id);
const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip";
const RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";
const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v3";
const PREVIEW_OUTPUT_FORMAT = "raw";
const PREVIEW_OUTPUT_QUALITY = null;
const RECONNECT_CLOSE_TIMEOUT_MS = 15000;
const LIVE_QUEUE_SECONDS = 0.45;
const LOW_LATENCY_FPS_FLOOR = 10;
const LOW_LATENCY_QUEUE_SECONDS = 0.35;
const MAX_CATCHUP_FPS = 30;
const EVENT_QUEUE_SECONDS = 0.25;
const KEYBOARD_EVENT_INTERVAL_MS = 240;
const CONTROL_KEY_ACTIONS = new Map([
  ["w", "w"],
  ["a", "a"],
  ["s", "s"],
  ["d", "d"],
  ["arrowup", "i"],
  ["arrowleft", "j"],
  ["arrowdown", "k"],
  ["arrowright", "l"],
]);
const CONTROL_ACTION_META = {
  w: {
    label: "Forward",
    type: "translation",
    axis: "+forward",
    amount: "0.05/frame",
  },
  a: { label: "Left", type: "translation", axis: "-right", amount: "0.05/frame" },
  s: {
    label: "Back",
    type: "translation",
    axis: "-forward",
    amount: "0.05/frame",
  },
  d: { label: "Right", type: "translation", axis: "+right", amount: "0.05/frame" },
  i: { label: "Pitch +", type: "rotation", axis: "+pitch", amount: "4deg/frame" },
  j: { label: "Yaw -", type: "rotation", axis: "-yaw", amount: "6deg/frame" },
  k: { label: "Pitch -", type: "rotation", axis: "-pitch", amount: "4deg/frame" },
  l: { label: "Yaw +", type: "rotation", axis: "+yaw", amount: "6deg/frame" },
};

const presets = [
  { name: "Dragon Dolly", tone: "green", size: "832x480", fps: 16, prompt: "A smooth first-person dolly toward the castle, natural parallax, stable fantasy scene detail.", actions: [["w"], ["w"], ["w"], ["w"], ["w"], ["w"], [], []], referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/00/image.jpg", source: "LingBot example 00" },
  { name: "Stone Orbit", tone: "blue", size: "832x480", fps: 16, prompt: "A controlled look-around of the stone monument, overcast daylight, consistent geometry, subtle camera arc.", actions: [["j"], ["j"], [], ["l"], ["l"], [], ["i"], ["k"]], referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/01/image.jpg", source: "LingBot example 01" },
  { name: "Urban Tilt", tone: "accent", size: "832x480", fps: 16, prompt: "A cinematic urban wall shot with a slow tilt and slight forward movement, warm backlight, stable architecture.", actions: [["i"], ["i"], ["w"], ["w"], ["d"], [], ["j"], []], referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/02/image.jpg", source: "LingBot example 02" },
  { name: "Lake Scout", tone: "green", size: "832x480", fps: 16, prompt: "A calm scouting shot across the lake, gentle camera drift, crisp mountains, stable reflections.", actions: [["w"], ["w"], ["d"], ["d"], ["w", "l"], ["w"], ["a"], []], referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/03/image.jpg", source: "LingBot example 03" },
  { name: "Plastic Beach", tone: "blue", size: "832x480", fps: 16, prompt: "A slow aerial orbit around a pastel floating island hotel in the open ocean, hazy sunlight, turquoise water, toy-like architectural detail, clean horizon, cinematic but playful.", actions: [["w"], ["d"], ["d"], ["l"], ["l"], [], ["i"], []], referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/b8/f9/b9/b8f9b9f8-a609-bde2-0302-349436ffc508/825646291038.jpg/600x600bb.jpg", source: "Gorillaz Plastic Beach artwork", mime: "image/jpeg" },
  { name: "Plastic Ono Band", tone: "green", size: "832x480", fps: 16, prompt: "A quiet sunlit park under a massive tree, a solitary figure resting in the grass, soft summer haze, restrained documentary camera, intimate and naturalistic.", actions: [[], ["w"], ["w"], ["j"], [], ["l"], [], []], referenceUrl: "https://upload.wikimedia.org/wikipedia/en/a/a4/JLPOBCover.jpg", source: "John Lennon/Plastic Ono Band artwork", mime: "image/jpeg" },
  { name: "Kid A", tone: "accent", size: "832x480", fps: 16, prompt: "A cold surreal mountain range with sharp icy peaks, black-red storm clouds, glacial light, slow lateral pan, abstract digital texture, uneasy atmospheric scale.", actions: [["d"], ["d"], ["l"], [], ["w"], ["w"], ["j"], []], referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music122/v4/bd/8e/13/bd8e1358-b367-a689-cb84-cebd0b067dc4/634904078263.png/600x600bb.jpg", source: "Radiohead Kid A artwork", mime: "image/jpeg" },
];

let ws = null;
const referenceCache = new Map();
let selectedPreset = null;
let selectedReferenceBytes = null;
let selectedReferenceLabel = "";
let pendingHeader = null;
let queue = [];
let frames = 0;
let bytes = 0;
let lastFrameAt = 0;
let chunkWaitStartedAt = 0;
let clearQueueOnClose = false;
let fpsSamples = [];
let playbackFps = 0;
let droppedFrames = 0;
let receiveChain = Promise.resolve();
let nextEventId = 1;
let awaitedEventId = 0;
let awaitedEventSentAt = 0;
let chunkReceiveStartedAt = 0;
let currentReceiveChunk = null;
let currentReceiveChunkFrames = 0;
let lastRawRgbFrame = null;
let decoderWorker = null;
let decodeWorkerUnavailable = false;
let decodeRequestId = 1;
let streamEpoch = 0;
let lastDecodeMs = 0;
let lastDisplayLagMs = 0;
let lastControlKeySentAt = 0;
let socketHadError = false;
let socketCloseExpected = false;
let socketServerError = "";
const decodeRequests = new Map();
const activeControlActions = new Set();

const canvas = $("viewport");
const ctx = canvas.getContext("2d", { alpha: false });
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d", { alpha: false });

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

function resetStreamStats() {
  pendingHeader = null;
  queue = [];
  frames = 0;
  bytes = 0;
  fpsSamples = [];
  chunkWaitStartedAt = 0;
  clearQueueOnClose = false;
  playbackFps = Number($("fps").value || 16);
  droppedFrames = 0;
  receiveChain = Promise.resolve();
  awaitedEventId = 0;
  awaitedEventSentAt = 0;
  chunkReceiveStartedAt = 0;
  currentReceiveChunk = null;
  currentReceiveChunkFrames = 0;
  activeControlActions.clear();
  lastControlKeySentAt = 0;
  resetDecoderState();
  updateStats();
  $("renderFps").textContent = "0";
  $("stageRenderFps").textContent = "0";
  $("latencyText").textContent = "-";
  $("stageLatencyText").textContent = "-";
  $("decodeText").textContent = "-";
  $("displayLagText").textContent = "-";
  $("chunkText").textContent = "chunk -";
}

function rejectPendingDecodes(message) {
  for (const request of decodeRequests.values()) {
    request.reject(new Error(message));
  }
  decodeRequests.clear();
}

function ensureDecoderWorker() {
  if (decoderWorker || decodeWorkerUnavailable) return;
  if (typeof Worker === "undefined") {
    decodeWorkerUnavailable = true;
    return;
  }

  decoderWorker = new Worker(DECODER_WORKER_URL);
  decoderWorker.onmessage = (event) => {
    const message = event.data;
    const request = decodeRequests.get(message.id);
    if (!request) return;
    decodeRequests.delete(message.id);
    if (message.type === "error") {
      request.reject(new Error(message.message || "decode failed"));
      return;
    }
    request.resolve(message);
  };
  decoderWorker.onerror = (event) => {
    decodeWorkerUnavailable = true;
    decoderWorker?.terminate();
    decoderWorker = null;
    rejectPendingDecodes(event.message || "decode worker failed");
  };
}

function resetDecoderState() {
  lastRawRgbFrame = null;
  if (decoderWorker) decoderWorker.postMessage({ type: "reset" });
}

async function decodeFrameBatch(header, data) {
  const decodeStartedAt = performance.now();
  if (!isWorkerDecodableRawContentType(header.content_type)) {
    const items = await framePayloadToImageData(header, data);
    const decodedAt = performance.now();
    lastDecodeMs = decodedAt - decodeStartedAt;
    return items.map((item) => ({
      ...item,
      receivedAt: header.__received_at,
      decodedAt,
      decodeMs: lastDecodeMs,
    }));
  }

  ensureDecoderWorker();
  if (!decoderWorker || decodeWorkerUnavailable) {
    const items = await framePayloadToImageData(header, data);
    const decodedAt = performance.now();
    lastDecodeMs = decodedAt - decodeStartedAt;
    return items.map((item) => ({
      ...item,
      receivedAt: header.__received_at,
      decodedAt,
      decodeMs: lastDecodeMs,
    }));
  }

  const payload = data instanceof ArrayBuffer ? data : await data.arrayBuffer();
  const id = decodeRequestId++;
  const decodeHeader = { ...header, __decode_id: id };
  return new Promise((resolve, reject) => {
    decodeRequests.set(id, {
      resolve: (message) => {
        const decodedAt = performance.now();
        lastDecodeMs = decodedAt - decodeStartedAt;
        resolve(message.frames.map((buffer) => ({
          image: new ImageData(new Uint8ClampedArray(buffer), message.width, message.height),
          chunk: message.chunk,
          receivedAt: header.__received_at,
          decodedAt,
          decodeMs: lastDecodeMs,
        })));
      },
      reject,
    });
    try {
      decoderWorker.postMessage(
        { type: "decode", header: decodeHeader, payload },
        [payload],
      );
    } catch (error) {
      decodeRequests.delete(id);
      reject(error);
    }
  });
}

function isWorkerDecodableRawContentType(contentType) {
  return (
    contentType === RAW_RGB_CONTENT_TYPE ||
    contentType === RAW_RGB_DELTA_GZIP_CONTENT_TYPE ||
    contentType === RAW_RGBA_DELTA_GZIP_CONTENT_TYPE
  );
}

function updateStats(header) {
  $("queueText").textContent = droppedFrames
    ? `queue ${queue.length} · drop ${droppedFrames}`
    : `queue ${queue.length}`;
  $("frameText").textContent = `frames ${frames}`;
  $("byteText").textContent = `${(bytes / 1048576).toFixed(1)} MB`;
}

function trimLiveQueue(latestFrameCount) {
  const targetFps = Number($("fps").value || 16);
  const maxQueue = Math.max(
    Number(latestFrameCount || 0),
    Math.round(targetFps * LIVE_QUEUE_SECONDS),
  );
  if (queue.length <= maxQueue) return;
  const dropCount = queue.length - maxQueue;
  queue.splice(0, dropCount);
  droppedFrames += dropCount;
}

function trimQueueForPendingEvent() {
  const targetFps = playbackFps || Number($("fps").value || 16);
  const keep = Math.max(1, Math.round(targetFps * EVENT_QUEUE_SECONDS));
  if (queue.length <= keep) return;
  const dropCount = queue.length - keep;
  queue.splice(0, dropCount);
  droppedFrames += dropCount;
}

function rgbToImageData(header, payload) {
  const width = Number(header.width), height = Number(header.height);
  const channels = Number(header.channels), count = Number(header.num_frames);
  const frameBytes = Number(header.bytes_per_frame);
  const src = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
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

function rgbaToImageData(header, payload) {
  const width = Number(header.width), height = Number(header.height);
  const count = Number(header.num_frames);
  const frameBytes = Number(header.bytes_per_frame);
  const src = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const items = [];
  for (let f = 0; f < count; f++) {
    const offset = f * frameBytes;
    const imageBytes = new Uint8ClampedArray(
      src.buffer,
      src.byteOffset + offset,
      frameBytes,
    );
    items.push({ image: new ImageData(imageBytes, width, height), chunk: header.chunk_index });
  }
  return items;
}

async function gunzipBytes(payload) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support gzip stream decoding");
  }
  const stream = new Blob([payload]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function restoreDeltaGzipRawRgb(header, payload) {
  const frameBytes = Number(header.bytes_per_frame);
  const count = Number(header.num_frames);
  const expectedSize = frameBytes * count;
  const restored = await gunzipBytes(payload);
  if (restored.length !== expectedSize) {
    throw new Error(`delta payload size mismatch: expected ${expectedSize}, got ${restored.length}`);
  }
  let previous = header.delta_reference === "previous-frame" ? lastRawRgbFrame : null;
  if (header.delta_reference === "previous-frame" && !previous) {
    throw new Error("Missing previous frame for delta payload");
  }
  for (let f = 0; f < count; f++) {
    const current = f * frameBytes;
    if (previous) {
      for (let i = 0; i < frameBytes; i++) {
        restored[current + i] ^= previous[i];
      }
    }
    previous = restored.slice(current, current + frameBytes);
  }
  return restored;
}

async function framePayloadToImageData(header, payload) {
  let rawPayload;
  const isRgba = header.content_type === RAW_RGBA_DELTA_GZIP_CONTENT_TYPE;
  if (
    header.content_type === WEBP_FRAME_CONTENT_TYPE ||
    header.content_type === JPEG_FRAME_CONTENT_TYPE
  ) {
    return encodedImageToImageData(header, payload);
  } else if (header.content_type === RAW_RGB_CONTENT_TYPE) {
    rawPayload = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  } else if (header.content_type === RAW_RGB_DELTA_GZIP_CONTENT_TYPE) {
    rawPayload = await restoreDeltaGzipRawRgb(header, payload);
  } else if (isRgba) {
    rawPayload = await restoreDeltaGzipRawRgb(header, payload);
  } else {
    throw new Error(`Unsupported content type ${header.content_type}`);
  }
  const frameBytes = Number(header.bytes_per_frame);
  const frameCount = Number(header.num_frames);
  if (frameCount > 0) {
    const offset = (frameCount - 1) * frameBytes;
    lastRawRgbFrame = rawPayload.slice(offset, offset + frameBytes);
  }
  if (isRgba) {
    return rgbaToImageData(header, rawPayload);
  }
  return rgbToImageData(header, rawPayload);
}

async function encodedImageToImageData(header, payload) {
  const blob = new Blob([payload], { type: header.content_type });
  const bitmap = await createImageBitmap(blob);
  const offscreen = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(bitmap.width, bitmap.height)
    : document.createElement("canvas");
  offscreen.width = bitmap.width;
  offscreen.height = bitmap.height;
  const imageCtx = offscreen.getContext("2d", { alpha: false });
  imageCtx.drawImage(bitmap, 0, 0);
  const image = imageCtx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close?.();
  return [{ image, chunk: header.chunk_index }];
}

function drawFrame(image) {
  if (scratchCanvas.width !== image.width || scratchCanvas.height !== image.height) {
    scratchCanvas.width = image.width;
    scratchCanvas.height = image.height;
  }
  scratchCtx.putImageData(image, 0, 0);

  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const targetWidth = Math.max(1, Math.round(rect.width * dpr));
  const targetHeight = Math.max(
    1,
    Math.round((targetWidth * image.height) / image.width),
  );
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(scratchCanvas, 0, 0, canvas.width, canvas.height);
}

function renderLoop(now) {
  const targetFps = playbackFps || Number($("fps").value || 16);
  const queueSeconds = queue.length / Math.max(1, targetFps);
  const catchupFps = queueSeconds > LOW_LATENCY_QUEUE_SECONDS
    ? Math.min(MAX_CATCHUP_FPS, Math.ceil(queue.length / LOW_LATENCY_QUEUE_SECONDS))
    : targetFps;
  const targetMs = 1000 / Math.max(1, catchupFps);
  if (queue.length && now - lastFrameAt >= targetMs) {
    const item = queue.shift();
    drawFrame(item.image);
    lastFrameAt = now;
    fpsSamples.push(now);
    fpsSamples = fpsSamples.filter((t) => now - t < 1000);
    const renderedFps = String(fpsSamples.length);
    $("renderFps").textContent = renderedFps;
    $("stageRenderFps").textContent = renderedFps;
    $("chunkText").textContent = `chunk ${item.chunk}`;
    lastDisplayLagMs = now - (item.receivedAt || now);
    $("decodeText").textContent = `${Math.round(item.decodeMs || lastDecodeMs)} ms`;
    $("displayLagText").textContent = `${(lastDisplayLagMs / 1000).toFixed(1)} s`;
    updateStats();
  }
  requestAnimationFrame(renderLoop);
}

async function readFirstFrame() {
  const file = $("firstFrame").files[0];
  if (file) return new Uint8Array(await file.arrayBuffer());
  return selectedReferenceBytes || undefined;
}

function drawReferencePreviewFromImageSource(src, label) {
  const preview = $("referencePreview");
  const previewCtx = preview.getContext("2d", { alpha: false });
  previewCtx.fillStyle = "#e5e7df";
  previewCtx.fillRect(0, 0, preview.width, preview.height);
  $("referenceName").textContent = label;
  const img = new Image();
  img.onload = () => {
    const scale = Math.min(preview.width / img.width, preview.height / img.height);
    const w = img.width * scale, h = img.height * scale;
    previewCtx.fillRect(0, 0, preview.width, preview.height);
    previewCtx.drawImage(img, (preview.width - w) / 2, (preview.height - h) / 2, w, h);
    if (src.startsWith("blob:")) URL.revokeObjectURL(src);
  };
  img.onerror = () => {
    if (src.startsWith("blob:")) URL.revokeObjectURL(src);
  };
  img.src = src;
}

function drawReferencePreview(file) {
  selectedReferenceBytes = null;
  selectedReferenceLabel = file ? file.name : "";
  if (!file) return;
  drawReferencePreviewFromImageSource(URL.createObjectURL(file), file.name);
}

async function fetchPresetReference(preset) {
  if (!referenceCache.has(preset.referenceUrl)) {
    referenceCache.set(preset.referenceUrl, fetch(preset.referenceUrl).then(async (response) => {
      if (!response.ok) throw new Error(`failed to load ${preset.source}`);
      return new Uint8Array(await response.arrayBuffer());
    }).catch((error) => {
      referenceCache.delete(preset.referenceUrl);
      throw error;
    }));
  }
  return referenceCache.get(preset.referenceUrl);
}

async function setPresetReference(preset) {
  selectedReferenceBytes = await fetchPresetReference(preset);
  selectedReferenceLabel = preset.source;
  $("firstFrame").value = "";
  const blob = new Blob([selectedReferenceBytes], { type: preset.mime || "image/jpeg" });
  drawReferencePreviewFromImageSource(URL.createObjectURL(blob), selectedReferenceLabel);
}

function showError(error) {
  setStatus("Reference load failed", "error");
  addHistory(error.message || "reference load failed");
}

function closeSession(reason = "session closed by client", clearFrames = true) {
  clearQueueOnClose = clearFrames;
  socketCloseExpected = true;
  if (clearFrames) {
    queue = [];
    updateStats();
  }
  if (!ws) {
    clearQueueOnClose = false;
    setStatus("Closed");
    return;
  }
  setStatus("Closing");
  addHistory(reason);
  ws.close(1000, "client close");
}

function waitForSocketClose(socket, timeoutMs = RECONNECT_CLOSE_TIMEOUT_MS) {
  return new Promise((resolve) => {
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      resolve();
      return;
    }
    const finish = () => {
      socket.removeEventListener("close", finish);
      window.clearTimeout(timer);
      resolve();
    };
    const timer = window.setTimeout(finish, timeoutMs);
    socket.addEventListener("close", finish, { once: true });
    socket.close(1000, "replace session");
  });
}

async function connect() {
  const epoch = ++streamEpoch;
  $("connectBtn").disabled = true;
  setStatus("Preparing");
  addHistory("preparing session");
  try {
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      socketCloseExpected = true;
      setStatus("Replacing");
      addHistory("closing previous socket before reconnect");
      await waitForSocketClose(ws);
    }
    resetStreamStats();
    if (!$("firstFrame").files[0] && !selectedReferenceBytes) {
      await setPresetReference(presets[0]);
    }
    const firstFrame = await readFirstFrame();
    if (!firstFrame) {
      setStatus("Pick a reference", "error");
      addHistory("reference image required");
      $("connectBtn").disabled = false;
      return;
    }
    const previewTransportParams = PREVIEW_OUTPUT_FORMAT
      ? {
          output_compression: PREVIEW_OUTPUT_QUALITY,
          realtime_output_format: PREVIEW_OUTPUT_FORMAT,
        }
      : {};
    const init = compact({
      type: "init",
      model: $("model").value,
      prompt: $("prompt").value,
      size: $("size").value,
      fps: Number($("fps").value),
      num_frames: Number($("numFrames").value),
      seed: Number($("seed").value),
      num_inference_steps: Number($("steps").value),
      guidance_scale: Number($("guidance").value),
      max_chunks: $("continuous").checked ? undefined : 1,
      first_frame: firstFrame,
      ...previewTransportParams,
    });
    ws = new WebSocket($("serverUrl").value);
    ws.binaryType = "arraybuffer";
    socketHadError = false;
    socketCloseExpected = false;
    socketServerError = "";
    ws.onopen = () => {
      if (epoch !== streamEpoch) return;
      chunkWaitStartedAt = performance.now();
      ws.send(pack(init));
      setStatus("Starting", "live");
      addHistory(
        `session started with ${selectedReferenceLabel || "uploaded reference"}`
      );
    };
    ws.onclose = (event) => {
      if (epoch !== streamEpoch) return;
      ws = null;
      $("connectBtn").disabled = false;
      if (clearQueueOnClose) {
        queue = [];
        updateStats();
      }
      clearQueueOnClose = false;
      const reason = event.reason ? ` · ${event.reason}` : "";
      const closeText = `socket closed code=${event.code}${reason}`;
      const normalClose = event.code === 1000 || event.code === 1001;
      if (socketServerError) {
        setStatus("Server closed", "error");
        addHistory(`${closeText} · ${socketServerError}`);
      } else if (socketHadError && !socketCloseExpected && !normalClose) {
        setStatus("Socket closed", "error");
        addHistory(`${closeText} · transport error`);
      } else {
        setStatus("Closed");
        addHistory(closeText);
      }
      socketCloseExpected = false;
    };
    ws.onerror = () => {
      if (epoch !== streamEpoch) return;
      if (!socketCloseExpected) {
        socketHadError = true;
        $("connectBtn").disabled = false;
      }
    };
    ws.onmessage = (event) => {
      receiveChain = receiveChain
        .then(() => {
          if (epoch !== streamEpoch) return undefined;
          return receive(event.data);
        })
        .catch((error) => {
          if (epoch !== streamEpoch) return;
          setStatus("Receive failed", "error");
          addHistory(error.message || "receive failed");
        });
    };
  } catch (error) {
    $("connectBtn").disabled = false;
    setStatus("Init failed", "error");
    addHistory(error.message || "init failed");
  }
}

async function receive(data) {
  if (!pendingHeader) {
    pendingHeader = unpack(new Uint8Array(data));
    pendingHeader.__received_at = performance.now();
    if (pendingHeader.type === "error") {
      socketServerError = pendingHeader.content || "unknown";
      setStatus(socketServerError, "error");
      addHistory(`server error: ${socketServerError}`);
      pendingHeader = null;
    }
    if (pendingHeader) setStatus("Receiving", "live");
    return;
  }
  const header = pendingHeader;
  pendingHeader = null;
  const eventId = Number(header.event_id || 0);
  const chunkFrameCount = Number(header.num_frames || 0);
  const payloadBytes = data.byteLength || data.size || 0;
  const isEventCutover = awaitedEventId && eventId >= awaitedEventId;
  const decodedFrames = await decodeFrameBatch(header, data);
  if (isEventCutover) {
    droppedFrames += queue.length;
    queue = [];
    lastFrameAt = 0;
    if (awaitedEventSentAt) {
      const eventLatency = (performance.now() - awaitedEventSentAt) / 1000;
      $("latencyText").textContent = `${eventLatency.toFixed(1)}s · event`;
      $("stageLatencyText").textContent = `${eventLatency.toFixed(1)}s · event`;
    }
    awaitedEventId = 0;
    awaitedEventSentAt = 0;
  }
  queue.push(...decodedFrames);
  trimLiveQueue(chunkFrameCount);
  frames += chunkFrameCount;
  bytes += payloadBytes;
  $("payloadMode").textContent = header.encoding || "raw RGB";
  updatePlaybackPace(header, performance.now(), chunkFrameCount);
  setStatus("Live", "live");
  updateStats(header);
}

function updatePlaybackPace(header, now, frameCount) {
  const chunkIndex = Number(header.chunk_index || 0);
  if (currentReceiveChunk !== chunkIndex) {
    currentReceiveChunk = chunkIndex;
    currentReceiveChunkFrames = 0;
    chunkReceiveStartedAt = chunkWaitStartedAt || now;
  }
  currentReceiveChunkFrames += Number(frameCount || 0);
  const frameBatchIndex = Number(header.frame_batch_index || 0);
  const numFrameBatches = Number(header.num_frame_batches || 1);
  const isFinalFrameBatch =
    Boolean(header.is_final_frame_batch) ||
    frameBatchIndex + 1 >= numFrameBatches;
  if (!isFinalFrameBatch || !chunkWaitStartedAt) return;

  const waitSeconds = (now - chunkWaitStartedAt) / 1000;
  if (waitSeconds > 0) {
    const generatedFps = currentReceiveChunkFrames / Math.max(0.001, waitSeconds);
    const requestedFps = Number($("fps").value || 16);
    playbackFps = Math.min(
      requestedFps,
      Math.max(LOW_LATENCY_FPS_FLOOR, generatedFps * 1.05),
    );
    const latencyText = `${waitSeconds.toFixed(1)}s · ${playbackFps.toFixed(1)}fps`;
    $("latencyText").textContent = latencyText;
    $("stageLatencyText").textContent = latencyText;
  }
  chunkWaitStartedAt = performance.now();
  chunkReceiveStartedAt = chunkWaitStartedAt;
  currentReceiveChunk = null;
  currentReceiveChunkFrames = 0;
}

function sendEvent(kind, payload, historyText = null) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const eventId = nextEventId++;
  ws.send(pack({ type: "event", kind, payload, event_id: eventId }));
  if (kind === "camera_actions" || kind === "prompt") {
    awaitedEventId = eventId;
    awaitedEventSentAt = performance.now();
    trimQueueForPendingEvent();
    updateStats();
    setStatus("Updating", "live");
  }
  addHistory(`${historyText || `${kind} event sent`} · event#${eventId}`);
}

function sendCameraControl(actions) {
  const payload = repeatActions([actions]);
  sendEvent("camera_actions", payload, describeCameraEvent(actions, payload.length));
}

function repeatActions(actions, frameCount) {
  const n = Math.max(1, Number(frameCount || $("eventFrames").value || 3));
  return Array.from({ length: n }, (_, i) => actions[i % actions.length] || []);
}

async function applyPreset(preset, options = {}) {
  const sendRuntimeEvents = options.sendRuntimeEvents
    ?? Boolean(ws && ws.readyState === WebSocket.OPEN);
  selectedPreset = preset;
  $("prompt").value = preset.prompt;
  $("size").value = preset.size;
  $("fps").value = preset.fps;
  await setPresetReference(preset);
  if (sendRuntimeEvents) {
    sendEvent("prompt", preset.prompt, `prompt update · ${preset.name}`);
    const payload = repeatActions(preset.actions, 24);
    sendEvent(
      "camera_actions",
      payload,
      describeCameraScript(preset.name, preset.actions, payload.length),
    );
  }
  addHistory(`preset ${preset.name}`);
}

function describeCameraEvent(actions, samples) {
  const parts =
    actions.map((action) => describeControlAction(action, samples)).join(" + ") ||
    "No-op";
  return `camera · ${parts} · duration=${samples} frames`;
}

function describeCameraScript(name, script, samples) {
  const actionCounts = countRepeatedActions(script, samples);
  const parts =
    Array.from(actionCounts.entries())
      .map(([action, count]) => describeControlAction(action, count))
      .join(" + ") ||
    "No-op";
  return `camera preset · ${name} · ${parts} · duration=${samples} frames`;
}

function countRepeatedActions(script, samples) {
  const counts = new Map();
  for (let i = 0; i < samples; i++) {
    const actions = script[i % script.length] || [];
    actions.forEach((action) => counts.set(action, (counts.get(action) || 0) + 1));
  }
  return counts;
}

function describeControlAction(action, samples = 1) {
  const meta = CONTROL_ACTION_META[action];
  if (!meta) return `${action} (custom)`;
  const distance = describeControlDistance(meta.amount, samples);
  return `${meta.label} [${meta.type}, ${meta.axis}, ${distance}]`;
}

function describeControlDistance(amount, samples) {
  const match = /^([0-9.]+)(deg)?\/frame$/.exec(amount);
  if (!match) return amount;
  const perFrame = Number(match[1]);
  const unit = match[2] || "";
  const total = perFrame * Math.max(1, Number(samples || 1));
  return `${amount} x ${samples} frames = ${formatControlDistance(total, unit)}`;
}

function formatControlDistance(value, unit) {
  if (unit === "deg") return `${value.toFixed(0)}deg`;
  return value.toFixed(2);
}

function enhancePrompt() {
  const suffix = " high-fidelity temporal consistency, stable camera geometry, natural motion, clean lighting.";
  if (!$("prompt").value.includes("temporal consistency")) {
    $("prompt").value = `${$("prompt").value.trim()},${suffix}`;
  }
}

function compact(obj) {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v !== undefined && v !== "" && v !== null)
  );
}

function renderPresets() {
  $("presetList").innerHTML = "";
  presets.forEach((preset) => {
    const btn = document.createElement("button");
    btn.className = "preset";
    btn.dataset.tone = preset.tone;
    btn.innerHTML = `<img class="preset-thumb" src="${preset.referenceUrl}" alt="" loading="lazy" /><b>${preset.name}</b><span>${preset.source} · ${preset.size} · ${preset.fps}fps</span>`;
    btn.onclick = () => applyPreset(preset).catch(showError);
    $("presetList").appendChild(btn);
  });
}

function applyQueryParams() {
  const params = new URLSearchParams(window.location.search);
  const server = params.get("server");
  if (server) $("serverUrl").value = server;
}

function pack(value) {
  const out = [];
  const bytes = (arr) => {
    for (const item of arr) out.push(item);
  };
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

applyQueryParams();
renderPresets();
drawIdle();
applyPreset(presets[0], { sendRuntimeEvents: false }).catch(showError);
requestAnimationFrame(renderLoop);
$("connectBtn").onclick = connect;
$("stopBtn").onclick = () => closeSession();
$("sendPromptBtn").onclick = () => sendEvent("prompt", $("prompt").value);
$("enhanceBtn").onclick = enhancePrompt;
$("firstFrame").onchange = () => drawReferencePreview($("firstFrame").files[0]);
document.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("pointerdown", () => btn.classList.add("is-pressed"));
  ["pointerup", "pointercancel", "pointerleave", "blur"].forEach((eventName) => {
    btn.addEventListener(eventName, () => btn.classList.remove("is-pressed"));
  });
});
document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.onclick = () => sendCameraControl([btn.dataset.action]);
});

function isTypingTarget(target) {
  return target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function keyboardAction(event) {
  return CONTROL_KEY_ACTIONS.get(event.key.toLowerCase()) || null;
}

function sendActiveKeyboardActions(force = false) {
  const now = performance.now();
  if (!force && now - lastControlKeySentAt < KEYBOARD_EVENT_INTERVAL_MS) return;
  lastControlKeySentAt = now;
  if (activeControlActions.size || force) {
    sendCameraControl(Array.from(activeControlActions));
  }
}

function setControlButtonActive(action, active) {
  document.querySelectorAll(`[data-action="${action}"]`).forEach((btn) => {
    btn.classList.toggle("is-key-active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

document.addEventListener("keydown", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  setControlButtonActive(action, true);
  activeControlActions.add(action);
  sendActiveKeyboardActions(!event.repeat);
});

document.addEventListener("keyup", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  setControlButtonActive(action, false);
  activeControlActions.delete(action);
  sendActiveKeyboardActions(true);
});
