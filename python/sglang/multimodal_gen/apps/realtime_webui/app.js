const $ = (id) => document.getElementById(id);
const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip";
const RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";
const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v6";
const DEFAULT_PREVIEW_OUTPUT_FORMAT = "webp";
const DEFAULT_PREVIEW_OUTPUT_QUALITY = 95;
const DEFAULT_TARGET_FPS = 25;
const DEFAULT_FRAME_INTERPOLATION_EXP = 1;
const DEFAULT_FRAME_INTERPOLATION_SCALE = 1.0;
const DEFAULT_UPSCALING_SCALE = 2;
const DEFAULT_PREVIEW_SCALE = 120;
const RECONNECT_CLOSE_TIMEOUT_MS = 15000;
const LIVE_QUEUE_SECONDS = 0.45;
const LOW_LATENCY_FPS_FLOOR = 10;
const LOW_LATENCY_QUEUE_SECONDS = 0.35;
const MAX_CATCHUP_FPS = 30;
const EVENT_QUEUE_SECONDS = 0.25;
const CONTROL_BUFFERED_AMOUNT_LIMIT = 1 << 20;
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

const REACTOR_PRESET_BASE_URL = "https://www.reactor.inc/lingbot-world-fast-v1";

const reactorPresets = [
  {
    name: "Dragon Ride",
    tone: "green",
    size: "832x480",
    fps: 25,
    prompt: "A first-person perspective from the back of a colossal obsidian-black dragon in mid-flight, looking over its horned head and outstretched wings toward an ancient moss-covered castle rising above a dense jungle canopy, with mist, river gorges, and golden humid light.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/dragon-ride.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Misted Kingdom",
    tone: "green",
    size: "832x480",
    fps: 25,
    prompt: "A third-person over-the-shoulder fantasy view following a sword-slung rider on a brown horse through curling valley mist, wildflower meadows, ruined stone arches, cottages, and a many-spired castle under a ringed gas giant and crescent moon.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/misted-kingdom.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Storm Crossing",
    tone: "blue",
    size: "832x480",
    fps: 25,
    prompt: "A third-person stern view of a battered grey aluminum work boat pushing through slate-black storm swells, wet wooden deck, warm cabin lamp, orange life rings, salt mist, churning wake, and a pale silver break in the dark horizon.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/storm-crossing.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Citadel Approach",
    tone: "accent",
    size: "832x480",
    fps: 25,
    prompt: "A third-person rear view of a mud-streaked vintage Defender 4x4 driving along a cobblestone-and-sand track through a coral-lit desert canyon toward a cliff-built sandstone citadel, with cacti, red poppies, ochre dunes, and peach sunset haze.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/citadel-approach.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Spring Valley",
    tone: "green",
    size: "832x480",
    fps: 25,
    prompt: "A third-person over-the-shoulder view following a golden retriever through a sunlit meadow with a patterned floral rug, stone bench, open book, potted seedling, cherry blossoms, rounded green oaks, soft hills, and a tender watercolor storybook atmosphere.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/spring-valley.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Reef Patrol",
    tone: "blue",
    size: "832x480",
    fps: 25,
    prompt: "A third-person follow view trailing a large grey reef shark through clear tropical water above a sunlit coral reef, with drifting sediment, shifting sun-ray lattices, clouds of reef fish, a sardine bait ball, and deep blue open-water haze.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/reef-patrol.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Alpine Run",
    tone: "blue",
    size: "832x480",
    fps: 25,
    prompt: "A third-person rear view of a yellow four-person whitewater raft plunging through churning rapids in an alpine canyon, red lifejackets, yellow helmets, wet paddles, dark boulders, conifer slopes, and a snow-capped mountain at the vanishing point.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/alpine-run.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Ice Kayak",
    tone: "blue",
    size: "832x480",
    fps: 25,
    prompt: "A centered elevated third-person game camera behind a lone kayaker in a bright red kayak crossing a calm deep blue alpine lake, scattered ice blocks, mirror reflections, huge snow-covered mountain ranges, vivid sky, and crisp cold wilderness scale.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/ice-kayak.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Penguin Colony",
    tone: "green",
    size: "832x480",
    fps: 25,
    prompt: "A third-person follow view of a single black-and-white penguin waddling across a windswept Antarctic ice shelf toward a distant colony, crystalline snow, small flippers, scattered dark boulders, rocky shoreline, and pale polar sky.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/penguin.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Mars Mountain",
    tone: "accent",
    size: "832x480",
    fps: 25,
    prompt: "A centered third-person rear view of a six-wheeled Martian rover marked XR-7A P-3317 crossing cracked basalt toward a vast volcanic mountain, dusty rose twilight, ochre wheel plumes, weathered grey panels, and a cold alien horizon.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/mars-rover.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Seaside Adventurer",
    tone: "green",
    size: "832x480",
    fps: 25,
    prompt: "A centered third-person anime view behind a young girl on a flower-covered coastal hillside overlooking a sparkling blue bay, rolling green hills, sailboats, dramatic cliffs, a small lighthouse, huge fluffy clouds, and warm hand-painted adventure atmosphere.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/anime3.png`,
    source: "Reactor LingBot preset",
    mime: "image/png",
  },
  {
    name: "Roman Chariot",
    tone: "accent",
    size: "832x480",
    fps: 25,
    prompt: "A centered elevated third-person game camera behind a Roman warrior riding an ancient chariot pulled by two white horses across an open grassy field, worn stone path, Roman ruins, broken columns, bright midday sky, and epic historical scale.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/chariot.png`,
    source: "Reactor LingBot preset",
    mime: "image/png",
  },
  {
    name: "Asylum Corridor",
    tone: "accent",
    size: "832x480",
    fps: 25,
    prompt: "A third-person over-the-shoulder traversal behind a man in a wet leather jacket holding a flashlight down a derelict asylum corridor, standing water, torn vinyl strips, rusted ceiling debris, bloodstains, a toppled wheelchair, and a distant cyan-grey doorway glow.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/horror.jpg`,
    source: "Reactor LingBot preset",
  },
];

const examplePresets = [
  { name: "Dragon Dolly", tone: "green", size: "832x480", fps: 25, prompt: "A stable first-person dolly from the same dragon-rider viewpoint, keeping the black dragon head, horns, wings, jungle canopy, and distant castle consistent; slow forward camera motion, natural parallax, no creature morphing, no scene replacement.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/00/image.jpg", source: "LingBot example 00" },
  { name: "Stone Orbit", tone: "blue", size: "832x480", fps: 25, prompt: "A controlled look-around of the stone monument, overcast daylight, consistent geometry, subtle camera arc.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/01/image.jpg", source: "LingBot example 01" },
  { name: "Urban Tilt", tone: "accent", size: "832x480", fps: 25, prompt: "A cinematic urban wall shot with a slow tilt and slight forward movement, warm backlight, stable architecture.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/02/image.jpg", source: "LingBot example 02" },
  { name: "Lake Scout", tone: "green", size: "832x480", fps: 25, prompt: "A calm scouting shot across the lake, gentle camera drift, crisp mountains, stable reflections.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/03/image.jpg", source: "LingBot example 03" },
  { name: "Ziggy Stardust", tone: "accent", size: "832x480", fps: 25, prompt: "A static night view of a narrow London alley in soft rain, wet pavement reflecting a yellow streetlamp, the blue K. West sign glowing above a doorway, cardboard boxes near the wall, a pale parked car in the distance, and a slender glam-rock figure holding a guitar under the lamp; preserve the album-cover composition, brick storefronts, muted teal and amber colors, subtle rain shimmer only.", referenceUrl: "https://upload.wikimedia.org/wikipedia/en/0/01/ZiggyStardust.jpg", source: "David Bowie Ziggy Stardust artwork", mime: "image/jpeg" },
  { name: "Plastic Beach", tone: "blue", size: "832x480", fps: 25, prompt: "A static locked-off view of the back side of Plastic Beach: a pastel toy-like island hotel built from plastic debris and washed-up junk in the open ocean, turquoise water in the foreground, distant horizon, clouds slowly drifting behind the island, tiny stars gently twinkling in the dusk sky, an occasional shooting star, and tiny distant pigeons crossing the sky; subtle water shimmer only, keep the island geometry fixed, no camera orbit, no dolly, no zoom.", referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/b8/f9/b9/b8f9b9f8-a609-bde2-0302-349436ffc508/825646291038.jpg/600x600bb.jpg", source: "Gorillaz Plastic Beach artwork", mime: "image/jpeg" },
  { name: "Plastic Ono Band", tone: "green", size: "832x480", fps: 25, prompt: "A quiet sunlit park under a massive tree, a solitary figure resting in the grass, soft summer haze, restrained documentary camera, intimate and naturalistic.", referenceUrl: "https://upload.wikimedia.org/wikipedia/en/a/a4/JLPOBCover.jpg", source: "John Lennon/Plastic Ono Band artwork", mime: "image/jpeg" },
  { name: "Kid A", tone: "accent", size: "832x480", fps: 25, prompt: "A cold surreal mountain range with sharp icy peaks, black-red storm clouds, glacial light, slow lateral pan, abstract digital texture, uneasy atmospheric scale.", referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music122/v4/bd/8e/13/bd8e1358-b367-a689-cb84-cebd0b067dc4/634904078263.png/600x600bb.jpg", source: "Radiohead Kid A artwork", mime: "image/jpeg" },
];

const presets = [
  ...reactorPresets,
  ...examplePresets,
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
let decodeChain = Promise.resolve();
let pendingDecodeBatches = 0;
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
let encodedDecodeErrors = 0;
let socketHadError = false;
let socketCloseExpected = false;
let socketServerError = "";
let renderedPreviewFrames = 0;
let previewScaleFrame = 0;
const decodeRequests = new Map();
let controlStateController = null;

const stage = document.querySelector(".stage");
const previewFrame = document.querySelector(".preview-frame");
const canvas = $("viewport");
const ctx = canvas.getContext("2d", { alpha: false });
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d", { alpha: false });

function setStatus(text, kind = "") {
  $("statusText").textContent = text;
  $("statusDot").className = "dot" + (kind ? ` ${kind}` : "");
}

function setPreviewState(state) {
  if (!stage) return;
  stage.dataset.previewState = state;
  canvas.setAttribute("aria-busy", state === "waiting" ? "true" : "false");
}

function addHistory(text) {
  const item = document.createElement("span");
  item.textContent = text;
  $("historyList").prepend(item);
  while ($("historyList").children.length > 8) $("historyList").lastChild.remove();
}

function drawIdle() {
  const w = 1280, h = 720;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  setPreviewState("idle");
  renderedPreviewFrames = 0;
  ctx.fillStyle = "#11140f";
  ctx.fillRect(0, 0, w, h);

  const surface = ctx.createLinearGradient(0, 0, 0, h);
  surface.addColorStop(0, "rgba(238,241,236,0.045)");
  surface.addColorStop(0.5, "rgba(238,241,236,0.012)");
  surface.addColorStop(1, "rgba(0,0,0,0.16)");
  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(238,241,236,0.11)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.strokeStyle = "rgba(238,241,236,0.08)";
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(w * 0.38, h * 0.42, w * 0.24, h * 0.16, 18);
  } else {
    ctx.rect(w * 0.38, h * 0.42, w * 0.24, h * 0.16);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(238,241,236,0.22)";
  for (let i = -1; i <= 1; i++) {
    ctx.beginPath();
    ctx.arc(w * 0.5 + i * 22, h * 0.5, 4.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function resetStreamStats() {
  pendingHeader = null;
  clearFrameQueue();
  frames = 0;
  bytes = 0;
  fpsSamples = [];
  chunkWaitStartedAt = 0;
  clearQueueOnClose = false;
  playbackFps = Number($("fps").value || DEFAULT_TARGET_FPS);
  droppedFrames = 0;
  decodeChain = Promise.resolve();
  pendingDecodeBatches = 0;
  awaitedEventId = 0;
  awaitedEventSentAt = 0;
  chunkReceiveStartedAt = 0;
  currentReceiveChunk = null;
  currentReceiveChunkFrames = 0;
  encodedDecodeErrors = 0;
  renderedPreviewFrames = 0;
  controlStateController?.reset({ sendRelease: false });
  resetDecoderState();
  updateStats();
  $("renderFps").textContent = "0";
  $("latencyText").textContent = "-";
  $("stageLatencyText").textContent = "-";
  $("decodeText").textContent = "-";
  $("displayLagText").textContent = "-";
  $("serverSendText").textContent = "-";
  $("chunkPayloadText").textContent = "-";
  $("theoreticalFpsText").textContent = "-";
  $("chunkText").textContent = "chunk -";
  $("payloadMode").textContent = selectedTransportLabel();
  updateOutputSizeText();
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
  if (!isWorkerDecodableContentType(header.content_type)) {
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

  const payload = await payloadToArrayBuffer(data);
  const id = decodeRequestId++;
  const decodeHeader = { ...header, __decode_id: id };
  const useTransfer = isWorkerDecodableRawContentType(header.content_type);
  try {
    return await new Promise((resolve, reject) => {
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
          useTransfer ? [payload] : [],
        );
      } catch (error) {
        decodeRequests.delete(id);
        reject(error);
      }
    });
  } catch (error) {
    if (isEncodedPreviewContentType(header.content_type)) {
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
    throw error;
  }
}

function isWorkerDecodableContentType(contentType) {
  return (
    isWorkerDecodableRawContentType(contentType) ||
    isEncodedPreviewContentType(contentType)
  );
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
  const targetFps = Number($("fps").value || DEFAULT_TARGET_FPS);
  const maxQueue = Math.max(
    Number(latestFrameCount || 0),
    Math.round(targetFps * LIVE_QUEUE_SECONDS),
  );
  if (queue.length <= maxQueue) return;
  const dropCount = queue.length - maxQueue;
  dropQueuedFrames(dropCount);
  droppedFrames += dropCount;
}

function liveQueueFrameFloor(header, decodedFrameCount) {
  const frameBatchCount = Number(header.num_frame_batches || 1);
  if (!isEncodedPreviewContentType(header.content_type) || frameBatchCount <= 1) {
    return decodedFrameCount;
  }
  return Math.max(decodedFrameCount, frameBatchCount * decodedFrameCount);
}

function trimQueueForPendingEvent() {
  const targetFps = playbackFps || Number($("fps").value || DEFAULT_TARGET_FPS);
  const keep = Math.max(1, Math.round(targetFps * EVENT_QUEUE_SECONDS));
  if (queue.length <= keep) return;
  const dropCount = queue.length - keep;
  dropQueuedFrames(dropCount);
  droppedFrames += dropCount;
}

function clearFrameQueue() {
  for (const item of queue) item.image?.close?.();
  queue = [];
}

function dropQueuedFrames(count) {
  for (const item of queue.splice(0, count)) item.image?.close?.();
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

function isEncodedPreviewContentType(contentType) {
  return (
    contentType === WEBP_FRAME_CONTENT_TYPE ||
    contentType === JPEG_FRAME_CONTENT_TYPE
  );
}

async function encodedImageToImageData(header, payload) {
  const blob = new Blob([payload], { type: header.content_type });
  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(blob);
      return [{ image: bitmap, chunk: header.chunk_index }];
    } catch (error) {
      return [await encodedImageElementFallback(blob, header, error)];
    }
  }
  return [
    await encodedImageElementFallback(
      blob,
      header,
      new Error("createImageBitmap unavailable"),
    ),
  ];
}

async function encodedImageElementFallback(blob, header, createBitmapError) {
  const url = URL.createObjectURL(blob);
  try {
    const image = await loadImageElement(url, createBitmapError);
    if (
      scratchCanvas.width !== image.naturalWidth ||
      scratchCanvas.height !== image.naturalHeight
    ) {
      scratchCanvas.width = image.naturalWidth;
      scratchCanvas.height = image.naturalHeight;
    }
    scratchCtx.drawImage(image, 0, 0);
    return {
      image: scratchCtx.getImageData(0, 0, image.naturalWidth, image.naturalHeight),
      chunk: header.chunk_index,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function loadImageElement(url, createBitmapError) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(createBitmapError);
    image.src = url;
  });
}

function handleEncodedPreviewDecodeError(error, header, data, payloadBytes) {
  encodedDecodeErrors += 1;
  const signature = payloadSignature(data);
  const mode = shortPayloadMode(header.content_type);
  const message = error?.message || "encoded preview decode failed";
  $("decodeText").textContent = `drop ${encodedDecodeErrors}`;
  setStatus("Decode dropped", "error");
  addHistory(
    `decode drop c${header.chunk_index} ${mode} ${formatBytes(payloadBytes)} ${signature} · ${message}`,
  );
}

function payloadSignature(data) {
  let bytes;
  if (data instanceof Uint8Array) {
    bytes = data.subarray(0, Math.min(12, data.byteLength));
  } else if (data instanceof ArrayBuffer) {
    bytes = new Uint8Array(data, 0, Math.min(12, data.byteLength));
  } else {
    return "";
  }
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function payloadToArrayBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (data instanceof Uint8Array) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  return data.arrayBuffer();
}

function drawFrame(image) {
  const sourceWidth = image.width;
  const sourceHeight = image.height;
  let drawSource = image;
  if (image instanceof ImageData) {
    if (scratchCanvas.width !== sourceWidth || scratchCanvas.height !== sourceHeight) {
      scratchCanvas.width = sourceWidth;
      scratchCanvas.height = sourceHeight;
    }
    scratchCtx.putImageData(image, 0, 0);
    drawSource = scratchCanvas;
  }

  if (canvas.width !== sourceWidth || canvas.height !== sourceHeight) {
    canvas.width = sourceWidth;
    canvas.height = sourceHeight;
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(drawSource, 0, 0, sourceWidth, sourceHeight);
  renderedPreviewFrames += 1;
  setPreviewState("live");
  if (!(image instanceof ImageData)) image.close?.();
}

function renderLoop(now) {
  const targetFps = playbackFps || Number($("fps").value || DEFAULT_TARGET_FPS);
  const queueSeconds = queue.length / Math.max(1, targetFps);
  const catchupFps = !$("superResolution").checked && queueSeconds > LOW_LATENCY_QUEUE_SECONDS
    ? Math.min(MAX_CATCHUP_FPS, Math.ceil(queue.length / LOW_LATENCY_QUEUE_SECONDS))
    : targetFps;
  const targetMs = 1000 / Math.max(1, catchupFps);
  const elapsedMs = lastFrameAt ? now - lastFrameAt : targetMs;
  if (queue.length && elapsedMs >= targetMs) {
    const item = queue.shift();
    drawFrame(item.image);
    // preserve remainder so 25fps is not quantized to 20fps on 60Hz rAF
    lastFrameAt = !lastFrameAt || elapsedMs > targetMs * 4
      ? now
      : now - (elapsedMs % targetMs);
    fpsSamples.push(now);
    fpsSamples = fpsSamples.filter((t) => now - t < 1000);
    const renderedFps = String(fpsSamples.length);
    $("renderFps").textContent = renderedFps;
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
  if (!renderedPreviewFrames) setPreviewState("idle");
  addHistory(error.message || "reference load failed");
}

function abortCurrentSession(reason = "session closed by client", {
  clearFrames = true,
  expectedClose = true,
  keepConnectDisabled = false,
} = {}) {
  const socket = ws;
  ws = null;
  streamEpoch++;
  clearQueueOnClose = clearFrames;
  socketCloseExpected = expectedClose;
  controlStateController?.reset({ sendRelease: false });
  pendingHeader = null;
  rejectPendingDecodes("session aborted");
  resetDecoderState();
  if (clearFrames) {
    clearFrameQueue();
    updateStats();
  }
  if (!socket) {
    clearQueueOnClose = false;
    if (!keepConnectDisabled) $("connectBtn").disabled = false;
    setStatus("Closed");
    if (!renderedPreviewFrames) setPreviewState("idle");
    return null;
  }
  if (!keepConnectDisabled) $("connectBtn").disabled = false;
  setStatus(expectedClose ? "Closing" : "Aborting");
  if (!renderedPreviewFrames) setPreviewState("idle");
  addHistory(reason);
  socket.close(expectedClose ? 1000 : 1011, reason.slice(0, 120));
  return socket;
}

function closeSession(reason = "session closed by client", clearFrames = true) {
  abortCurrentSession(reason, { clearFrames, expectedClose: true });
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
  $("connectBtn").disabled = true;
  setStatus("Preparing");
  setPreviewState("waiting");
  addHistory("preparing session");
  try {
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      setStatus("Replacing");
      const oldSocket = abortCurrentSession("closing previous socket before reconnect", {
        keepConnectDisabled: true,
      });
      await waitForSocketClose(oldSocket);
    }
    resetStreamStats();
    const epoch = ++streamEpoch;
    if (!$("firstFrame").files[0] && !selectedReferenceBytes) {
      await setPresetReference(presets[0]);
    }
    const firstFrame = await readFirstFrame();
    if (!firstFrame) {
      setStatus("Pick a reference", "error");
      setPreviewState("idle");
      addHistory("reference image required");
      $("connectBtn").disabled = false;
      return;
    }
    const previewTransportParams = readPreviewTransportParams();
    const frameInterpolationParams = readFrameInterpolationParams();
    const superResolutionParams = readSuperResolutionParams();
    const init = compact({
      type: "init",
      model: $("model").value,
      prompt: $("prompt").value,
      size: $("size").value,
      fps: Number($("fps").value || DEFAULT_TARGET_FPS),
      num_frames: Number($("numFrames").value),
      seed: Number($("seed").value),
      num_inference_steps: Number($("steps").value),
      guidance_scale: Number($("guidance").value),
      realtime_causal_sink_size: readOptionalInteger("sinkSize"),
      realtime_causal_kv_cache_num_frames: readOptionalInteger("windowFrames"),
      max_chunks: $("continuous").checked ? undefined : 1,
      first_frame: firstFrame,
      ...previewTransportParams,
      ...frameInterpolationParams,
      ...superResolutionParams,
    });
    document.activeElement?.blur?.();
    canvas.tabIndex = 0;
    canvas.focus();
    const socket = new WebSocket($("serverUrl").value);
    ws = socket;
    socket.binaryType = "arraybuffer";
    socketHadError = false;
    socketCloseExpected = false;
    socketServerError = "";
    socket.onopen = () => {
      if (epoch !== streamEpoch) return;
      chunkWaitStartedAt = performance.now();
      socket.send(pack(init));
      setStatus("Starting", "live");
      addHistory(
        `session started with ${selectedReferenceLabel || "uploaded reference"}`
      );
    };
    socket.onclose = (event) => {
      if (epoch !== streamEpoch) return;
      if (ws === socket) ws = null;
      $("connectBtn").disabled = false;
      if (clearQueueOnClose) {
        clearFrameQueue();
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
      if (!renderedPreviewFrames) setPreviewState("idle");
      socketCloseExpected = false;
    };
    socket.onerror = () => {
      if (epoch !== streamEpoch) return;
      if (!socketCloseExpected) {
        socketHadError = true;
        $("connectBtn").disabled = false;
      }
    };
    socket.onmessage = (event) => {
      if (epoch !== streamEpoch) return;
      try {
        receive(event.data, epoch);
      } catch (error) {
        handleReceiveError(error, epoch);
      }
    };
  } catch (error) {
    $("connectBtn").disabled = false;
    setStatus("Init failed", "error");
    if (!renderedPreviewFrames) setPreviewState("idle");
    addHistory(error.message || "init failed");
  }
}

function handleReceiveError(error, epoch) {
  if (epoch !== streamEpoch) return;
  setStatus("Receive failed", "error");
  addHistory(error.message || "receive failed");
  abortCurrentSession(error.message || "receive failed", {
    clearFrames: false,
    expectedClose: false,
  });
}

function receive(data, epoch) {
  if (!pendingHeader) {
    const message = unpack(new Uint8Array(data));
    message.__received_at = performance.now();
    if (message.type === "error") {
      socketServerError = message.content || "unknown";
      setStatus(socketServerError, "error");
      addHistory(`server error: ${socketServerError}`);
      return;
    }
    if (message.type === "chunk_stats") {
      updateServerChunkStats(message);
      return;
    }
    if (message.type === "frame_batch") {
      const payload = message.payload;
      delete message.payload;
      pendingDecodeBatches += 1;
      decodeChain = decodeChain
        .then(() => decodeAndEnqueueFrameBatch(message, payload, epoch))
        .catch((error) => handleReceiveError(error, epoch))
        .finally(() => {
          pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
        });
      setStatus("Live", "live");
      return;
    }
    pendingHeader = message;
    if (pendingHeader) setStatus("Live", "live");
    return;
  }
  const header = pendingHeader;
  pendingHeader = null;
  pendingDecodeBatches += 1;
  decodeChain = decodeChain
    .then(() => decodeAndEnqueueFrameBatch(header, data, epoch))
    .catch((error) => handleReceiveError(error, epoch))
    .finally(() => {
      pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
    });
}

async function decodeAndEnqueueFrameBatch(header, data, epoch) {
  const eventId = Number(header.event_id || 0);
  const chunkFrameCount = Number(header.num_frames || 0);
  const payloadBytes = data.byteLength || data.size || 0;
  const isEventCutover = awaitedEventId && eventId >= awaitedEventId;
  let decodedFrames;
  try {
    decodedFrames = await decodeFrameBatch(header, data);
    if (isEncodedPreviewContentType(header.content_type)) encodedDecodeErrors = 0;
  } catch (error) {
    if (!isEncodedPreviewContentType(header.content_type)) throw error;
    handleEncodedPreviewDecodeError(error, header, data, payloadBytes);
    return;
  }
  if (epoch !== streamEpoch) {
    for (const item of decodedFrames) item.image?.close?.();
    return;
  }
  if (isEventCutover) {
    droppedFrames += queue.length;
    clearFrameQueue();
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
  trimLiveQueue(liveQueueFrameFloor(header, chunkFrameCount));
  frames += chunkFrameCount;
  bytes += payloadBytes;
  $("payloadMode").textContent = header.encoding || "raw RGB";
  updateOutputSizeFromHeader(header);
  updatePlaybackPace(header, performance.now(), chunkFrameCount);
  setStatus("Live", "live");
  updateStats(header);
}

function updateServerChunkStats(stats) {
  const rawWrite = Number(stats.raw_write_ms || 0) / 1000;
  const wsWrite = Number(stats.ws_write_ms || 0) / 1000;
  const chunkTotal = Number(stats.chunk_total_ms || 0) / 1000;
  const numFrames = Number(stats.num_frames || 0);
  const requestedFps = Number($("fps").value || DEFAULT_TARGET_FPS);
  const theoreticalFps = chunkTotal > 0 ? numFrames / chunkTotal : 0;
  const realtimeRatio = requestedFps > 0 ? theoreticalFps / requestedFps : 0;
  $("serverSendText").textContent = `raw ${rawWrite.toFixed(2)}s · ws ${wsWrite.toFixed(2)}s`;
  $("chunkPayloadText").textContent = `${formatBytes(stats.ws_payload_bytes || 0)} · ${numFrames}f`;
  $("theoreticalFpsText").textContent = theoreticalFps > 0
    ? `${theoreticalFps.toFixed(1)} fps · ${realtimeRatio.toFixed(2)}x`
    : "-";
  if (stats.content_type) $("payloadMode").textContent = shortPayloadMode(stats.content_type);
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
    const requestedFps = Number($("fps").value || DEFAULT_TARGET_FPS);
    const playbackFloor = $("superResolution").checked ? 1 : LOW_LATENCY_FPS_FLOOR;
    playbackFps = Math.min(
      requestedFps,
      Math.max(playbackFloor, generatedFps * 1.05),
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
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addHistory(`${historyText || `${kind} event`} · socket not open`);
    return null;
  }
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
  return eventId;
}

function sendCameraControlTransitions(transitions) {
  if (!transitions.length) return null;
  const payload = {
    mode: "state",
    transitions: transitions.map((transition) => ({
      actions: transition.actions,
      client_ts_ms: transition.clientTsMs,
    })),
  };
  return sendEvent(
    "camera_actions",
    payload,
    describeCameraStateEvent(transitions),
  );
}

async function applyPreset(preset, options = {}) {
  const sendRuntimeEvents = options.sendRuntimeEvents
    ?? Boolean(ws && ws.readyState === WebSocket.OPEN);
  selectedPreset = preset;
  $("prompt").value = preset.prompt;
  $("size").value = preset.size;
  $("fps").value = preset.fps;
  updateOutputSizeText();
  await setPresetReference(preset);
  if (sendRuntimeEvents) {
    sendEvent("prompt", preset.prompt, `prompt update · ${preset.name}`);
  }
  addHistory(`preset ${preset.name}`);
}

function describeCameraStateEvent(transitions) {
  const parts = transitions
    .map((transition) => describeControlActions(transition.actions))
    .join(" -> ");
  return `camera state · ${parts} · transitions=${transitions.length}`;
}

function describeControlActions(actions) {
  return actions.map((action) => describeControlAction(action)).join(" + ") || "No-op";
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

function modelsUrlFromServerUrl(serverUrl) {
  const url = new URL(serverUrl, window.location.href);
  if (url.protocol === "ws:") url.protocol = "http:";
  if (url.protocol === "wss:") url.protocol = "https:";
  url.pathname = "/v1/models";
  url.search = "";
  url.hash = "";
  return url.toString();
}

function firstServedModelInfo(payload) {
  if (Array.isArray(payload?.data) && payload.data.length > 0) return payload.data[0];
  if (payload && typeof payload === "object") return payload;
  return null;
}

function servedModelId(info) {
  return String(info?.id || info?.model || info?.root || "");
}

function presetForModelInfo(info) {
  const id = servedModelId(info).toLowerCase();
  if (!id) return null;
  return presets.find((preset) => (
    preset.model && id.includes(preset.model.toLowerCase())
  )) || null;
}

async function queryServerModelInfo(options = {}) {
  const applyPresetForModel = options.applyPresetForModel ?? true;
  let info;
  try {
    const response = await fetch(modelsUrlFromServerUrl($("serverUrl").value), {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`/v1/models ${response.status}`);
    info = firstServedModelInfo(await response.json());
  } catch (error) {
    addHistory(`model query failed · ${error.message || "unknown"}`);
    return null;
  }
  if (!info) return null;

  const modelId = servedModelId(info);
  const preset = presetForModelInfo(info);
  if (preset && applyPresetForModel && preset !== selectedPreset) {
    await applyPreset(preset, { sendRuntimeEvents: false });
  }
  if (modelId) $("model").value = modelId;
  addHistory(
    preset
      ? `server model · ${preset.name}`
      : `server model · ${modelId || "unknown"}`,
  );
  return info;
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

function readOptionalInteger(id) {
  const value = $(id).value;
  if (value === "") return undefined;
  return Number(value);
}

function readPreviewTransportParams() {
  const outputFormat = $("transportFormat").value;
  const outputQuality = Number($("transportQuality").value || DEFAULT_PREVIEW_OUTPUT_QUALITY);
  if (!outputFormat) return {};
  const params = { realtime_output_format: outputFormat };
  if (outputFormat === "webp" || outputFormat === "jpeg") {
    params.output_compression = outputQuality;
  }
  return params;
}

function readFrameInterpolationParams() {
  if (!$("frameInterpolation").checked) return {};
  return {
    enable_frame_interpolation: true,
    frame_interpolation_exp: DEFAULT_FRAME_INTERPOLATION_EXP,
    frame_interpolation_scale: DEFAULT_FRAME_INTERPOLATION_SCALE,
  };
}

function readUpscalingScale() {
  return Number($("upscalingScale").value || DEFAULT_UPSCALING_SCALE);
}

function readSuperResolutionParams() {
  if (!$("superResolution").checked) return {};
  return {
    enable_upscaling: true,
    upscaling_scale: readUpscalingScale(),
  };
}

function parseSizeValue(sizeText) {
  const match = /^(\d+)\s*x\s*(\d+)$/i.exec(String(sizeText || "").trim());
  if (!match) return null;
  return {
    width: Number(match[1]),
    height: Number(match[2]),
  };
}

function updateOutputSizeText(width = null, height = null) {
  let outputWidth = Number(width || 0);
  let outputHeight = Number(height || 0);
  const srEnabled = $("superResolution").checked;
  const scale = srEnabled ? readUpscalingScale() : 1;
  if (!outputWidth || !outputHeight) {
    const base = parseSizeValue($("size").value);
    if (base) {
      outputWidth = base.width * scale;
      outputHeight = base.height * scale;
    }
  }
  $("outputSizeText").textContent = outputWidth && outputHeight
    ? `${outputWidth}x${outputHeight}${srEnabled ? ` · SR ${scale}x` : ""}`
    : "-";
}

function updateOutputSizeFromHeader(header) {
  const width = Number(header.width || 0);
  const height = Number(header.height || 0);
  if (width && height) updateOutputSizeText(width, height);
}

function updateSuperResolutionControls() {
  $("upscalingScale").disabled = !$("superResolution").checked;
  updateOutputSizeText();
}

function setPreviewScale(value) {
  if (!previewFrame) return;
  const scale = Math.max(80, Math.min(170, Number(value || DEFAULT_PREVIEW_SCALE)));
  $("previewScale").value = String(scale);
  $("previewScaleText").textContent = `${scale}%`;
  if (previewScaleFrame) cancelAnimationFrame(previewScaleFrame);
  previewScaleFrame = requestAnimationFrame(() => {
    previewScaleFrame = 0;
    previewFrame.style.setProperty("--preview-scale", String(scale / 100));
  });
}

function selectedTransportLabel() {
  const select = $("transportFormat");
  return select.options[select.selectedIndex]?.textContent || "raw RGB";
}

function shortPayloadMode(contentType) {
  if (contentType === WEBP_FRAME_CONTENT_TYPE) return "webp";
  if (contentType === JPEG_FRAME_CONTENT_TYPE) return "jpeg";
  if (contentType === RAW_RGB_DELTA_GZIP_CONTENT_TYPE) return "delta-gzip";
  if (contentType === RAW_RGB_CONTENT_TYPE) return "raw RGB";
  return contentType;
}

function formatBytes(value) {
  return `${(Number(value || 0) / 1048576).toFixed(1)} MB`;
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

async function applyQueryParams() {
  const params = new URLSearchParams(window.location.search);
  const presetKey = params.get("preset");
  let appliedPreset = false;
  if (presetKey) {
    const normalized = presetKey.toLowerCase();
    const preset = presets.find((item) => (
      item.name.toLowerCase() === normalized
      || item.name.toLowerCase().replaceAll(" ", "-") === normalized
    ));
    if (preset && preset !== selectedPreset) {
      await applyPreset(preset, { sendRuntimeEvents: false });
      appliedPreset = true;
    }
  }
  const server = params.get("server");
  if (server) $("serverUrl").value = server;
  const model = params.get("model");
  if (model) $("model").value = model;
  $("transportFormat").value = params.get("transport") || DEFAULT_PREVIEW_OUTPUT_FORMAT;
  $("transportQuality").value = params.get("quality") || String(DEFAULT_PREVIEW_OUTPUT_QUALITY);
  const srParam = params.get("sr");
  $("superResolution").checked = srParam === "1" || srParam === "true";
  $("upscalingScale").value = params.get("sr_scale") || String(DEFAULT_UPSCALING_SCALE);
  setPreviewScale(params.get("preview_scale") || params.get("zoom"));
  updateSuperResolutionControls();
  return {
    model: Boolean(model),
    preset: Boolean(presetKey && appliedPreset),
  };
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
    if (b === 0xca) {
      const value = new DataView(buf.buffer, buf.byteOffset + i, 4).getFloat32(0);
      i += 4;
      return value;
    }
    if (b === 0xcb) {
      const value = new DataView(buf.buffer, buf.byteOffset + i, 8).getFloat64(0);
      i += 8;
      return value;
    }
    if (b === 0xc4) return readBin(buf[i++]);
    if (b === 0xc5) return readBin((buf[i++] << 8) | buf[i++]);
    if (b === 0xc6) {
      return readBin(
        (buf[i++] * 16777216) + (buf[i++] << 16) + (buf[i++] << 8) + buf[i++],
      );
    }
    if (b === 0xd9) return readStr(buf[i++]);
    if (b === 0xda) return readStr((buf[i++] << 8) | buf[i++]);
    if (b === 0xde) return readMap((buf[i++] << 8) | buf[i++]);
    throw new Error(`Unsupported msgpack byte ${b}`);
  };
  const readStr = (n) => text.decode(buf.slice(i, i += n));
  const readBin = (n) => buf.subarray(i, i += n);
  const readMap = (n) => {
    const obj = {};
    for (let j = 0; j < n; j++) obj[read()] = read();
    return obj;
  };
  return read();
}

renderPresets();
drawIdle();
setPreviewScale(DEFAULT_PREVIEW_SCALE);
updateSuperResolutionControls();
applyPreset(presets[0], { sendRuntimeEvents: false })
  .then(applyQueryParams)
  .then((query) => queryServerModelInfo({
    applyPresetForModel: !query.model && !query.preset,
  }))
  .catch(showError);
requestAnimationFrame(renderLoop);
$("connectBtn").onclick = connect;
$("stopBtn").onclick = () => closeSession();
$("sendPromptBtn").onclick = () => sendEvent("prompt", $("prompt").value);
$("enhanceBtn").onclick = enhancePrompt;
$("firstFrame").onchange = () => drawReferencePreview($("firstFrame").files[0]);
$("size").addEventListener("input", () => updateOutputSizeText());
$("superResolution").addEventListener("change", updateSuperResolutionControls);
$("upscalingScale").addEventListener("change", () => updateOutputSizeText());
$("previewScale").addEventListener("input", () => setPreviewScale($("previewScale").value));
$("serverUrl").addEventListener("change", () => {
  queryServerModelInfo({ applyPresetForModel: true }).catch(showError);
});
document.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("pointerdown", () => btn.classList.add("is-pressed"));
  ["pointerup", "pointercancel", "pointerleave", "blur"].forEach((eventName) => {
    btn.addEventListener(eventName, () => btn.classList.remove("is-pressed"));
  });
});
document.querySelectorAll("[data-action]").forEach((btn) => {
  const action = btn.dataset.action;
  btn.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    controlStateController.setAction(action, true);
  });
  ["pointerup", "pointercancel", "pointerleave", "blur"].forEach((eventName) => {
    btn.addEventListener(eventName, (event) => {
      event.preventDefault();
      controlStateController.setAction(action, false);
    });
  });
});

function isTypingTarget(target) {
  return target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function keyboardAction(event) {
  return CONTROL_KEY_ACTIONS.get(event.key.toLowerCase()) || null;
}

function setControlButtonActive(action, active) {
  document.querySelectorAll(`[data-action="${action}"]`).forEach((btn) => {
    btn.classList.toggle("is-key-active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

class ControlStateController {
  constructor() {
    this.activeActions = new Set();
    this.pendingTransitions = [];
    this.flushScheduled = false;
  }

  reset({ sendRelease = false } = {}) {
    const hadActions = this.activeActions.size > 0;
    this.activeActions.clear();
    this.pendingTransitions = [];
    this.flushScheduled = false;
    this.updateButtons();
    if (sendRelease && hadActions) {
      this.enqueueTransition();
    }
  }

  setAction(action, active) {
    const hadAction = this.activeActions.has(action);
    if (active === hadAction) return;
    if (active) {
      this.activeActions.add(action);
    } else {
      this.activeActions.delete(action);
    }
    this.updateButtons();
    this.enqueueTransition();
  }

  releaseAll() {
    this.reset({ sendRelease: true });
  }

  enqueueTransition() {
    const actions = Array.from(this.activeActions).sort();
    const last = this.pendingTransitions[this.pendingTransitions.length - 1];
    if (last && this.sameActions(last.actions, actions)) return;
    this.pendingTransitions.push({
      actions,
      clientTsMs: Math.round(performance.now()),
    });
    this.compactPendingIfNeeded();
    this.scheduleFlush();
  }

  scheduleFlush() {
    if (this.flushScheduled) return;
    this.flushScheduled = true;
    queueMicrotask(() => {
      this.flushScheduled = false;
      this.flush();
    });
  }

  flush() {
    if (!this.pendingTransitions.length) return;
    if (ws && ws.bufferedAmount > CONTROL_BUFFERED_AMOUNT_LIMIT) {
      this.compactPendingToLatestPulse();
    }
    const transitions = this.pendingTransitions;
    this.pendingTransitions = [];
    sendCameraControlTransitions(transitions);
  }

  compactPendingIfNeeded() {
    if (this.pendingTransitions.length <= 8) return;
    this.compactPendingToLatestPulse();
  }

  compactPendingToLatestPulse() {
    const final = this.pendingTransitions[this.pendingTransitions.length - 1];
    const latestPulse = [...this.pendingTransitions]
      .reverse()
      .find((transition) => transition.actions.length > 0);
    if (latestPulse && !this.sameActions(latestPulse.actions, final.actions)) {
      this.pendingTransitions = [latestPulse, final];
    } else {
      this.pendingTransitions = [final];
    }
  }

  updateButtons() {
    CONTROL_ACTION_META_KEYS.forEach((action) => {
      setControlButtonActive(action, this.activeActions.has(action));
    });
  }

  sameActions(left, right) {
    return left.length === right.length && left.every((item, idx) => item === right[idx]);
  }
}

const CONTROL_ACTION_META_KEYS = Object.keys(CONTROL_ACTION_META);
controlStateController = new ControlStateController();

document.addEventListener("keydown", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  if (event.repeat) return;
  controlStateController.setAction(action, true);
});

document.addEventListener("keyup", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  controlStateController.setAction(action, false);
});

window.addEventListener("blur", () => {
  controlStateController.releaseAll();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    controlStateController.releaseAll();
  }
});
