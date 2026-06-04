const $ = (id) => document.getElementById(id);
const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip";
const RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";
const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v10";
const DEFAULT_PREVIEW_OUTPUT_FORMAT = "webp";
const DEFAULT_PREVIEW_OUTPUT_QUALITY = 80;
const MAX_WEBP_PREVIEW_OUTPUT_QUALITY = 80;
const SMOOTH_PREVIEW_OUTPUT_QUALITY = 70;
const SR_PREVIEW_OUTPUT_QUALITY = 70;
const HEAVY_PREVIEW_OUTPUT_QUALITY = 60;
const DEFAULT_TARGET_FPS = 25;
const DEFAULT_FRAME_INTERPOLATION_EXP = 1;
const DEFAULT_FRAME_INTERPOLATION_SCALE = 1.0;
const DEFAULT_UPSCALING_SCALE = 2;
const DEFAULT_UPSCALING_MODEL =
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth";
const DEFAULT_PREVIEW_SCALE = 120;
const RECONNECT_CLOSE_TIMEOUT_MS = 15000;
const DECODE_QUEUE_SECONDS = 2.0;
const STARTUP_DECODE_QUEUE_SECONDS = 2.5;
const RECENT_DROP_DISPLAY_MS = 1800;
const CONTROL_BUFFERED_AMOUNT_LIMIT = 1 << 20;
const CONTROL_TRANSITION_FLUSH_DELAY_MS = 140;
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
    prompt: "A locked first-person dragon-rider view matching the reference image: both tan forearms in brown leather gloves stay visible at the bottom, gripping leather reins around the green-brown scaled dragon neck; the dragon head, horns, and both wide wings frame the jungle valley, waterfalls, mist, and tall castle on the right. Smooth forward flight only, keep the same rider hands, dragon body, wing silhouette, castle placement, and humid daylight colors in every frame.",
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
  { name: "Plastic Beach", tone: "blue", size: "832x480", fps: 25, prompt: "A static album-cover view matching the reference image: the Plastic Beach island stays centered above a dark midnight-blue ocean, the lighthouse remains on the left with its white reflection path, the starry navy sky stays unchanged, and the large white Plastic Beach title graphic stays in the lower foreground. Keep the original camera height, horizon, waterline, island silhouette, and deep blue color palette fixed; only tiny water shimmer, lighthouse glint, and subtle star twinkle, with no camera descent, no push-in, no orbit, and no turquoise color shift.", referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/b8/f9/b9/b8f9b9f8-a609-bde2-0302-349436ffc508/825646291038.jpg/600x600bb.jpg", source: "Gorillaz Plastic Beach artwork", mime: "image/jpeg" },
  { name: "Plastic Ono Band", tone: "green", size: "832x480", fps: 25, prompt: "A quiet sunlit park under a massive tree, a solitary figure resting in the grass, soft summer haze, restrained documentary camera, intimate and naturalistic.", referenceUrl: "https://upload.wikimedia.org/wikipedia/en/a/a4/JLPOBCover.jpg", source: "John Lennon/Plastic Ono Band artwork", mime: "image/jpeg" },
  { name: "Kid A", tone: "accent", size: "832x480", fps: 25, prompt: "A cold surreal mountain range with sharp icy peaks, black-red storm clouds, glacial light, slow lateral pan, abstract digital texture, uneasy atmospheric scale.", referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music122/v4/bd/8e/13/bd8e1358-b367-a689-cb84-cebd0b067dc4/634904078263.png/600x600bb.jpg", source: "Radiohead Kid A artwork", mime: "image/jpeg" },
];

const presets = [
  ...reactorPresets,
  ...examplePresets,
];

let ws = null;
let selectedPreset = null;
let selectedReferenceBytes = null;
let selectedReferenceUrl = "";
let selectedReferenceLabel = "";
let pendingHeader = null;
let frames = 0;
let bytes = 0;
let clearQueueOnClose = false;
let fpsSamples = [];
let decodeQueue = [];
let queuedDecodeFrames = 0;
let decodeInProgress = false;
let pendingDecodeBatches = 0;
let droppedDecodeFrames = 0;
let lastDecodeDropAt = 0;
let lastDecodeDropCount = 0;
let nextEventId = 1;
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
let recordingActive = false;
let recordingSamples = [];
let recordingEncoder = null;
let recordingEncoderReady = null;
let recordingEncoderConfig = null;
let recordingFrameIndex = 0;
let recordingFps = DEFAULT_TARGET_FPS;
let recordingTimer = 0;
let recordingSaving = false;
let recordingEncodeChain = Promise.resolve();
const decodeRequests = new Map();
let controlStateController = null;

const stage = document.querySelector(".stage");
const previewFrame = document.querySelector(".preview-frame");
const canvas = $("viewport");
const ctx = canvas.getContext("2d", { alpha: false });
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d", { alpha: false });
const recordingCanvas = document.createElement("canvas");
const recordingCtx = recordingCanvas.getContext("2d", { alpha: false });
const playbackController = new RealtimePlaybackController({
  targetFps: DEFAULT_TARGET_FPS,
});

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
}

function resetStreamStats() {
  pendingHeader = null;
  clearFrameQueue();
  playbackController.reset({ targetFps: previewPlaybackTargetFps() });
  frames = 0;
  bytes = 0;
  fpsSamples = [];
  clearQueueOnClose = false;
  decodeQueue = [];
  queuedDecodeFrames = 0;
  decodeInProgress = false;
  pendingDecodeBatches = 0;
  droppedDecodeFrames = 0;
  lastDecodeDropAt = 0;
  lastDecodeDropCount = 0;
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
  const useTransfer =
    isWorkerDecodableRawContentType(header.content_type) ||
    isEncodedPreviewContentType(header.content_type);
  try {
    return await new Promise((resolve, reject) => {
      decodeRequests.set(id, {
        resolve: (message) => {
          const decodedAt = performance.now();
          lastDecodeMs = decodedAt - decodeStartedAt;
          resolve(message.frames.map((frame) => ({
            image: message.frame_type === "bitmap"
              ? frame
              : new ImageData(new Uint8ClampedArray(frame), message.width, message.height),
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
    if (isEncodedPreviewContentType(header.content_type) && !useTransfer) {
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

function updateStats() {
  const playback = playbackController.snapshot();
  const queueParts = [`buffer ${formatMs(playback.bufferMs)}`];
  queueParts.push(`q ${playback.queueFrames}`);
  if (playback.buffering && playback.queueFrames) queueParts.push("hold");
  if (pendingDecodeBatches) queueParts.push(`decode ${pendingDecodeBatches}`);
  const now = performance.now();
  if (playback.lastDropAt && now - playback.lastDropAt < RECENT_DROP_DISPLAY_MS) {
    const reason = playback.lastDropReason ? ` ${playback.lastDropReason}` : "";
    queueParts.push(`drop +${playback.lastDropCount}${reason}`);
  }
  if (lastDecodeDropAt && now - lastDecodeDropAt < RECENT_DROP_DISPLAY_MS) {
    queueParts.push(`decode drop +${lastDecodeDropCount}`);
  }
  $("queueText").textContent = queueParts.join(" · ");
  $("frameText").textContent = `frames ${frames}`;
  $("byteText").textContent = `${(bytes / 1048576).toFixed(1)} MB`;
  $("stageLatencyText").textContent =
    `${formatMs(playback.bufferMs)} / ${formatMs(playback.targetLeadMs)}`;
}

function requestedInputFps() {
  return Number($("fps").value || DEFAULT_TARGET_FPS);
}

function frameInterpolationMultiplier() {
  return $("frameInterpolation").checked ? 2 ** DEFAULT_FRAME_INTERPOLATION_EXP : 1;
}

function previewPlaybackTargetFps() {
  return requestedInputFps() * frameInterpolationMultiplier();
}

function syncPlaybackTargetFps() {
  playbackController.setTargetFps(previewPlaybackTargetFps());
  updateStats();
}

function clearFrameQueue() {
  closeFrames(playbackController.clear());
}

function closeFrames(items) {
  for (const item of items || []) item.image?.close?.();
}

function recordingFileName() {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `sglang-realtime-${stamp}.mp4`;
}

function updateRecordButton() {
  const button = $("recordBtn");
  button.classList.toggle("is-recording", recordingActive);
  button.classList.toggle("is-saving", recordingSaving);
  button.disabled = recordingSaving;
  button.setAttribute("aria-pressed", recordingActive ? "true" : "false");
  $("recordLabel").textContent = recordingSaving
    ? "Saving"
    : recordingActive ? "Stop" : "Record";
  const elapsedMs = recordingActive ? recordingFrameIndex / Math.max(1, recordingFps) * 1000 : 0;
  $("recordDuration").textContent = formatRecordingDuration(elapsedMs);
}

function formatRecordingDuration(elapsedMs) {
  const seconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function startRecording() {
  if (recordingActive || recordingSaving) return;
  if (!window.VideoEncoder || !window.VideoFrame) {
    setStatus("MP4 unsupported", "error");
    addHistory("MP4 recording requires WebCodecs H.264 support");
    return;
  }
  recordingActive = true;
  recordingSamples = [];
  recordingEncoder = null;
  recordingEncoderReady = null;
  recordingEncoderConfig = null;
  recordingFrameIndex = 0;
  recordingFps = Math.max(1, previewPlaybackTargetFps());
  recordingEncodeChain = Promise.resolve();
  recordingTimer = window.setInterval(updateRecordButton, 250);
  updateRecordButton();
  addHistory("recording started");
}

async function stopRecording() {
  if (!recordingActive || recordingSaving) return;
  recordingActive = false;
  if (recordingTimer) {
    window.clearInterval(recordingTimer);
    recordingTimer = 0;
  }
  recordingSaving = true;
  updateRecordButton();

  let fileHandle = null;
  const fileName = recordingFileName();
  try {
    if (window.showSaveFilePicker) {
      fileHandle = await window.showSaveFilePicker({
        suggestedName: fileName,
        types: [{
          description: "MP4 video",
          accept: { "video/mp4": [".mp4"] },
        }],
      });
    }
    await recordingEncodeChain;
    if (!recordingEncoder || !recordingSamples.length) throw new Error("No frames were recorded");
    await recordingEncoder.flush();
    const mp4Blob = buildRecordingMp4();
    if (fileHandle) {
      const writable = await fileHandle.createWritable();
      await writable.write(mp4Blob);
      await writable.close();
    } else {
      downloadBlob(mp4Blob, fileName);
    }
    addHistory(`saved ${recordingSamples.length} frames as mp4`);
  } catch (error) {
    if (error?.name === "AbortError") {
      addHistory("recording save canceled");
    } else {
      addHistory(error.message || "recording save failed");
      setStatus("Save failed", "error");
    }
  } finally {
    recordingEncoder?.close?.();
    recordingEncoder = null;
    recordingEncoderReady = null;
    recordingSaving = false;
    recordingSamples = [];
    updateRecordButton();
  }
}

function recordDecodedFrameBatch(decodedFrames) {
  if (!recordingActive || recordingSaving) return;
  for (const item of decodedFrames) {
    if (!recordingActive) break;
    recordDecodedFrame(item.image);
  }
  updateRecordButton();
}

function recordDecodedFrame(image) {
  if (!recordingActive || recordingSaving) return;
  const frameIndex = recordingFrameIndex;
  const duration = Math.round(1_000_000 / Math.max(1, recordingFps));
  const timestamp = frameIndex * duration;
  let frame;
  try {
    frame = createRecordingFrame(image, timestamp, duration);
  } catch (error) {
    recordingActive = false;
    addHistory(error.message || "recording frame capture failed");
    updateRecordButton();
    return;
  }
  recordingFrameIndex += 1;
  recordingEncodeChain = recordingEncodeChain
    .then(async () => {
      await ensureRecordingEncoder(frame.displayWidth, frame.displayHeight);
      recordingEncoder.encode(frame, { keyFrame: frameIndex === 0 || frameIndex % 120 === 0 });
      frame.close();
    })
    .catch((error) => {
      frame.close();
      recordingActive = false;
      addHistory(error.message || "recording encode failed");
      updateRecordButton();
    });
}

function createRecordingFrame(image, timestamp, duration) {
  if (image instanceof ImageData) {
    if (recordingCanvas.width !== image.width || recordingCanvas.height !== image.height) {
      recordingCanvas.width = image.width;
      recordingCanvas.height = image.height;
    }
    recordingCtx.putImageData(image, 0, 0);
    return new VideoFrame(recordingCanvas, { timestamp, duration });
  }
  return new VideoFrame(image, { timestamp, duration });
}

async function ensureRecordingEncoder(width, height) {
  if (recordingEncoderReady) return recordingEncoderReady;
  recordingEncoderReady = createRecordingEncoder(width, height);
  return recordingEncoderReady;
}

async function createRecordingEncoder(width, height) {
  const fps = Math.max(1, recordingFps);
  const bitrate = Math.round(Math.min(
    180_000_000,
    Math.max(24_000_000, width * height * fps * 0.8),
  ));
  const configs = [
    { codec: "avc1.640028", width, height, bitrate, framerate: fps },
    { codec: "avc1.4d4028", width, height, bitrate, framerate: fps },
    { codec: "avc1.42e028", width, height, bitrate, framerate: fps },
  ];
  let supported = null;
  for (const config of configs) {
    const candidate = {
      ...config,
      avc: { format: "avc" },
      bitrateMode: "variable",
      hardwareAcceleration: "prefer-hardware",
      latencyMode: "realtime",
    };
    const result = await VideoEncoder.isConfigSupported(candidate);
    if (result.supported) {
      supported = result.config;
      break;
    }
  }
  if (!supported) throw new Error("This browser cannot encode H.264 MP4");
  recordingEncoderConfig = supported;
  recordingEncoder = new VideoEncoder({
    output: (chunk, metadata) => recordEncodedChunk(chunk, metadata),
    error: (error) => {
      recordingActive = false;
      addHistory(error.message || "recording encoder failed");
      updateRecordButton();
    },
  });
  recordingEncoder.configure(supported);
}

function recordEncodedChunk(chunk, metadata) {
  if (metadata?.decoderConfig?.description) {
    recordingEncoderConfig.description = metadata.decoderConfig.description;
  }
  const data = new Uint8Array(chunk.byteLength);
  chunk.copyTo(data);
  recordingSamples.push({
    data,
    timestamp: chunk.timestamp,
    duration: chunk.duration || 0,
    key: chunk.type === "key",
  });
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function buildRecordingMp4() {
  if (!recordingEncoderConfig.description) {
    throw new Error("H.264 encoder did not return MP4 decoder config");
  }
  const width = recordingEncoderConfig.width;
  const height = recordingEncoderConfig.height;
  const samples = normalizeRecordingSamples(recordingSamples);
  const mdatPayload = concatBytes(samples.map((sample) => sample.data));
  const ftyp = mp4Box("ftyp", ascii("isom"), u32(0x200), ascii("isom"), ascii("iso2"), ascii("avc1"), ascii("mp41"));
  const mdat = mp4Box("mdat", mdatPayload);
  const firstSampleOffset = ftyp.byteLength + 8;
  const moov = buildMoovBox({
    width,
    height,
    samples,
    firstSampleOffset,
    avcConfig: new Uint8Array(recordingEncoderConfig.description),
  });
  return new Blob([ftyp, mdat, moov], { type: "video/mp4" });
}

function normalizeRecordingSamples(samples) {
  const ordered = [...samples].sort((left, right) => left.timestamp - right.timestamp);
  const timescale = 90_000;
  const fallbackDuration = Math.round(timescale / Math.max(1, recordingFps));
  const normalized = ordered.map((sample) => ({
    ...sample,
    time: Math.round(sample.timestamp * timescale / 1_000_000),
  }));
  for (let i = 0; i < normalized.length; i++) {
    const next = normalized[i + 1];
    normalized[i].duration = next
      ? Math.max(1, next.time - normalized[i].time)
      : Math.max(1, Math.round((ordered[i].duration || 0) * timescale / 1_000_000) || fallbackDuration);
  }
  return normalized;
}

function buildMoovBox({ width, height, samples, firstSampleOffset, avcConfig }) {
  const timescale = 90_000;
  const duration = samples.reduce((sum, sample) => sum + sample.duration, 0);
  const movieTimescale = 1000;
  const movieDuration = Math.ceil(duration * movieTimescale / timescale);
  return mp4Box(
    "moov",
    buildMvhdBox(movieTimescale, movieDuration),
    mp4Box(
      "trak",
      buildTkhdBox(width, height, movieDuration),
      mp4Box(
        "mdia",
        buildMdhdBox(timescale, duration),
        buildHdlrBox(),
        mp4Box(
          "minf",
          buildVmhdBox(),
          buildDinfBox(),
          buildStblBox({ width, height, samples, firstSampleOffset, avcConfig }),
        ),
      ),
    ),
  );
}

function buildMvhdBox(timescale, duration) {
  return mp4Box(
    "mvhd",
    u32(0),
    u32(0),
    u32(0),
    u32(timescale),
    u32(duration),
    u32(0x00010000),
    u16(0x0100),
    u16(0),
    zeros(8),
    u32(0x00010000), u32(0), u32(0),
    u32(0), u32(0x00010000), u32(0),
    u32(0), u32(0), u32(0x40000000),
    zeros(24),
    u32(2),
  );
}

function buildTkhdBox(width, height, duration) {
  return mp4Box(
    "tkhd",
    u32(0x00000007),
    u32(0),
    u32(0),
    u32(1),
    u32(0),
    u32(duration),
    zeros(8),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(0x00010000), u32(0), u32(0),
    u32(0), u32(0x00010000), u32(0),
    u32(0), u32(0), u32(0x40000000),
    u32(width << 16),
    u32(height << 16),
  );
}

function buildMdhdBox(timescale, duration) {
  return mp4Box(
    "mdhd",
    u32(0),
    u32(0),
    u32(0),
    u32(timescale),
    u32(duration),
    u16(0x55c4),
    u16(0),
  );
}

function buildHdlrBox() {
  return mp4Box("hdlr", u32(0), u32(0), ascii("vide"), zeros(12), ascii("VideoHandler\0"));
}

function buildVmhdBox() {
  return mp4Box("vmhd", u32(0x00000001), u16(0), u16(0), u16(0), u16(0));
}

function buildDinfBox() {
  return mp4Box(
    "dinf",
    mp4Box(
      "dref",
      u32(0),
      u32(1),
      mp4Box("url ", u32(0x00000001)),
    ),
  );
}

function buildStblBox({ width, height, samples, firstSampleOffset, avcConfig }) {
  return mp4Box(
    "stbl",
    buildStsdBox(width, height, avcConfig),
    buildSttsBox(samples),
    buildStssBox(samples),
    buildStscBox(samples.length),
    buildStszBox(samples),
    buildStcoBox(firstSampleOffset),
  );
}

function buildStsdBox(width, height, avcConfig) {
  const compressor = new Uint8Array(32);
  return mp4Box(
    "stsd",
    u32(0),
    u32(1),
    mp4Box(
      "avc1",
      zeros(6),
      u16(1),
      zeros(16),
      u16(width),
      u16(height),
      u32(0x00480000),
      u32(0x00480000),
      u32(0),
      u16(1),
      compressor,
      u16(24),
      u16(0xffff),
      mp4Box("avcC", avcConfig),
    ),
  );
}

function buildSttsBox(samples) {
  const entries = [];
  for (const sample of samples) {
    const last = entries[entries.length - 1];
    if (last && last.duration === sample.duration) {
      last.count += 1;
    } else {
      entries.push({ count: 1, duration: sample.duration });
    }
  }
  return mp4Box("stts", u32(0), u32(entries.length), ...entries.flatMap((entry) => [u32(entry.count), u32(entry.duration)]));
}

function buildStssBox(samples) {
  const keySamples = samples
    .map((sample, index) => sample.key ? index + 1 : 0)
    .filter(Boolean);
  if (!keySamples.length && samples.length) keySamples.push(1);
  return mp4Box("stss", u32(0), u32(keySamples.length), ...keySamples.map(u32));
}

function buildStscBox(sampleCount) {
  return mp4Box("stsc", u32(0), u32(1), u32(1), u32(sampleCount), u32(1));
}

function buildStszBox(samples) {
  return mp4Box("stsz", u32(0), u32(0), u32(samples.length), ...samples.map((sample) => u32(sample.data.byteLength)));
}

function buildStcoBox(firstSampleOffset) {
  return mp4Box("stco", u32(0), u32(1), u32(firstSampleOffset));
}

function mp4Box(type, ...payloads) {
  const size = 8 + payloads.reduce((sum, payload) => sum + payload.byteLength, 0);
  const output = new Uint8Array(size);
  const view = new DataView(output.buffer);
  view.setUint32(0, size, false);
  output.set(ascii(type), 4);
  let offset = 8;
  for (const payload of payloads) {
    output.set(payload, offset);
    offset += payload.byteLength;
  }
  return output;
}

function concatBytes(parts) {
  const output = new Uint8Array(parts.reduce((sum, part) => sum + part.byteLength, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function ascii(text) {
  const output = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) output[i] = text.charCodeAt(i);
  return output;
}

function zeros(length) {
  return new Uint8Array(length);
}

function u16(value) {
  const output = new Uint8Array(2);
  new DataView(output.buffer).setUint16(0, value, false);
  return output;
}

function u32(value) {
  const output = new Uint8Array(4);
  new DataView(output.buffer).setUint32(0, value >>> 0, false);
  return output;
}

function hasPendingPlaybackInput() {
  return (
    pendingDecodeBatches > 0 ||
    decodeInProgress ||
    decodeQueue.length > 0 ||
    Boolean(ws && ws.readyState === WebSocket.OPEN)
  );
}

function enqueueDecodeBatch(header, data, epoch) {
  const frameCount = Number(header.num_frames || 1);
  decodeQueue.push({ header, data, epoch, frameCount });
  queuedDecodeFrames += frameCount;
  pendingDecodeBatches += 1;
  trimDecodeQueue();
  pumpDecodeQueue();
  updateStats();
}

function trimDecodeQueue() {
  if (recordingActive) return;
  if (!decodeQueue.length) return;
  const playback = playbackController.snapshot();
  const decodeWindowSeconds = renderedPreviewFrames
    ? Math.max(DECODE_QUEUE_SECONDS, (playback.maxLeadMs || 0) / 1000)
    : STARTUP_DECODE_QUEUE_SECONDS;
  const maxQueuedFrames = Math.max(
    2,
    Math.round(previewPlaybackTargetFps() * decodeWindowSeconds),
  );
  while (queuedDecodeFrames > maxQueuedFrames && decodeQueue.length > 1) {
    const item = decodeQueue[0];
    if (!isEncodedPreviewContentType(item.header.content_type)) break;
    decodeQueue.shift();
    queuedDecodeFrames = Math.max(0, queuedDecodeFrames - item.frameCount);
    pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
    droppedDecodeFrames += item.frameCount;
    lastDecodeDropAt = performance.now();
    lastDecodeDropCount = item.frameCount;
  }
}

async function pumpDecodeQueue() {
  if (decodeInProgress) return;
  const item = decodeQueue.shift();
  if (!item) return;
  queuedDecodeFrames = Math.max(0, queuedDecodeFrames - item.frameCount);
  decodeInProgress = true;
  try {
    await decodeAndEnqueueFrameBatch(item.header, item.data, item.epoch);
  } catch (error) {
    handleReceiveError(error, item.epoch);
  } finally {
    pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
    decodeInProgress = false;
    updateStats();
    if (decodeQueue.length) pumpDecodeQueue();
  }
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
  const framePayloads = splitEncodedPayload(header, payload);
  if (typeof createImageBitmap === "function") {
    try {
      return await Promise.all(framePayloads.map(async (framePayload) => ({
        image: await createImageBitmap(new Blob([framePayload], { type: header.content_type })),
        chunk: header.chunk_index,
      })));
    } catch (error) {
      return Promise.all(framePayloads.map((framePayload) => (
        encodedImageElementFallback(
          new Blob([framePayload], { type: header.content_type }),
          header,
          error,
        )
      )));
    }
  }
  return Promise.all(framePayloads.map((framePayload) => (
    encodedImageElementFallback(
      new Blob([framePayload], { type: header.content_type }),
      header,
      new Error("createImageBitmap unavailable"),
    )
  )));
}

function splitEncodedPayload(header, payload) {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const lengths = Array.isArray(header.payload_lengths) && header.payload_lengths.length
    ? header.payload_lengths.map(Number)
    : [bytes.byteLength];
  const payloads = [];
  let offset = 0;
  for (const length of lengths) {
    payloads.push(bytes.buffer.slice(
      bytes.byteOffset + offset,
      bytes.byteOffset + offset + length,
    ));
    offset += length;
  }
  return payloads;
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
  const decision = playbackController.render(now, {
    hasPendingInput: hasPendingPlaybackInput(),
  });
  closeFrames(decision.droppedFrames);
  if (decision.action === "draw") {
    const item = decision.frame;
    drawFrame(item.image);
    fpsSamples.push(now);
    fpsSamples = fpsSamples.filter((t) => now - t < 1000);
    const renderedFps = String(fpsSamples.length);
    $("renderFps").textContent = renderedFps;
    $("chunkText").textContent = `chunk ${item.chunk}`;
    lastDisplayLagMs = now - (item.receivedAt || now);
    $("decodeText").textContent = `${Math.round(item.decodeMs || lastDecodeMs)} ms`;
    $("displayLagText").textContent = `${(lastDisplayLagMs / 1000).toFixed(1)} s`;
    updateStats();
  } else if (decision.action === "hold") {
    updateStats();
  }
  requestAnimationFrame(renderLoop);
}

async function readFirstFrame() {
  const file = $("firstFrame").files[0];
  if (file) return new Uint8Array(await file.arrayBuffer());
  return selectedReferenceBytes || selectedReferenceUrl || undefined;
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
  selectedReferenceUrl = "";
  selectedReferenceLabel = file ? file.name : "";
  if (!file) return;
  drawReferencePreviewFromImageSource(URL.createObjectURL(file), file.name);
}

async function setPresetReference(preset) {
  selectedReferenceBytes = null;
  selectedReferenceUrl = preset.referenceUrl;
  selectedReferenceLabel = preset.source;
  $("firstFrame").value = "";
  drawReferencePreviewFromImageSource(preset.referenceUrl, selectedReferenceLabel);
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
    if (!$("firstFrame").files[0] && !selectedReferenceBytes && !selectedReferenceUrl) {
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
      enqueueDecodeBatch(message, payload, epoch);
      setStatus("Live", "live");
      return;
    }
    pendingHeader = message;
    if (pendingHeader) setStatus("Live", "live");
    return;
  }
  const header = pendingHeader;
  pendingHeader = null;
  enqueueDecodeBatch(header, data, epoch);
}

async function decodeAndEnqueueFrameBatch(header, data, epoch) {
  const chunkFrameCount = Number(header.num_frames || 0);
  const payloadBytes = data.byteLength || data.size || 0;
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
  const now = performance.now();
  // record source frames before preview playback can hold or drop for latency
  recordDecodedFrameBatch(decodedFrames);
  const enqueueResult = playbackController.enqueueDecodedFrames(header, decodedFrames, now);
  closeFrames(enqueueResult.droppedFrames);
  if (enqueueResult.cutover?.latencyMs) {
    const eventLatency = enqueueResult.cutover.latencyMs / 1000;
    $("latencyText").textContent = `${eventLatency.toFixed(1)}s · event`;
  }
  frames += chunkFrameCount;
  bytes += payloadBytes;
  $("payloadMode").textContent = header.encoding || "raw RGB";
  updateOutputSizeFromHeader(header);
  setStatus("Live", "live");
  updateStats();
}

function updateServerChunkStats(stats) {
  const rawWrite = Number(stats.raw_write_ms || 0) / 1000;
  const wsWrite = Number(stats.ws_write_ms || 0) / 1000;
  const chunkTotal = Number(stats.chunk_total_ms || 0) / 1000;
  const numFrames = Number(stats.num_frames || 0);
  const chunkIndex = Number(stats.chunk_index || 0);
  const targetFps = previewPlaybackTargetFps();
  const theoreticalFps = chunkTotal > 0 ? numFrames / chunkTotal : 0;
  const playback = playbackController.observeServerStats(stats, performance.now());
  const realtimeRatio = targetFps > 0 ? theoreticalFps / targetFps : 0;
  const isWarmupChunk =
    chunkIndex === 0 && theoreticalFps > 0 && theoreticalFps < targetFps * 0.8;
  $("serverSendText").textContent = `raw ${rawWrite.toFixed(2)}s · ws ${wsWrite.toFixed(2)}s`;
  $("chunkPayloadText").textContent = `${formatBytes(stats.ws_payload_bytes || 0)} · ${numFrames}f`;
  $("theoreticalFpsText").textContent = isWarmupChunk
    ? `warmup · ${chunkTotal.toFixed(2)}s`
    : theoreticalFps > 0
    ? `${playback.sourceFps.toFixed(1)} fps · ${realtimeRatio.toFixed(2)}x`
    : "-";
  if (chunkTotal > 0) {
    $("latencyText").textContent = `${chunkTotal.toFixed(2)}s · ${playback.sourceFps.toFixed(1)}fps`;
  }
  if (stats.content_type) $("payloadMode").textContent = shortPayloadMode(stats.content_type);
}

function sendEvent(kind, payload, historyText = null) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addHistory(`${historyText || `${kind} event`} · socket not open`);
    return null;
  }
  const eventId = nextEventId++;
  ws.send(pack({ type: "event", kind, payload, event_id: eventId }));
  if (kind === "camera_actions" || kind === "prompt") {
    playbackController.noteInputEvent(eventId, performance.now(), {
      cutoverMode: cameraActionHasActiveMotion(payload) || kind === "prompt" ? "motion" : "settle",
    });
    updateStats();
    setStatus("Updating", "live");
  }
  addHistory(`${historyText || `${kind} event sent`} · event#${eventId}`);
  return eventId;
}

function cameraActionHasActiveMotion(payload) {
  const transitions = payload?.transitions || [];
  const finalTransition = transitions[transitions.length - 1];
  return Array.isArray(finalTransition?.actions) && finalTransition.actions.length > 0;
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
  syncPlaybackTargetFps();
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
  const params = {
    realtime_output_format: outputFormat,
    realtime_output_pacing: true,
  };
  if (outputFormat === "webp" || outputFormat === "jpeg") {
    params.output_compression = outputQuality;
    if ($("superResolution").checked && $("frameInterpolation").checked) {
      const baseSize = parseSizeValue($("size").value);
      if (baseSize?.width) params.realtime_preview_max_width = baseSize.width;
    }
  }
  return params;
}

function tunePreviewQualityForPostprocess() {
  if ($("transportFormat").value !== "webp") return;
  const currentQuality = Number($("transportQuality").value || DEFAULT_PREVIEW_OUTPUT_QUALITY);
  let qualityCap = MAX_WEBP_PREVIEW_OUTPUT_QUALITY;
  if ($("frameInterpolation").checked && $("superResolution").checked) {
    qualityCap = HEAVY_PREVIEW_OUTPUT_QUALITY;
  } else if ($("frameInterpolation").checked) {
    qualityCap = SMOOTH_PREVIEW_OUTPUT_QUALITY;
  } else if ($("superResolution").checked) {
    qualityCap = SR_PREVIEW_OUTPUT_QUALITY;
  }
  if (currentQuality > qualityCap) $("transportQuality").value = String(qualityCap);
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
  const params = {
    enable_upscaling: true,
    upscaling_scale: readUpscalingScale(),
  };
  const modelPath = $("upscalingModel").value;
  if (modelPath) params.upscaling_model_path = modelPath;
  return params;
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
  const width = Number(header.source_width || header.width || 0);
  const height = Number(header.source_height || header.height || 0);
  if (!width || !height) return;
  updateOutputSizeText(width, height);
  if (header.preview_width && header.preview_height) {
    $("outputSizeText").textContent += ` · preview ${header.preview_width}x${header.preview_height}`;
  }
}

function updateSuperResolutionControls() {
  const disabled = !$("superResolution").checked;
  $("upscalingScale").disabled = disabled;
  $("upscalingModel").disabled = disabled;
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

function formatMs(value) {
  const ms = Number(value || 0);
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
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
  const server = params.get("server");
  if (server) $("serverUrl").value = server;
  const model = params.get("model");
  if (model) $("model").value = model;
  $("transportFormat").value = params.get("transport") || DEFAULT_PREVIEW_OUTPUT_FORMAT;
  $("transportQuality").value = params.get("quality") || String(DEFAULT_PREVIEW_OUTPUT_QUALITY);
  const srParam = params.get("sr");
  $("superResolution").checked = srParam === "1" || srParam === "true";
  const smoothParam = params.get("smooth");
  $("frameInterpolation").checked = smoothParam === "1" || smoothParam === "true";
  $("upscalingScale").value = params.get("sr_scale") || String(DEFAULT_UPSCALING_SCALE);
  $("upscalingModel").value = params.get("sr_model") || DEFAULT_UPSCALING_MODEL;
  tunePreviewQualityForPostprocess();
  setPreviewScale(params.get("preview_scale") || params.get("zoom"));
  updateSuperResolutionControls();
  syncPlaybackTargetFps();

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
    if (b === 0xdc) return Array.from({ length: (buf[i++] << 8) | buf[i++] }, read);
    if (b === 0xdd) {
      return Array.from({
        length: (buf[i++] * 16777216) + (buf[i++] << 16) + (buf[i++] << 8) + buf[i++],
      }, read);
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
applyQueryParams()
  .then(async (query) => {
    if (!query.preset) await applyPreset(presets[0], { sendRuntimeEvents: false });
    return query;
  })
  .then((query) => queryServerModelInfo({
    applyPresetForModel: !query.model && !query.preset,
  }))
  .catch(showError);
requestAnimationFrame(renderLoop);
updateRecordButton();
$("connectBtn").onclick = connect;
$("stopBtn").onclick = () => closeSession();
$("sendPromptBtn").onclick = () => sendEvent("prompt", $("prompt").value);
$("enhanceBtn").onclick = enhancePrompt;
$("recordBtn").onclick = () => {
  if (recordingActive) {
    stopRecording();
  } else {
    startRecording();
  }
};
$("firstFrame").onchange = () => drawReferencePreview($("firstFrame").files[0]);
$("size").addEventListener("input", () => updateOutputSizeText());
$("fps").addEventListener("input", syncPlaybackTargetFps);
$("superResolution").addEventListener("change", updateSuperResolutionControls);
$("upscalingScale").addEventListener("change", () => updateOutputSizeText());
$("frameInterpolation").addEventListener("change", () => {
  tunePreviewQualityForPostprocess();
  syncPlaybackTargetFps();
});
$("superResolution").addEventListener("change", tunePreviewQualityForPostprocess);
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
    this.flushTimer = 0;
  }

  reset({ sendRelease = false } = {}) {
    const hadActions = this.activeActions.size > 0;
    this.activeActions.clear();
    this.pendingTransitions = [];
    this.clearFlushTimer();
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
    if (this.flushTimer) return;
    this.flushTimer = window.setTimeout(() => {
      this.flushTimer = 0;
      this.flush();
    }, CONTROL_TRANSITION_FLUSH_DELAY_MS);
  }

  flush() {
    this.clearFlushTimer();
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

  clearFlushTimer() {
    if (!this.flushTimer) return;
    window.clearTimeout(this.flushTimer);
    this.flushTimer = 0;
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
