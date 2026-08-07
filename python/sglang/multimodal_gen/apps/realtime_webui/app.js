const $ = (id) => document.getElementById(id);
const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip";
const RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";
const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v10";
const UI_CONFIG = Object.freeze(globalThis.SGLANG_REALTIME_UI_CONFIG || {});
const SESSION_ARTIFACT_SCHEMA_VERSION = 1;
const SESSION_ARTIFACT_EVENT_LIMIT = 20000;
const MAX_EMBEDDED_REFERENCE_IMAGE_BYTES = 2 * 1024 * 1024;

function configuredNumber(name, fallback) {
  const value = Number(UI_CONFIG[name]);
  return Number.isFinite(value) ? value : fallback;
}

function configuredGenerationModes() {
  const requestedModes = Array.isArray(UI_CONFIG.generationModes)
    ? UI_CONFIG.generationModes
    : UI_CONFIG.generationMode || UI_CONFIG.defaultGenerationMode
    ? ["i2v", UI_CONFIG.generationMode || UI_CONFIG.defaultGenerationMode]
    : ["i2v"];
  const modes = requestedModes
    .map((mode) => String(mode).toLowerCase())
    .filter((mode, index, values) => (
      (mode === "i2v" || mode === "t2v") && values.indexOf(mode) === index
    ));
  return modes.length ? modes : ["i2v"];
}

const DEFAULT_PREVIEW_OUTPUT_FORMAT = "webp";
const DEFAULT_PREVIEW_OUTPUT_QUALITY = 55;
const MAX_WEBP_PREVIEW_OUTPUT_QUALITY = 80;
const SMOOTH_PREVIEW_OUTPUT_QUALITY = 70;
const SR_PREVIEW_OUTPUT_QUALITY = 70;
const HEAVY_PREVIEW_OUTPUT_QUALITY = 60;
const DEFAULT_TARGET_FPS = configuredNumber("targetFps", 24);
const DEFAULT_PREVIEW_MAX_WIDTH = configuredNumber("previewMaxWidth", 832);
const MAX_AUTO_PREVIEW_WIDTH = configuredNumber("maxAutoPreviewWidth", 1280);
const DEFAULT_FRAME_INTERPOLATION_EXP = 1;
const DEFAULT_FRAME_INTERPOLATION_SCALE = 1.0;
const DEFAULT_UPSCALING_SCALE = 2;
const DEFAULT_UPSCALING_MODEL =
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth";
const DEFAULT_PREVIEW_SCALE = 100;
const ENABLED_GENERATION_MODES = configuredGenerationModes();
const CONFIGURED_DEFAULT_GENERATION_MODE = String(
  UI_CONFIG.defaultGenerationMode || UI_CONFIG.generationMode || "",
).toLowerCase();
const DEFAULT_GENERATION_MODE = ENABLED_GENERATION_MODES.includes(
  CONFIGURED_DEFAULT_GENERATION_MODE,
)
  ? CONFIGURED_DEFAULT_GENERATION_MODE
  : ENABLED_GENERATION_MODES[0];
const T2V_FRAME_STEP = Math.max(
  1,
  Math.trunc(configuredNumber("t2vFrameStep", 4)),
);
const DEFAULT_T2V_NUM_FRAMES = 9;
const RECONNECT_CLOSE_TIMEOUT_MS = 15000;
const DECODE_QUEUE_SECONDS = 0.5;
const STARTUP_DECODE_QUEUE_SECONDS = 0.75;
const RECENT_DROP_DISPLAY_MS = 1800;
const CONTROL_BUFFERED_AMOUNT_LIMIT = 1 << 20;
const CONTROL_TRANSITION_FLUSH_DELAY_MS = 50;
const CONTROL_HELD_STATE_HEARTBEAT_MS = 100;
const SESSION_HEARTBEAT_MS = 15000;
const BROWSER_USER_ID_STORAGE_KEY = "sglang-realtime-user-id";
const MIN_RENDER_TIMER_FPS = 30;
const MAX_RENDER_TIMER_FPS = 60;
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
const RECORDING_STAGE_WIDTH = 1600;
const RECORDING_STAGE_TOPBAR_HEIGHT = 54;
const RECORDING_STAGE_PREVIEW_HEIGHT = 586;
const RECORDING_STAGE_CONTROLS_HEIGHT = 144;
const RECORDING_STAGE_TIMELINE_HEIGHT = 48;
const RECORDING_STAGE_TELEMETRY_HEIGHT = 96;
const RECORDING_STAGE_HEIGHT =
  RECORDING_STAGE_TOPBAR_HEIGHT +
  RECORDING_STAGE_PREVIEW_HEIGHT +
  RECORDING_STAGE_CONTROLS_HEIGHT +
  RECORDING_STAGE_TIMELINE_HEIGHT +
  RECORDING_STAGE_TELEMETRY_HEIGHT;
const RECORDING_STAGE_PADDING = 18;

function applyRuntimeUiConfig() {
  $("fps").value = String(DEFAULT_TARGET_FPS);
  $("guidance").value = String(
    configuredNumber("guidanceScale", Number($("guidance").value)),
  );
  $("sinkSize").value = String(
    configuredNumber("sinkSize", Number($("sinkSize").value)),
  );
  $("windowFrames").value = String(
    configuredNumber("windowFrames", Number($("windowFrames").value)),
  );
  $("targetFpsSummary").textContent = `${DEFAULT_TARGET_FPS} fps`;
  if (UI_CONFIG.modelLabel) {
    $("modelSectionTitle").textContent = String(UI_CONFIG.modelLabel);
  }
  if (UI_CONFIG.titleSuffix) {
    const suffix = String(UI_CONFIG.titleSuffix);
    $("studioTitle").textContent = `Realtime Studio · ${suffix}`;
    document.title = `Realtime Studio · ${suffix}`;
  }
  if (UI_CONFIG.actionAmountLabel) {
    Object.values(CONTROL_ACTION_META).forEach((meta) => {
      meta.amount = String(UI_CONFIG.actionAmountLabel);
    });
  }
  configureGenerationModeSelect();
}

function configureGenerationModeSelect() {
  const select = $("generationMode");
  Array.from(select.options).forEach((option) => {
    const enabled = ENABLED_GENERATION_MODES.includes(option.value);
    option.disabled = !enabled;
    option.hidden = !enabled;
  });
  select.value = DEFAULT_GENERATION_MODE;
  $("generationModeField").hidden = ENABLED_GENERATION_MODES.length < 2;
  updateGenerationModeUi();
}

function selectedGenerationMode() {
  return $("generationMode").value === "t2v" ? "t2v" : "i2v";
}

function updateT2VFrameHint() {
  if (selectedGenerationMode() === "t2v" && $("continuous").checked) {
    $("t2vFrameHint").textContent = "Continuous T2V runs until Stop is pressed.";
    return;
  }
  const frames = Number($("numFrames").value);
  const fps = Number($("fps").value || DEFAULT_TARGET_FPS);
  const duration = Number.isFinite(frames) && Number.isFinite(fps) && fps > 0
    ? Math.max(0, frames) / fps
    : 0;
  $("t2vFrameHint").textContent = (
    `MinWM requires 1 + N × ${T2V_FRAME_STEP}; `
    + `${frames || 0} frames ≈ ${duration.toFixed(2)}s at ${fps || 0}fps.`
  );
}

function updateGenerationModeUi() {
  const mode = selectedGenerationMode();
  const isT2V = mode === "t2v";
  if (lastGenerationMode !== mode) {
    if (isT2V) {
      savedI2VNumFrames = $("numFrames").value;
      savedI2VContinuous = $("continuous").checked;
      $("numFrames").value = savedT2VNumFrames;
      $("continuous").checked = savedT2VContinuous;
    } else if (lastGenerationMode === "t2v") {
      savedT2VNumFrames = $("numFrames").value;
      savedT2VContinuous = $("continuous").checked;
      $("numFrames").value = savedI2VNumFrames;
      $("continuous").checked = savedI2VContinuous;
    }
  }
  $("referenceSection").hidden = isT2V;
  $("t2vFrameHint").hidden = !isT2V;
  $("numFrames").min = isT2V ? "1" : "5";
  $("numFrames").step = isT2V ? String(T2V_FRAME_STEP) : "4";
  $("continuous").disabled = false;
  $("numFrames").disabled = isT2V && $("continuous").checked;
  $("continuousLabelText").textContent = isT2V
    ? "Continuous T2V session"
    : "Continuous session";
  lastGenerationMode = mode;
  updateT2VFrameHint();
}

function readT2VNumFrames() {
  const numFrames = Number($("numFrames").value);
  if (
    !Number.isInteger(numFrames)
    || numFrames < 1
    || (numFrames - 1) % T2V_FRAME_STEP !== 0
  ) {
    throw new Error(
      `MinWM T2V Frames must equal 1 + N × ${T2V_FRAME_STEP}`,
    );
  }
  return numFrames;
}

const REACTOR_PRESET_BASE_URL = "https://www.reactor.inc/lingbot-world-fast-v1";

const reactorPresets = [
  {
    name: "Dragon Ride",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A locked first-person dragon-rider view matching the reference image: both tan forearms in brown leather gloves stay visible at the bottom, gripping leather reins around the green-brown scaled dragon neck; the dragon head, horns, and both wide wings frame the jungle valley, waterfalls, mist, and tall castle on the right. Smooth forward flight only, keep the same rider hands, dragon body, wing silhouette, castle placement, and humid daylight colors in every frame.",
    referenceUrl: "./assets/dragon-ride.jpg",
    source: "Reactor LingBot preset",
  },
  {
    name: "Misted Kingdom",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person over-the-shoulder fantasy view following a sword-slung rider on a brown horse through curling valley mist, wildflower meadows, ruined stone arches, cottages, and a many-spired castle under a ringed gas giant and crescent moon.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/misted-kingdom.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Storm Crossing",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person stern view of a battered grey aluminum work boat pushing through slate-black storm swells, wet wooden deck, warm cabin lamp, orange life rings, salt mist, churning wake, and a pale silver break in the dark horizon.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/storm-crossing.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Citadel Approach",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person rear view of a mud-streaked vintage Defender 4x4 driving along a cobblestone-and-sand track through a coral-lit desert canyon toward a cliff-built sandstone citadel, with cacti, red poppies, ochre dunes, and peach sunset haze.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/citadel-approach.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Spring Valley",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person over-the-shoulder view following a golden retriever through a sunlit meadow with a patterned floral rug, stone bench, open book, potted seedling, cherry blossoms, rounded green oaks, soft hills, and a tender watercolor storybook atmosphere.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/spring-valley.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Reef Patrol",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person follow view trailing a large grey reef shark through clear tropical water above a sunlit coral reef, with drifting sediment, shifting sun-ray lattices, clouds of reef fish, a sardine bait ball, and deep blue open-water haze.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/reef-patrol.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Alpine Run",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person rear view of a yellow four-person whitewater raft plunging through churning rapids in an alpine canyon, red lifejackets, yellow helmets, wet paddles, dark boulders, conifer slopes, and a snow-capped mountain at the vanishing point.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/alpine-run.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Ice Kayak",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered elevated third-person game camera behind a lone kayaker in a bright red kayak crossing a calm deep blue alpine lake, scattered ice blocks, mirror reflections, huge snow-covered mountain ranges, vivid sky, and crisp cold wilderness scale.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/ice-kayak.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Penguin Colony",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person follow view of a single black-and-white penguin waddling across a windswept Antarctic ice shelf toward a distant colony, crystalline snow, small flippers, scattered dark boulders, rocky shoreline, and pale polar sky.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/penguin.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Mars Mountain",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered third-person rear view of a six-wheeled Martian rover marked XR-7A P-3317 crossing cracked basalt toward a vast volcanic mountain, dusty rose twilight, ochre wheel plumes, weathered grey panels, and a cold alien horizon.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/mars-rover.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Seaside Adventurer",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered third-person anime view behind a young girl on a flower-covered coastal hillside overlooking a sparkling blue bay, rolling green hills, sailboats, dramatic cliffs, a small lighthouse, huge fluffy clouds, and warm hand-painted adventure atmosphere.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/anime3.png`,
    source: "Reactor LingBot preset",
    mime: "image/png",
  },
  {
    name: "Roman Chariot",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered elevated third-person game camera behind a Roman warrior riding an ancient chariot pulled by two white horses across an open grassy field, worn stone path, Roman ruins, broken columns, bright midday sky, and epic historical scale.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/chariot.png`,
    source: "Reactor LingBot preset",
    mime: "image/png",
  },
  {
    name: "Asylum Corridor",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person over-the-shoulder traversal behind a man in a wet leather jacket holding a flashlight down a derelict asylum corridor, standing water, torn vinyl strips, rusted ceiling debris, bloodstains, a toppled wheelchair, and a distant cyan-grey doorway glow.",
    referenceUrl: `${REACTOR_PRESET_BASE_URL}/horror.jpg`,
    source: "Reactor LingBot preset",
  },
];

const examplePresets = [
  { name: "Dragon Dolly", tone: "green", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A stable first-person dolly from the same dragon-rider viewpoint, keeping the black dragon head, horns, wings, jungle canopy, and distant castle consistent; slow forward camera motion, natural parallax, no creature morphing, no scene replacement.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/00/image.jpg", source: "LingBot example 00" },
  { name: "Stone Orbit", tone: "blue", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A controlled look-around of the stone monument, overcast daylight, consistent geometry, subtle camera arc.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/01/image.jpg", source: "LingBot example 01" },
  { name: "Urban Tilt", tone: "accent", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A cinematic urban wall shot with a slow tilt and slight forward movement, warm backlight, stable architecture.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/02/image.jpg", source: "LingBot example 02" },
  { name: "Lake Scout", tone: "green", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A calm scouting shot across the lake, gentle camera drift, crisp mountains, stable reflections.", referenceUrl: "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/03/image.jpg", source: "LingBot example 03" },
  { name: "Ziggy Stardust", tone: "accent", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A static night view of a narrow London alley in soft rain, wet pavement reflecting a yellow streetlamp, the blue K. West sign glowing above a doorway, cardboard boxes near the wall, a pale parked car in the distance, and a slender glam-rock figure holding a guitar under the lamp; preserve the album-cover composition, brick storefronts, muted teal and amber colors, subtle rain shimmer only.", referenceUrl: "https://upload.wikimedia.org/wikipedia/en/0/01/ZiggyStardust.jpg", source: "David Bowie Ziggy Stardust artwork", mime: "image/jpeg" },
  { name: "Plastic Beach", tone: "blue", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A static album-cover view matching the reference image: the Plastic Beach island stays centered above a dark midnight-blue ocean, the lighthouse remains on the left with its white reflection path, the starry navy sky stays unchanged, and the large white Plastic Beach title graphic stays in the lower foreground. Keep the original camera height, horizon, waterline, island silhouette, and deep blue color palette fixed; only tiny water shimmer, lighthouse glint, and subtle star twinkle, with no camera descent, no push-in, no orbit, and no turquoise color shift.", referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/b8/f9/b9/b8f9b9f8-a609-bde2-0302-349436ffc508/825646291038.jpg/600x600bb.jpg", source: "Gorillaz Plastic Beach artwork", mime: "image/jpeg" },
  { name: "Plastic Ono Band", tone: "green", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A quiet sunlit park under a massive tree, a solitary figure resting in the grass, soft summer haze, restrained documentary camera, intimate and naturalistic.", referenceUrl: "https://upload.wikimedia.org/wikipedia/en/a/a4/JLPOBCover.jpg", source: "John Lennon/Plastic Ono Band artwork", mime: "image/jpeg" },
  { name: "Kid A", tone: "accent", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A cold surreal mountain range with sharp icy peaks, black-red storm clouds, glacial light, slow lateral pan, abstract digital texture, uneasy atmospheric scale.", referenceUrl: "https://is1-ssl.mzstatic.com/image/thumb/Music122/v4/bd/8e/13/bd8e1358-b367-a689-cb84-cebd0b067dc4/634904078263.png/600x600bb.jpg", source: "Radiohead Kid A artwork", mime: "image/jpeg" },
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
let lastGenerationMode = null;
let savedI2VNumFrames = "9";
let savedT2VNumFrames = String(DEFAULT_T2V_NUM_FRAMES);
let savedI2VContinuous = true;
let savedT2VContinuous = true;
let pendingHeader = null;
let frames = 0;
let bytes = 0;
let clearQueueOnClose = false;
let fpsSamples = [];
let renderLoopSamples = [];
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
let recordingMode = "";
let recordingMediaRecorder = null;
let recordingMediaChunks = [];
let recordingCaptureStream = null;
let recordingMimeType = "video/mp4";
let recordingDirectoryHandle = null;
let recordingBaseFileName = "";
let currentSessionArtifact = null;
let recordingArtifact = null;
let currentTrace = null;
let renderedTraceChunks = new Set();
const decodeRequests = new Map();
let controlStateController = null;
let lastSentEventId = 0;
let lastSampledEventId = 0;
const traceTopologyApi = window.SGLangRealtimeTraceTopology || {};
const traceTopology = traceTopologyApi.createRealtimeTraceTopology
  ? traceTopologyApi.createRealtimeTraceTopology({ maxEvents: 220 })
  : null;
const traceTransportApi = window.SGLangRealtimeTraceTransport || {};
const traceHttpClient = traceTransportApi.RealtimeTraceHttpClient
  ? new traceTransportApi.RealtimeTraceHttpClient({
      onServerEvents: (events) => {
        events.forEach((event) => recordTraceTopologyEvent(event));
      },
      onAggregate: (aggregate) => {
        const metricsChanged = traceTopology?.setAggregate?.(aggregate);
        if (metricsChanged) renderTraceTopology();
        else if (traceTopology) updateTraceSummary(traceTopology.summary());
      },
    })
  : null;
const formatTraceDuration = traceTopologyApi.formatTraceDuration || formatMs;
let activeWorkspaceView = "preview";
let traceRenderFrame = 0;

const stage = document.querySelector(".stage");
const previewFrame = document.querySelector(".preview-frame");
const canvas = $("viewport");
const ctx = canvas.getContext("2d", { alpha: false });
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d", { alpha: false });
const recordingCanvas = document.createElement("canvas");
const recordingCtx = recordingCanvas.getContext("2d", { alpha: false });
const playbackController = new RealtimePlaybackController({
  mode: "live",
  targetFps: DEFAULT_TARGET_FPS,
  lowLatencyPlayback: true,
  holdForTargetLead: true,
  targetLeadChunkRatio: 0.75,
  minTargetLeadMs: 360,
  maxTargetLeadMs: 900,
  lowLatencyMaxLeadFrames: 12,
  startLeadChunkRatio: 0.55,
  minStartLeadMs: 260,
  resumeLeadChunkRatio: 0.55,
  minResumeLeadMs: 260,
  maxResumeLeadMs: 900,
  maxDeliveryLeadBoostMs: 360,
  deliveryStallExpectedMultiplier: 1.8,
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
  const now = new Date();
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  item.textContent = `${now.toLocaleTimeString("zh-CN", { hour12: false })}.${ms} ${text}`;
  $("historyList").prepend(item);
  while ($("historyList").children.length > 8) $("historyList").lastChild.remove();
}

function createClientTrace() {
  return {
    traceId: createTraceId(),
    seq: 0,
    createdPerfMs: performance.now(),
    createdEpochMs: Date.now(),
    events: [],
  };
}

function createTraceId() {
  if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
  const random = crypto.getRandomValues(new Uint32Array(4));
  return Array.from(random, (part) => part.toString(16).padStart(8, "0")).join("");
}

function stableBrowserUserId() {
  try {
    let value = localStorage.getItem(BROWSER_USER_ID_STORAGE_KEY);
    if (!value) {
      value = createTraceId();
      localStorage.setItem(BROWSER_USER_ID_STORAGE_KEY, value);
    }
    return value;
  } catch {
    return createTraceId();
  }
}

const browserUserId = stableBrowserUserId();

function traceWebSocketUrl(baseUrl) {
  try {
    const url = new URL(baseUrl, window.location.href);
    if (currentTrace) url.searchParams.set("trace_id", currentTrace.traceId);
    url.searchParams.set("user_id", browserUserId);
    return url.toString();
  } catch {
    const separator = baseUrl.includes("?") ? "&" : "?";
    const trace = currentTrace
      ? `trace_id=${encodeURIComponent(currentTrace.traceId)}&`
      : "";
    return `${baseUrl}${separator}${trace}user_id=${encodeURIComponent(browserUserId)}`;
  }
}

function markClientTrace(name, fields = {}, options = {}) {
  if (!currentTrace) return null;
  const event = {
    name,
    seq: ++currentTrace.seq,
    trace_id: currentTrace.traceId,
    client_perf_ms: roundTraceNumber(performance.now()),
    client_epoch_ms: Date.now(),
    ...fields,
  };
  currentTrace.events.push(event);
  if (currentTrace.events.length > 64) currentTrace.events.shift();
  recordTraceTopologyEvent(event);
  if (options.send !== false) traceHttpClient?.enqueueClientEvent(event);
  return event;
}

function roundTraceNumber(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}

function recordTraceTopologyEvent(event, receivedPerfMs = performance.now()) {
  if (!traceTopology || !event) return;
  const traceEvent = event.trace ? event.trace : event;
  traceTopology.addEvent(traceEvent, receivedPerfMs);
  recordTrajectoryEvent("trace_event", { trace: traceEvent });
  renderTraceTopology();
}

function resetTraceTopology(traceId = "") {
  traceTopology?.reset(traceId);
  renderTraceTopology();
}

function renderTraceTopology() {
  if (traceRenderFrame) return;
  traceRenderFrame = requestAnimationFrame(() => {
    traceRenderFrame = 0;
    renderTraceTopologyNow();
  });
}

function renderTraceTopologyNow() {
  if (!traceTopology) return;
  const summary = traceTopology.summary();
  updateTraceSummary(summary);
  if (activeWorkspaceView !== "trace") return;
  renderTraceSvg(summary);
  renderTraceEventList(summary.recentEvents);
}

function updateTraceSummary(summary) {
  $("traceIdText").textContent = shortTraceId(summary.traceId);
  $("traceEventCountText").textContent = String(summary.eventCount);
  const aggregate = summary.aggregate;
  const observedLabel = aggregate
    ? `${aggregate.window?.seconds || 300}s · ${aggregate.stale ? "stale" : "fresh"} · ${aggregate.observed_at || "-"}`
    : "-";
  $("traceObservedText").textContent = observedLabel;
  const chunk = summary.latestChunk;
  $("traceChunkText").textContent = chunk ? `#${chunk.chunkIndex}` : "-";
  $("traceChunkTotalText").textContent = chunk ? formatTraceDuration(chunk.chunkTotalMs) : "-";
  $("traceSchedulerText").textContent = traceStageMetric(summary, "scheduler", chunk?.schedulerForwardMs);
  $("traceVaeEncodeText").textContent = traceStageMetric(summary, "vae_encode", chunk?.vaeEncodeMs);
  $("traceDenoiseText").textContent = traceStageMetric(summary, "denoise", chunk?.denoiseMs);
  const vaeDecodeMs = chunk
    ? sumTraceNumbers(chunk.vaeDecodeMs, chunk.postDecodeMs)
    : null;
  $("traceVaeDecodeText").textContent = traceStageMetric(summary, "vae_decode", vaeDecodeMs);
  $("traceAsyncEstimateText").textContent = formatAsyncEstimate(summary.asyncEstimate);
}

function traceStageMetric(summary, stageId, fallbackMs) {
  const stage = summary.aggregate?.stages?.find((candidate) => candidate.id === stageId);
  if (stage && Number(stage.count || 0) > 0) {
    return `p50 ${formatTraceDuration(stage.p50_ms)} · p95 ${formatTraceDuration(stage.p95_ms)}`;
  }
  return formatTraceDuration(fallbackMs);
}

function renderTraceSvg(summary) {
  const container = $("traceTopology");
  const nodes = summary.nodes || [];
  if (!nodes.length || summary.eventCount === 0) {
    container.innerHTML = `<svg viewBox="0 0 1180 240" role="img" aria-label="Trace topology"><text class="trace-empty" x="36" y="122">Trace events will appear after Generate starts.</text></svg>`;
    return;
  }

  const width = 1180;
  const height = 240;
  const marginX = 28;
  const nodeW = nodes.length > 8 ? 112 : 124;
  const nodeH = 74;
  const gap = (width - marginX * 2 - nodeW * nodes.length) / Math.max(1, nodes.length - 1);
  const nodeY = 72;
  const positions = new Map();
  nodes.forEach((node, index) => {
    positions.set(node.id, {
      x: marginX + index * (nodeW + gap),
      y: nodeY,
    });
  });

  const edges = (summary.edges || []).map((edge) => {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (!from || !to) return "";
    const x1 = from.x + nodeW;
    const x2 = to.x;
    const y = nodeY + nodeH / 2;
    return `
      <line class="trace-edge-line" x1="${x1}" y1="${y}" x2="${x2 - 8}" y2="${y}" />
      <text class="trace-edge-label" x="${(x1 + x2) / 2}" y="${y - 10}" text-anchor="middle">${escapeHtml(nodeLabel(edge.label || "-"))}</text>
    `;
  }).join("");

  const nodeMarkup = nodes.map((node) => {
    const pos = positions.get(node.id);
    return `
      <g class="trace-node ${node.status === "active" ? "is-active" : ""}" transform="translate(${pos.x} ${pos.y})">
        <rect width="${nodeW}" height="${nodeH}" rx="8"></rect>
        <text class="trace-node-title" x="12" y="24">${escapeHtml(node.title)}</text>
        <text class="trace-node-subtitle" x="12" y="43">${escapeHtml(node.subtitle || "")}</text>
        <text class="trace-node-metric" x="12" y="62">${escapeHtml(node.metric || "-")}</text>
      </g>
    `;
  }).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Realtime trace topology">
      <defs>
        <marker id="traceArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8c9288"></path>
        </marker>
      </defs>
      ${edges}
      ${nodeMarkup}
    </svg>
  `;
}

function renderTraceEventList(events) {
  const list = $("traceEventList");
  list.replaceChildren();
  for (const event of [...events].reverse()) {
    const item = document.createElement("div");
    item.className = "trace-event-item";
    const name = document.createElement("b");
    name.textContent = event.event || event.name || "-";
    const time = document.createElement("span");
    time.textContent = traceEventTimeLabel(event);
    const details = document.createElement("code");
    details.textContent = traceEventDetails(event);
    item.append(name, time, details);
    list.appendChild(item);
  }
}

function traceEventTimeLabel(event) {
  if (Number.isFinite(Number(event.server_elapsed_ms))) {
    return `server +${formatTraceDuration(event.server_elapsed_ms)}`;
  }
  if (Number.isFinite(Number(event.client_perf_ms)) && currentTrace) {
    return `client +${formatTraceDuration(Number(event.client_perf_ms) - currentTrace.createdPerfMs)}`;
  }
  return "-";
}

function traceEventDetails(event) {
  const parts = [];
  if (event.chunk_index !== null && event.chunk_index !== undefined) parts.push(`chunk=${event.chunk_index}`);
  if (event.event_id !== null && event.event_id !== undefined) parts.push(`event=${event.event_id}`);
  if (Number.isFinite(Number(event.duration_ms))) parts.push(`duration=${formatTraceDuration(event.duration_ms)}`);
  if (Number.isFinite(Number(event.cuda_ms))) parts.push(`cuda=${formatTraceDuration(event.cuda_ms)}`);
  if (Number.isFinite(Number(event.chunk_total_ms))) parts.push(`chunk_total=${formatTraceDuration(event.chunk_total_ms)}`);
  if (Number.isFinite(Number(event.display_lag_ms))) parts.push(`display_lag=${formatTraceDuration(event.display_lag_ms)}`);
  if (event.content_type) parts.push(shortPayloadMode(event.content_type));
  return parts.join(" · ") || "-";
}

function formatAsyncEstimate(estimate) {
  if (!estimate) return "-";
  return `${formatTraceDuration(estimate.savedMs)} saved · ${estimate.speedup.toFixed(2)}x`;
}

function sumTraceNumbers(...values) {
  let total = 0;
  let seen = false;
  for (const value of values) {
    if (!Number.isFinite(Number(value))) continue;
    total += Number(value);
    seen = true;
  }
  return seen ? total : null;
}

function shortTraceId(traceId) {
  const value = String(traceId || "");
  if (!value) return "-";
  if (value.length <= 12) return value;
  return `${value.slice(0, 8)}...${value.slice(-4)}`;
}

function nodeLabel(value) {
  return value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setWorkspaceView(view) {
  activeWorkspaceView = view === "trace" ? "trace" : "preview";
  document.querySelectorAll("[data-workspace-view]").forEach((button) => {
    const active = button.dataset.workspaceView === activeWorkspaceView;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".workspace-pane").forEach((pane) => {
    const active = pane.id === `${activeWorkspaceView}Pane`;
    pane.classList.toggle("is-active", active);
    pane.hidden = !active;
  });
  if (activeWorkspaceView === "trace") {
    renderTraceTopologyNow();
    traceHttpClient?.setActive(true, 5000);
  } else {
    traceHttpClient?.setActive(false);
  }
}

function updateControlDebugText() {
  const activeActions = controlStateController
    ? Array.from(controlStateController.activeActions).sort().join("+")
    : "";
  const activeText = activeActions || "idle";
  const sentText = lastSentEventId ? `sent #${lastSentEventId}` : "sent -";
  const sampledText = lastSampledEventId ? `sampled #${lastSampledEventId}` : "sampled -";
  $("actionStateText").textContent = `${activeText} · ${sentText} · ${sampledText}`;
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
  playbackController.reset({
    mode: selectedPlaybackMode(),
    targetFps: previewPlaybackTargetFps(),
  });
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
  lastSentEventId = 0;
  lastSampledEventId = 0;
  renderedTraceChunks = new Set();
  controlStateController?.reset({ sendRelease: false });
  resetDecoderState();
  updateStats();
  $("renderFps").textContent = "0";
  $("latencyText").textContent = "-";
  $("stageLatencyText").textContent = "-";
  $("actionStateText").textContent = "-";
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
  return isWorkerDecodableRawContentType(contentType);
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
  const queueParts = [
    playback.mode === "timeline" ? "full timeline" : "low latency",
    `buffer ${formatMs(playback.bufferMs)}`,
  ];
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
  const totalDroppedFrames = playback.droppedFrames + droppedDecodeFrames;
  if (totalDroppedFrames) queueParts.push(`dropped ${totalDroppedFrames} total`);
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

function selectedPlaybackMode() {
  return $("playbackMode")?.value === "timeline" ? "timeline" : "live";
}

function syncPlaybackMode({ addToHistory = true } = {}) {
  const mode = selectedPlaybackMode();
  playbackController.setMode(mode);
  if (addToHistory) {
    addHistory(
      mode === "timeline"
        ? "playback · full timeline (no frame skipping)"
        : "playback · low latency (may skip old frames)",
    );
  }
  trimDecodeQueue();
  updateStats();
}

function clearFrameQueue() {
  closeFrames(playbackController.clear());
}

function closeFrames(items) {
  for (const item of items || []) item.image?.close?.();
}

function recordingFileName(extension = "mp4") {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `sglang-realtime-${stamp}.${extension}`;
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

function selectRecordingMode() {
  if (window.VideoEncoder && window.VideoFrame) return "webcodecs-mp4";
  if (window.MediaRecorder && recordingCanvas.captureStream && supportedWebmMimeType()) {
    return "mediarecorder-webm";
  }
  return "";
}

function supportedWebmMimeType() {
  if (!window.MediaRecorder) return "";
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return candidates.find((mimeType) => (
    typeof MediaRecorder.isTypeSupported !== "function" ||
    MediaRecorder.isTypeSupported(mimeType)
  )) || "";
}

function updateRecordFolderButton() {
  const button = $("recordFolderBtn");
  if (!button) return;
  const supported = typeof window.showDirectoryPicker === "function";
  button.disabled = recordingSaving || !supported;
  button.classList.toggle("is-selected", Boolean(recordingDirectoryHandle));
  $("recordFolderLabel").textContent = recordingDirectoryHandle ? "Set" : "Folder";
  button.title = supported
    ? recordingDirectoryHandle
      ? "Recording artifacts will be saved to the selected folder"
      : "Choose a folder for MP4, JSON, and HTML recording artifacts"
    : "Folder save is unavailable in this browser; artifacts will download";
}

async function chooseRecordingDirectory() {
  if (typeof window.showDirectoryPicker !== "function") {
    addHistory("record folder unavailable · using downloads");
    updateRecordFolderButton();
    return;
  }
  try {
    recordingDirectoryHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    addHistory("record folder selected");
  } catch (error) {
    if (error?.name !== "AbortError") {
      addHistory(error.message || "record folder selection failed");
    }
  } finally {
    updateRecordFolderButton();
  }
}

function recordingAssetBaseUrl() {
  return String(UI_CONFIG.recordingAssetBaseUrl || "").trim().replace(/\/+$/, "");
}

function recordingAssetUrl(fileName) {
  const baseUrl = recordingAssetBaseUrl();
  return baseUrl ? `${baseUrl}/${encodeURIComponent(fileName)}` : fileName;
}

function generateTraceId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `trace-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function artifactClientMs(artifact = currentSessionArtifact) {
  if (!artifact?.client_started_at_ms) return 0;
  return Math.round(performance.now() - artifact.client_started_at_ms);
}

function currentRequestSnapshot() {
  const generationMode = selectedGenerationMode();
  const continuousT2V = generationMode === "t2v" && $("continuous").checked;
  const numFrames = generationMode === "t2v"
    ? (continuousT2V ? undefined : readT2VNumFrames())
    : Number($("numFrames").value);
  return compact({
    type: "init_snapshot",
    generation_mode: generationMode,
    model: $("model").value,
    prompt: $("prompt").value,
    size: $("size").value,
    fps: Number($("fps").value || DEFAULT_TARGET_FPS),
    num_frames: continuousT2V ? undefined : numFrames,
    seed: Number($("seed").value),
    num_inference_steps: Number($("steps").value),
    guidance_scale: Number($("guidance").value),
    realtime_causal_sink_size: readOptionalInteger("sinkSize"),
    realtime_causal_kv_cache_num_frames: readOptionalInteger("windowFrames"),
    max_chunks: generationMode === "t2v" || $("continuous").checked ? undefined : 1,
    ...readPreviewTransportParams(),
    ...readFrameInterpolationParams(),
    ...readSuperResolutionParams(),
  });
}

function createSessionArtifact(init = currentRequestSnapshot(), referenceImage = null) {
  const now = new Date();
  const artifact = {
    schema_version: SESSION_ARTIFACT_SCHEMA_VERSION,
    trace_id: generateTraceId(),
    created_at: now.toISOString(),
    page_url: window.location.href,
    user_agent: navigator.userAgent,
    client_started_at_ms: performance.now(),
    server_url: $("serverUrl").value,
    request: {},
    prompt_history: [],
    events: [],
    chunks: [],
    first_rendered_chunks: [],
    recording: null,
  };
  updateSessionArtifactRequest(artifact, init, referenceImage);
  recordPromptHistory(init.prompt, "init", null, artifact);
  return artifact;
}

function updateSessionArtifactRequest(artifact, init, referenceImage = null) {
  artifact.server_url = $("serverUrl").value;
  artifact.request = {
    ...stripBinaryFields(init),
    reference_image: referenceImage,
  };
  artifact.reference_image = referenceImage || null;
  artifact.generation_mode = init.generation_mode || artifact.request.generation_mode || null;
  artifact.model = init.model || artifact.model || "";
}

function beginSessionArtifact(init, referenceImage = null) {
  const artifact = recordingActive && recordingArtifact
    ? recordingArtifact
    : createSessionArtifact(init, referenceImage);
  updateSessionArtifactRequest(artifact, init, referenceImage);
  currentSessionArtifact = artifact;
  if (recordingActive) recordingArtifact = artifact;
  recordPromptHistory(init.prompt, "init", null, artifact);
  recordTrajectoryEvent("session_init", {
    generation_mode: init.generation_mode,
    has_reference_image: Boolean(referenceImage),
    num_frames: init.num_frames,
    max_chunks: init.max_chunks ?? null,
  });
  return artifact;
}

function ensureSessionArtifact() {
  if (!currentSessionArtifact) {
    currentSessionArtifact = createSessionArtifact(currentRequestSnapshot(), null);
  }
  return currentSessionArtifact;
}

function recordTrajectoryEvent(kind, details = {}) {
  if (!currentSessionArtifact && !recordingActive) return null;
  const artifact = ensureSessionArtifact();
  const event = {
    kind,
    client_ms: artifactClientMs(artifact),
    ...jsonSafe(details),
  };
  artifact.events.push(event);
  if (artifact.events.length > SESSION_ARTIFACT_EVENT_LIMIT) {
    artifact.events.splice(0, artifact.events.length - SESSION_ARTIFACT_EVENT_LIMIT);
  }
  return event;
}

function recordPromptHistory(prompt, kind = "prompt_update", eventId = null, artifact = null) {
  const target = artifact || currentSessionArtifact;
  if (!target || typeof prompt !== "string") return;
  const lastPrompt = target.prompt_history[target.prompt_history.length - 1];
  if (lastPrompt && lastPrompt.prompt === prompt && lastPrompt.kind === kind) return;
  target.prompt_history.push(compact({
    kind,
    event_id: eventId,
    client_ms: artifactClientMs(target),
    prompt,
  }));
}

async function createReferenceImageMeta(firstFrame) {
  if (!firstFrame) return null;
  const file = $("firstFrame").files[0];
  const mime = file?.type || selectedPreset?.mime || mimeFromReferenceUrl(selectedReferenceUrl);
  const bytes = firstFrame.byteLength || firstFrame.length || 0;
  const meta = {
    source: file ? "upload" : selectedReferenceUrl ? "preset_url" : "bytes",
    label: file?.name || selectedReferenceLabel || selectedPreset?.name || "",
    url: selectedReferenceUrl || undefined,
    mime,
    bytes,
    first_frame_sha256: await sha256Bytes(firstFrame),
  };
  if (bytes > 0 && bytes <= MAX_EMBEDDED_REFERENCE_IMAGE_BYTES) {
    meta.data_url = await bytesToDataUrl(firstFrame, mime);
  }
  return compact(meta);
}

function mimeFromReferenceUrl(url) {
  const path = String(url || "").split("?")[0].toLowerCase();
  if (path.endsWith(".png")) return "image/png";
  if (path.endsWith(".webp")) return "image/webp";
  return "image/jpeg";
}

async function sha256Bytes(bytes) {
  if (!bytes || !globalThis.crypto?.subtle) return null;
  const buffer = bytes instanceof Uint8Array
    ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    : bytes;
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function bytesToDataUrl(bytes, mime = "application/octet-stream") {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("reference image encode failed"));
    reader.readAsDataURL(new Blob([bytes], { type: mime }));
  });
}

function stripBinaryFields(value) {
  const safe = jsonSafe(value);
  if (value?.first_frame instanceof Uint8Array) {
    safe.first_frame = {
      byte_length: value.first_frame.byteLength,
      note: "binary bytes summarized; see request.reference_image",
    };
  }
  return safe;
}

function jsonSafe(value, depth = 0) {
  if (depth > 8) return "[MaxDepth]";
  if (value == null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (value instanceof Uint8Array) {
    return { binary_type: "Uint8Array", byte_length: value.byteLength };
  }
  if (value instanceof ArrayBuffer) {
    return { binary_type: "ArrayBuffer", byte_length: value.byteLength };
  }
  if (value instanceof Blob) {
    return { binary_type: "Blob", byte_length: value.size, type: value.type };
  }
  if (Array.isArray(value)) return value.map((item) => jsonSafe(item, depth + 1));
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => typeof item !== "function" && item !== undefined)
        .map(([key, item]) => [key, jsonSafe(item, depth + 1)]),
    );
  }
  return String(value);
}

function startRecording() {
  if (recordingActive || recordingSaving) return;
  recordingMode = selectRecordingMode();
  if (!recordingMode) {
    setStatus("Recording unsupported", "error");
    addHistory("recording requires WebCodecs MP4 or MediaRecorder WebM support");
    return;
  }
  recordingActive = true;
  recordingSamples = [];
  recordingEncoder = null;
  recordingEncoderReady = null;
  recordingEncoderConfig = null;
  recordingMediaRecorder = null;
  recordingMediaChunks = [];
  recordingCaptureStream = null;
  recordingMimeType = recordingMode === "mediarecorder-webm"
    ? supportedWebmMimeType()
    : "video/mp4";
  recordingFrameIndex = 0;
  recordingFps = Math.max(1, previewPlaybackTargetFps());
  recordingEncodeChain = Promise.resolve();
  recordingBaseFileName = recordingFileName().replace(/\.[^.]*$/, "");
  recordingArtifact = ensureSessionArtifact();
  recordingArtifact.recording = {
    base_file_name: recordingBaseFileName,
    started_at: new Date().toISOString(),
    started_client_ms: artifactClientMs(recordingArtifact),
    mode: recordingMode,
    mime_type: recordingMimeType,
    capture_scope: "stage",
    capture_width: RECORDING_STAGE_WIDTH,
    capture_height: RECORDING_STAGE_HEIGHT,
    target_fps: recordingFps,
  };
  if (recordingMode === "mediarecorder-webm") startWebmRecording();
  recordTrajectoryEvent("record_start", { target_fps: recordingFps });
  recordingTimer = window.setInterval(updateRecordButton, 250);
  updateRecordButton();
  updateRecordFolderButton();
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
  updateRecordFolderButton();

  const extension = recordingMode === "mediarecorder-webm" ? "webm" : "mp4";
  const fileName = `${recordingBaseFileName || recordingFileName(extension).replace(/\.[^.]*$/, "")}.${extension}`;
  try {
    recordTrajectoryEvent("record_stop", {
      encoded_frames: recordingSamples.length,
      captured_frames: recordingFrameIndex,
      mode: recordingMode,
    });
    const videoBlob = recordingMode === "mediarecorder-webm"
      ? await stopWebmRecording()
      : await buildMp4RecordingBlob();
    await saveRecordingArtifactFiles(videoBlob, fileName);
    addHistory(`saved ${recordingFrameIndex} frames · ${extension}/json/html`);
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
    stopRecordingCaptureStream();
    recordingMediaRecorder = null;
    recordingMediaChunks = [];
    recordingCaptureStream = null;
    recordingMode = "";
    recordingSaving = false;
    recordingSamples = [];
    updateRecordButton();
    updateRecordFolderButton();
  }
}

async function buildMp4RecordingBlob() {
  await recordingEncodeChain;
  if (!recordingEncoder || !recordingSamples.length) throw new Error("No frames were recorded");
  await recordingEncoder.flush();
  return buildRecordingMp4();
}

function startWebmRecording() {
  drawRecordingStageFrame(canvas);
  recordingMediaChunks = [];
  recordingCaptureStream = recordingCanvas.captureStream(recordingFps);
  recordingMediaRecorder = new MediaRecorder(
    recordingCaptureStream,
    recordingMimeType ? { mimeType: recordingMimeType } : undefined,
  );
  recordingMimeType = recordingMediaRecorder.mimeType || recordingMimeType || "video/webm";
  recordingMediaRecorder.ondataavailable = (event) => {
    if (event.data?.size) recordingMediaChunks.push(event.data);
  };
  recordingMediaRecorder.onerror = (event) => {
    recordingActive = false;
    addHistory(event.error?.message || "recording media recorder failed");
    updateRecordButton();
  };
  recordingMediaRecorder.start(250);
}

function stopWebmRecording() {
  return new Promise((resolve, reject) => {
    const recorder = recordingMediaRecorder;
    if (!recorder) {
      reject(new Error("No WebM recorder was started"));
      return;
    }
    recorder.onstop = () => {
      stopRecordingCaptureStream();
      if (!recordingMediaChunks.length) {
        reject(new Error("No frames were recorded"));
        return;
      }
      resolve(new Blob(recordingMediaChunks, { type: recordingMimeType || "video/webm" }));
    };
    recorder.onerror = (event) => {
      stopRecordingCaptureStream();
      reject(event.error || new Error("recording media recorder failed"));
    };
    if (recorder.state === "inactive") {
      recorder.onstop();
      return;
    }
    try {
      recorder.requestData?.();
      recorder.stop();
    } catch (error) {
      reject(error);
    }
  });
}

function stopRecordingCaptureStream() {
  for (const track of recordingCaptureStream?.getTracks?.() || []) track.stop();
  recordingCaptureStream = null;
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
  if (recordingMode === "mediarecorder-webm") {
    drawRecordingStageFrame(image);
    recordingFrameIndex += 1;
    recordingMediaRecorder?.requestData?.();
    return;
  }
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
  drawRecordingStageFrame(image);
  return new VideoFrame(recordingCanvas, { timestamp, duration });
}

function drawRecordingStageFrame(image) {
  ensureRecordingStageCanvas();
  const source = recordingDrawableSource(image || canvas);
  const sourceWidth = source.width || canvas.width || 1280;
  const sourceHeight = source.height || canvas.height || 720;
  recordingCtx.save();
  recordingCtx.imageSmoothingEnabled = true;
  recordingCtx.imageSmoothingQuality = "medium";
  recordingCtx.fillStyle = "#11140f";
  recordingCtx.fillRect(0, 0, RECORDING_STAGE_WIDTH, RECORDING_STAGE_HEIGHT);
  drawRecordingTopbar();
  drawRecordingPreview(source, sourceWidth, sourceHeight);
  drawRecordingControls();
  drawRecordingTimeline();
  drawRecordingTelemetry();
  recordingCtx.restore();
}

function ensureRecordingStageCanvas() {
  if (
    recordingCanvas.width !== RECORDING_STAGE_WIDTH ||
    recordingCanvas.height !== RECORDING_STAGE_HEIGHT
  ) {
    recordingCanvas.width = RECORDING_STAGE_WIDTH;
    recordingCanvas.height = RECORDING_STAGE_HEIGHT;
  }
}

function recordingDrawableSource(image) {
  if (image instanceof ImageData) {
    if (scratchCanvas.width !== image.width || scratchCanvas.height !== image.height) {
      scratchCanvas.width = image.width;
      scratchCanvas.height = image.height;
    }
    scratchCtx.putImageData(image, 0, 0);
    return scratchCanvas;
  }
  return image || canvas;
}

function drawRecordingTopbar() {
  const y = 0;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_TOPBAR_HEIGHT, "#10140f");
  recordingCtx.fillStyle = "rgba(232, 234, 223, 0.12)";
  recordingCtx.fillRect(0, RECORDING_STAGE_TOPBAR_HEIGHT - 1, RECORDING_STAGE_WIDTH, 1);

  let x = RECORDING_STAGE_PADDING;
  const dotKind = $("statusDot")?.classList.contains("live")
    ? "live"
    : $("statusDot")?.classList.contains("error") ? "error" : "";
  recordingCtx.beginPath();
  recordingCtx.arc(x + 6, y + RECORDING_STAGE_TOPBAR_HEIGHT / 2, 5, 0, Math.PI * 2);
  recordingCtx.fillStyle = dotKind === "live" ? "#8ecf9d" : dotKind === "error" ? "#b9543c" : "#687164";
  recordingCtx.fill();
  x += 24;
  drawRecordingLabel(recordingElementText("statusText", "Idle"), x, y + 33, {
    color: "#e8eadf",
    font: "18px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 120,
  });
  x += 126;
  drawRecordingLabel(recordingElementText("chunkText", "chunk -"), x, y + 33, {
    color: "#e8eadf",
    font: "15px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 96,
  });
  x += 118;
  drawRecordingPill(x, y + 11, 126, 32, {
    label: recordingActive ? "Stop" : "Record",
    detail: recordingElementText("recordDuration", "00:00"),
    active: recordingActive,
  });
  x += 146;
  drawRecordingPill(x, y + 11, 86, 32, {
    label: recordingElementText("recordFolderLabel", "Folder"),
    active: $("recordFolderBtn")?.classList.contains("is-selected"),
  });

  let right = RECORDING_STAGE_WIDTH - RECORDING_STAGE_PADDING;
  right = drawRecordingTopbarStatRight(`buffer ${recordingElementText("stageLatencyText", "-")}`, right, y);
  right = drawRecordingTopbarStatRight(`action ${recordingElementText("actionStateText", "-")}`, right, y);
  right = drawRecordingTopbarStatRight(`source ${recordingElementText("theoreticalFpsText", "-")}`, right, y);
  right = drawRecordingTopbarStatRight(`render ${recordingElementText("renderFps", "0")} fps`, right, y);
  right = drawRecordingTopbarStatRight(`output ${recordingElementText("outputSizeText", "-")}`, right, y);
  drawRecordingTopbarStatRight(
    `Preview ${recordingElementText("previewScaleText", "100%")}`,
    right,
    y,
  );
}

function drawRecordingTopbarStatRight(text, right, y) {
  recordingCtx.font = "600 15px ui-sans-serif, system-ui, sans-serif";
  const width = Math.min(recordingCtx.measureText(text).width, 250);
  drawRecordingLabel(text, right - width, y + 33, {
    color: "#fffdf7",
    font: "600 15px ui-sans-serif, system-ui, sans-serif",
    maxWidth: width,
  });
  return right - width - 24;
}

function drawRecordingPreview(source, sourceWidth, sourceHeight) {
  const y = RECORDING_STAGE_TOPBAR_HEIGHT;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_PREVIEW_HEIGHT, "#11140f");
  const previewRect = {
    x: 118,
    y,
    width: RECORDING_STAGE_WIDTH - 236,
    height: RECORDING_STAGE_PREVIEW_HEIGHT,
  };
  const scale = Math.min(
    previewRect.width / Math.max(1, sourceWidth),
    previewRect.height / Math.max(1, sourceHeight),
  );
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);
  const drawX = Math.round(previewRect.x + (previewRect.width - drawWidth) / 2);
  const drawY = Math.round(previewRect.y + (previewRect.height - drawHeight) / 2);
  fillRecordingRect(previewRect.x, previewRect.y, previewRect.width, previewRect.height, "#151912");
  recordingCtx.drawImage(source, drawX, drawY, drawWidth, drawHeight);
}

function drawRecordingControls() {
  const y = RECORDING_STAGE_TOPBAR_HEIGHT + RECORDING_STAGE_PREVIEW_HEIGHT;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_CONTROLS_HEIGHT, "#151912");
  recordingCtx.fillStyle = "rgba(232, 234, 223, 0.12)";
  recordingCtx.fillRect(0, y, RECORDING_STAGE_WIDTH, 1);
  const gap = 38;
  const clusterWidth = (RECORDING_STAGE_WIDTH - RECORDING_STAGE_PADDING * 2 - gap) / 2;
  drawRecordingControlCluster("MOVE", RECORDING_STAGE_PADDING, y + 24, clusterWidth, [
    [null, "w", null],
    ["a", "s", "d"],
  ]);
  drawRecordingControlCluster(
    "LOOK",
    RECORDING_STAGE_PADDING + clusterWidth + gap,
    y + 24,
    clusterWidth,
    [
      [null, "i", null],
      ["j", "k", "l"],
    ],
  );
}

function drawRecordingControlCluster(title, x, y, width, rows) {
  drawRecordingLabel(title, x, y + 61, {
    color: "rgba(232, 234, 223, 0.62)",
    font: "15px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 66,
  });
  const padX = x + 72;
  const cellGap = 8;
  const buttonWidth = (width - 72 - cellGap * 2) / 3;
  const buttonHeight = 44;
  rows.forEach((row, rowIndex) => {
    row.forEach((action, columnIndex) => {
      if (!action) return;
      drawRecordingControlButton(
        action,
        padX + columnIndex * (buttonWidth + cellGap),
        y + rowIndex * (buttonHeight + cellGap),
        buttonWidth,
        buttonHeight,
      );
    });
  });
}

function drawRecordingControlButton(action, x, y, width, height) {
  const active = controlStateController?.activeActions?.has(action);
  const radius = 5;
  fillRecordingRoundedRect(x, y, width, height, radius, active ? "#8c9288" : "#eef1ec");
  strokeRecordingRoundedRect(
    x,
    y,
    width,
    height,
    radius,
    active ? "#aeb4aa" : "rgba(232, 234, 223, 0.18)",
  );
  const meta = CONTROL_ACTION_META[action] || {};
  drawRecordingLabel(meta.label || action.toUpperCase(), x + width / 2, y + 28, {
    align: "center",
    color: active ? "#fffdf7" : "#11140f",
    font: "16px ui-sans-serif, system-ui, sans-serif",
    maxWidth: width - 34,
  });
  const keyLabel = action === "i" ? "↑" : action === "j" ? "←" : action === "k" ? "↓" : action === "l" ? "→" : action.toUpperCase();
  drawRecordingLabel(keyLabel, x + width - 16, y + 16, {
    align: "right",
    color: active ? "rgba(255, 253, 247, 0.78)" : "#687164",
    font: "700 13px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 28,
  });
}

function drawRecordingTimeline() {
  const y = RECORDING_STAGE_TOPBAR_HEIGHT +
    RECORDING_STAGE_PREVIEW_HEIGHT +
    RECORDING_STAGE_CONTROLS_HEIGHT;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_TIMELINE_HEIGHT, "#11140f");
  recordingCtx.fillStyle = "rgba(232, 234, 223, 0.12)";
  recordingCtx.fillRect(0, y, RECORDING_STAGE_WIDTH, 1);
  const text = [
    recordingElementText("queueText", "queue 0"),
    recordingElementText("frameText", "frames 0"),
    recordingElementText("byteText", "0 MB"),
  ].join("   ");
  drawRecordingLabel(text, RECORDING_STAGE_WIDTH - RECORDING_STAGE_PADDING, y + 31, {
    align: "right",
    color: "#e8eadf",
    font: "16px ui-sans-serif, system-ui, sans-serif",
    maxWidth: RECORDING_STAGE_WIDTH - RECORDING_STAGE_PADDING * 2,
  });
}

function drawRecordingTelemetry() {
  const y = RECORDING_STAGE_TOPBAR_HEIGHT +
    RECORDING_STAGE_PREVIEW_HEIGHT +
    RECORDING_STAGE_CONTROLS_HEIGHT +
    RECORDING_STAGE_TIMELINE_HEIGHT;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_TELEMETRY_HEIGHT, "#11140f");
  const rows = [
    [
      ["Payload", recordingElementText("payloadMode", selectedTransportLabel())],
      ["Server send", recordingElementText("serverSendText", "-")],
      ["Chunk bytes", recordingElementText("chunkPayloadText", "-")],
    ],
    [
      ["Chunk wait", recordingElementText("latencyText", "-")],
      ["Decode", recordingElementText("decodeText", "-")],
      ["Display lag", recordingElementText("displayLagText", "-")],
    ],
  ];
  const cellWidth = RECORDING_STAGE_WIDTH / 3;
  const cellHeight = RECORDING_STAGE_TELEMETRY_HEIGHT / 2;
  rows.forEach((row, rowIndex) => {
    row.forEach(([label, value], columnIndex) => {
      const x = columnIndex * cellWidth;
      const cellY = y + rowIndex * cellHeight;
      recordingCtx.fillStyle = "rgba(232, 234, 223, 0.1)";
      recordingCtx.fillRect(x, cellY, cellWidth, 1);
      if (columnIndex > 0) recordingCtx.fillRect(x, cellY, 1, cellHeight);
      drawRecordingLabel(label, x + 18, cellY + 30, {
        color: "rgba(232, 234, 223, 0.62)",
        font: "15px ui-sans-serif, system-ui, sans-serif",
        maxWidth: cellWidth * 0.45,
      });
      drawRecordingLabel(value, x + cellWidth - 18, cellY + 30, {
        align: "right",
        color: "#fffdf7",
        font: "700 16px ui-sans-serif, system-ui, sans-serif",
        maxWidth: cellWidth * 0.5,
      });
    });
  });
}

function drawRecordingPill(x, y, width, height, { label, detail = "", active = false }) {
  fillRecordingRoundedRect(
    x,
    y,
    width,
    height,
    6,
    active ? "#b9543c" : "rgba(238, 241, 236, 0.08)",
  );
  strokeRecordingRoundedRect(x, y, width, height, 6, "rgba(232, 234, 223, 0.24)");
  drawRecordingLabel(label, x + 14, y + 21, {
    color: "#e8eadf",
    font: "14px ui-sans-serif, system-ui, sans-serif",
    maxWidth: width - (detail ? 62 : 28),
  });
  if (detail) {
    drawRecordingLabel(detail, x + width - 12, y + 21, {
      align: "right",
      color: "rgba(232, 234, 223, 0.78)",
      font: "14px ui-sans-serif, system-ui, sans-serif",
      maxWidth: 48,
    });
  }
}

function recordingElementText(id, fallback = "-") {
  const value = $(id)?.textContent;
  return value && String(value).trim() ? String(value).trim() : fallback;
}

function drawRecordingLabel(text, x, y, {
  color = "#fffdf7",
  font = "14px ui-sans-serif, system-ui, sans-serif",
  align = "left",
  maxWidth = undefined,
} = {}) {
  recordingCtx.save();
  recordingCtx.fillStyle = color;
  recordingCtx.font = font;
  recordingCtx.textAlign = align;
  recordingCtx.textBaseline = "alphabetic";
  if (maxWidth === undefined) {
    recordingCtx.fillText(String(text), x, y);
  } else {
    recordingCtx.fillText(String(text), x, y, maxWidth);
  }
  recordingCtx.restore();
}

function fillRecordingRect(x, y, width, height, fillStyle) {
  recordingCtx.fillStyle = fillStyle;
  recordingCtx.fillRect(x, y, width, height);
}

function fillRecordingRoundedRect(x, y, width, height, radius, fillStyle) {
  recordingCtx.beginPath();
  recordingRoundedRectPath(x, y, width, height, radius);
  recordingCtx.fillStyle = fillStyle;
  recordingCtx.fill();
}

function strokeRecordingRoundedRect(x, y, width, height, radius, strokeStyle) {
  recordingCtx.beginPath();
  recordingRoundedRectPath(x, y, width, height, radius);
  recordingCtx.strokeStyle = strokeStyle;
  recordingCtx.lineWidth = 1;
  recordingCtx.stroke();
}

function recordingRoundedRectPath(x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  if (recordingCtx.roundRect) {
    recordingCtx.roundRect(x, y, width, height, r);
    return;
  }
  recordingCtx.moveTo(x + r, y);
  recordingCtx.lineTo(x + width - r, y);
  recordingCtx.quadraticCurveTo(x + width, y, x + width, y + r);
  recordingCtx.lineTo(x + width, y + height - r);
  recordingCtx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  recordingCtx.lineTo(x + r, y + height);
  recordingCtx.quadraticCurveTo(x, y + height, x, y + height - r);
  recordingCtx.lineTo(x, y + r);
  recordingCtx.quadraticCurveTo(x, y, x + r, y);
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

function sidecarFileName(fileName, extension) {
  return `${String(fileName).replace(/\.[^.]*$/, "")}.${extension}`;
}

async function saveRecordingArtifactFiles(videoBlob, fileName) {
  const artifact = finalizeRecordingArtifact(videoBlob, fileName);
  const jsonFileName = artifact.recording.json_file;
  const htmlFileName = artifact.recording.html_file;
  const jsonBlob = new Blob(
    [JSON.stringify(artifact, null, 2)],
    { type: "application/json" },
  );
  const htmlBlob = new Blob(
    [buildReplayHtml(artifact)],
    { type: "text/html" },
  );
  await saveRecordingFiles([
    { name: fileName, blob: videoBlob },
    { name: jsonFileName, blob: jsonBlob },
    { name: htmlFileName, blob: htmlBlob },
  ]);
}

function finalizeRecordingArtifact(videoBlob, fileName) {
  const artifact = recordingArtifact || ensureSessionArtifact();
  const jsonFileName = sidecarFileName(fileName, "json");
  const htmlFileName = sidecarFileName(fileName, "html");
  artifact.recording = {
    ...(artifact.recording || {}),
    stopped_at: new Date().toISOString(),
    stopped_client_ms: artifactClientMs(artifact),
    mode: recordingMode,
    mime_type: videoBlob.type || recordingMimeType,
    fps: recordingFps,
    frames: recordingFrameIndex,
    encoded_chunks: recordingMode === "mediarecorder-webm"
      ? recordingMediaChunks.length
      : recordingSamples.length,
    video_file: fileName,
    video_url: recordingAssetUrl(fileName),
    video_bytes: videoBlob.size,
    json_file: jsonFileName,
    json_url: recordingAssetUrl(jsonFileName),
    html_file: htmlFileName,
    html_url: recordingAssetUrl(htmlFileName),
    asset_base_url: recordingAssetBaseUrl() || null,
  };
  return artifact;
}

async function saveRecordingFiles(files) {
  if (recordingDirectoryHandle) {
    await ensureRecordingDirectoryWritable(recordingDirectoryHandle);
    for (const file of files) {
      const handle = await recordingDirectoryHandle.getFileHandle(file.name, { create: true });
      const writable = await handle.createWritable();
      await writable.write(file.blob);
      await writable.close();
    }
    return;
  }
  for (const file of files) downloadBlob(file.blob, file.name);
}

async function ensureRecordingDirectoryWritable(directoryHandle) {
  const options = { mode: "readwrite" };
  if (directoryHandle.queryPermission) {
    const existing = await directoryHandle.queryPermission(options);
    if (existing === "granted") return;
  }
  if (directoryHandle.requestPermission) {
    const requested = await directoryHandle.requestPermission(options);
    if (requested === "granted") return;
  }
  throw new Error("recording folder permission denied");
}

function buildReplayHtml(artifact) {
  const recording = artifact.recording || {};
  const request = artifact.request || {};
  const referenceImage = request.reference_image || null;
  const prompts = artifact.prompt_history || [];
  const events = artifact.events || [];
  const eventRows = events.slice(-600).map((event) => (
    `<tr><td>${escapeHtmlText(event.kind)}</td><td>${formatReplayMs(event.client_ms)}</td><td><code>${escapeHtmlText(JSON.stringify(event))}</code></td></tr>`
  )).join("");
  const promptRows = prompts.map((item) => (
    `<li><b>${escapeHtmlText(item.kind)}</b> ${formatReplayMs(item.client_ms)}<pre>${escapeHtmlText(item.prompt || "")}</pre></li>`
  )).join("");
  function replayReferenceImageSrc(referenceImage) {
    return referenceImage?.data_url || referenceImage?.url || referenceImage?.source_url || referenceImage?.preset_url || "";
  }
  const referenceSrc = replayReferenceImageSrc(referenceImage);
  const referenceBlock = referenceSrc
    ? `<img class="reference" src="${escapeHtmlAttribute(referenceSrc)}" alt="reference image" />`
    : `<div class="reference empty">${escapeHtmlText(referenceImage ? referenceImage.label || "reference image" : "T2V session: no reference image")}</div>`;
  const artifactJson = JSON.stringify(artifact)
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SGLang realtime replay ${escapeHtmlText(artifact.trace_id || "")}</title>
  <style>
    body { margin: 0; background: #eef1ec; color: #171a16; font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(1480px, calc(100vw - 32px)); margin: 24px auto 56px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    .meta, .grid, .events, .prompt-list { margin-top: 16px; }
    .grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 16px; align-items: start; }
    .replay-stage { overflow: hidden; border: 1px solid #11140f; border-radius: 8px; background: #11140f; box-shadow: 0 18px 60px rgba(23, 26, 22, 0.12); }
    .replay-topbar, .replay-timeline { display: flex; align-items: center; gap: 10px; min-width: 0; height: 44px; padding: 0 14px; color: #e8eadf; background: rgba(17, 20, 15, 0.9); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .replay-topbar-spacer { flex: 1; }
    .replay-dot { width: 8px; height: 8px; border-radius: 50%; background: #8ecf9d; box-shadow: 0 0 0 4px rgba(142, 207, 157, 0.14); }
    .replay-pill { display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 10px; border: 1px solid rgba(232, 234, 223, 0.22); border-radius: 6px; background: rgba(238, 241, 236, 0.08); color: #e8eadf; }
    .replay-video-shell { position: relative; display: grid; place-items: center; min-height: 320px; background: #11140f; }
    .replay-video { display: block; width: 100%; max-height: 72vh; border: 0; border-radius: 0; background: #11140f; }
    .replay-cursor { position: absolute; inset: 0 auto 0 0; width: 2px; transform: translateX(var(--replay-cursor-x, -200%)); background: rgba(142, 207, 157, 0.86); box-shadow: 0 0 0 1px rgba(17, 20, 15, 0.62); pointer-events: none; opacity: 0; }
    .replay-video-shell.is-inspecting .replay-cursor { opacity: 1; }
    .replay-inspector { position: fixed; left: 0; top: 0; z-index: 40; width: min(430px, calc(100vw - 28px)); max-height: min(520px, calc(100vh - 28px)); overflow: auto; border: 1px solid rgba(232, 234, 223, 0.34); border-radius: 8px; background: rgba(251, 250, 245, 0.95); color: #171a16; box-shadow: 0 18px 50px rgba(17, 20, 15, 0.34); pointer-events: none; transform: translate(14px, 14px); }
    .replay-inspector[hidden] { display: none; }
    .replay-inspector-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px 8px; border-bottom: 1px solid #cbd2c4; }
    .replay-inspector-header b { font-size: 13px; }
    .replay-inspector-header span { color: #687164; font-size: 12px; font-variant-numeric: tabular-nums; }
    .replay-inspector-grid { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 6px 10px; padding: 10px 12px; }
    .replay-inspector-grid span { color: #687164; font-size: 12px; }
    .replay-inspector-grid b { min-width: 0; font-size: 12px; word-break: break-word; }
    .replay-inspector-block { padding: 0 12px 10px; }
    .replay-inspector-block span { display: block; margin-bottom: 4px; color: #687164; font-size: 12px; }
    .replay-inspector-block pre { max-height: 110px; margin: 0; padding: 8px; border-radius: 6px; font-size: 11px; }
    .replay-inspector-image { display: none; width: 86px; height: 48px; object-fit: cover; margin: 0 0 8px; border: 1px solid #cbd2c4; border-radius: 5px; }
    .replay-inspector-image.has-image { display: block; }
    .replay-timeline { justify-content: flex-end; border-top: 1px solid rgba(232, 234, 223, 0.12); }
    video, .reference { width: 100%; border: 1px solid #cbd2c4; border-radius: 8px; background: #11140f; }
    .replay-stage video { border: 0; border-radius: 0; }
    .reference.empty { min-height: 160px; display: grid; place-items: center; border: 1px dashed #cbd2c4; border-radius: 8px; color: #687164; }
    pre, code { white-space: pre-wrap; word-break: break-word; }
    pre { margin: 8px 0 0; padding: 12px; background: #fbfaf5; border: 1px solid #cbd2c4; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; background: #fbfaf5; border: 1px solid #cbd2c4; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #d8ddd2; vertical-align: top; text-align: left; }
    th { color: #687164; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .card { padding: 10px; border: 1px solid #cbd2c4; border-radius: 8px; background: #fbfaf5; }
    .card b { display: block; font-size: 16px; }
    @media (max-width: 860px) { .grid, .cards { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>SGLang realtime replay</h1>
    <div class="meta">Trace ${escapeHtmlText(artifact.trace_id || "-")} · ${escapeHtmlText(request.generation_mode || "-")} · ${escapeHtmlText(recording.video_file || "-")}</div>
    <div class="cards">
      <div class="card"><span>Frames</span><b>${escapeHtmlText(recording.frames ?? "-")}</b></div>
      <div class="card"><span>FPS</span><b>${escapeHtmlText(recording.fps ?? request.fps ?? "-")}</b></div>
      <div class="card"><span>Events</span><b>${escapeHtmlText(events.length)}</b></div>
      <div class="card"><span>Mode</span><b>${escapeHtmlText(request.generation_mode || "-")}</b></div>
    </div>
    <section class="grid">
      <div>
        <section class="replay-stage" aria-label="Recorded realtime stage">
          <div class="replay-topbar">
            <span class="replay-dot" aria-hidden="true"></span>
            <span>Replay</span>
            <span>frames ${escapeHtmlText(recording.frames ?? "-")}</span>
            <span class="replay-pill">Record ${escapeHtmlText(recording.fps ?? request.fps ?? "-")} fps</span>
            <span class="replay-topbar-spacer"></span>
            <span>mode ${escapeHtmlText(request.generation_mode || "-")}</span>
            <span>scope ${escapeHtmlText(recording.capture_scope || "viewport")}</span>
          </div>
          <div class="replay-video-shell">
            <video id="replayVideo" class="replay-video" controls preload="metadata" src="${escapeHtmlAttribute(recording.video_url || recording.video_file || "")}"></video>
            <div id="replayCursor" class="replay-cursor" aria-hidden="true"></div>
            <aside id="replayInspector" class="replay-inspector" hidden aria-live="polite">
              <div class="replay-inspector-header">
                <b>Cursor trace</b>
                <span id="replayInspectorTime">-</span>
              </div>
              <div class="replay-inspector-grid">
                <span>User keys</span><b id="replayInspectorUserKeys">-</b>
                <span>SGLang keys</span><b id="replayInspectorSglangKeys">-</b>
                <span>Chunk / event</span><b id="replayInspectorChunk">-</b>
                <span>Reference image</span><b id="replayInspectorImageMeta">-</b>
              </div>
              <div class="replay-inspector-block">
                <img id="replayInspectorImage" class="replay-inspector-image" alt="reference image at cursor" />
                <span>Prompt at cursor</span>
                <pre id="replayInspectorPrompt">-</pre>
              </div>
              <div class="replay-inspector-block">
                <span>Nearby events</span>
                <pre id="replayInspectorEvents">-</pre>
              </div>
            </aside>
          </div>
          <div class="replay-timeline">
            <span id="replayActiveText">input idle</span>
            <span>${escapeHtmlText(recording.video_file || "-")}</span>
          </div>
        </section>
        <h2>Prompt History</h2>
        <ol class="prompt-list">${promptRows}</ol>
      </div>
      <aside>
        <h2>Reference</h2>
        ${referenceBlock}
        <h2>Request</h2>
        <pre>${escapeHtmlText(JSON.stringify(request, null, 2))}</pre>
      </aside>
    </section>
    <section class="events">
      <h2>Recent Events</h2>
      <table>
        <thead><tr><th>Kind</th><th>Time</th><th>Payload</th></tr></thead>
        <tbody>${eventRows}</tbody>
      </table>
    </section>
  </main>
  <script id="recording-artifact" type="application/json">${artifactJson}</script>
  <script>
    (() => {
      const artifactNode = document.getElementById("recording-artifact");
      const video = document.getElementById("replayVideo");
      const activeText = document.getElementById("replayActiveText");
      const videoShell = video && video.closest(".replay-video-shell");
      const inspector = document.getElementById("replayInspector");
      const inspectorTime = document.getElementById("replayInspectorTime");
      const inspectorUserKeys = document.getElementById("replayInspectorUserKeys");
      const inspectorSglangKeys = document.getElementById("replayInspectorSglangKeys");
      const inspectorChunk = document.getElementById("replayInspectorChunk");
      const inspectorPrompt = document.getElementById("replayInspectorPrompt");
      const inspectorEvents = document.getElementById("replayInspectorEvents");
      const inspectorImage = document.getElementById("replayInspectorImage");
      const inspectorImageMeta = document.getElementById("replayInspectorImageMeta");
      if (!artifactNode || !video) return;
      const artifact = JSON.parse(artifactNode.textContent || "{}");
      const events = Array.isArray(artifact.events)
        ? artifact.events.slice().sort((left, right) => Number(left.client_ms || 0) - Number(right.client_ms || 0))
        : [];
      const recordingStartMs = Number(artifact.recording && artifact.recording.started_client_ms) || 0;
      const recording = artifact.recording || {};
      const request = artifact.request || {};
      const prompts = Array.isArray(artifact.prompt_history)
        ? artifact.prompt_history.slice().sort((left, right) => Number(left.client_ms || 0) - Number(right.client_ms || 0))
        : [];
      const tracedChunks = events
        .filter((event) => event.kind === "trace_event" && event.trace?.event === "server.chunk_complete")
        .map((event) => ({
          ...event.trace,
          client_ms: event.client_ms,
          received_client_ms: event.client_ms,
        }));
      const legacyChunks = events.filter((event) => event.kind === "server_chunk_stats");
      const chunks = Array.isArray(artifact.chunks) && artifact.chunks.length
        ? artifact.chunks.slice().sort((left, right) => replayEventTime(left) - replayEventTime(right))
        : (tracedChunks.length ? tracedChunks : legacyChunks)
          .sort((left, right) => replayEventTime(left) - replayEventTime(right));
      const referenceImage = request.reference_image || artifact.reference_image || null;
      const referenceSrc = replayReferenceImageSrc(referenceImage);
      const cameraEventsById = new Map();
      events.forEach((event) => {
        if (event.kind === "camera_actions_sent" && event.event_id !== undefined && event.event_id !== null) {
          cameraEventsById.set(Number(event.event_id), event);
        }
      });
      const replayActionLabels = {
        w: "W Forward",
        a: "A Left",
        s: "S Back",
        d: "D Right",
        i: "↑ Pitch +",
        j: "← Yaw -",
        k: "↓ Pitch -",
        l: "→ Yaw +",
      };
      const REPLAY_INSPECTOR_OFFSET_PX = 16;

      function applyReplayEvent(active, event) {
        if (!event || typeof event.kind !== "string") return active;
        if (event.kind === "camera_actions_sent" && Array.isArray(event.active_actions)) {
          return new Set(event.active_actions.map(String));
        }
        const action = typeof event.action === "string" ? event.action : "";
        if (!action) return active;
        if (event.kind === "key_down" || event.kind === "control_button_down") active.add(action);
        if (event.kind === "key_up" || event.kind === "control_button_up") active.delete(action);
        return active;
      }

      function replayActionsAt(clientMs) {
        let active = new Set();
        for (const event of events) {
          if (Number(event.client_ms || 0) > clientMs) break;
          active = applyReplayEvent(active, event);
        }
        return active;
      }

      function userActionsAt(clientMs) {
        const active = new Set();
        for (const event of events) {
          if (Number(event.client_ms || 0) > clientMs) break;
          const action = typeof event.action === "string" ? event.action : "";
          if (!action) continue;
          if (event.kind === "key_down" || event.kind === "control_button_down") active.add(action);
          if (event.kind === "key_up" || event.kind === "control_button_up") active.delete(action);
        }
        return active;
      }

      function replayEventTime(event) {
        return Number(event.received_client_ms ?? event.client_ms ?? 0);
      }

      function promptAt(clientMs) {
        let prompt = { prompt: request.prompt || "-", kind: "request", client_ms: 0 };
        for (const item of prompts) {
          if (Number(item.client_ms || 0) > clientMs) break;
          prompt = item;
        }
        return prompt;
      }

      function chunkAt(clientMs) {
        let selected = null;
        for (const chunk of chunks) {
          if (replayEventTime(chunk) > clientMs) break;
          selected = chunk;
        }
        return selected || chunks[0] || null;
      }

      function sglangActionsForEventId(eventId) {
        if (eventId === undefined || eventId === null || eventId === "") return new Set();
        const event = cameraEventsById.get(Number(eventId));
        return new Set(Array.isArray(event?.active_actions) ? event.active_actions.map(String) : []);
      }

      function actionText(actions) {
        const labels = Array.from(actions)
          .sort()
          .map((action) => replayActionLabels[action] || action.toUpperCase());
        return labels.length ? labels.join(" + ") : "idle";
      }

      function eventSummaryAt(clientMs) {
        const interesting = events.filter((event) => (
          [
            "key_down",
            "key_up",
            "control_button_down",
            "control_button_up",
            "camera_actions_sent",
            "prompt_update",
            "server_chunk_stats",
          ].includes(event.kind)
        ));
        let nearby = interesting.filter((event) => Math.abs(Number(event.client_ms || 0) - clientMs) <= 750);
        if (!nearby.length) {
          nearby = interesting.filter((event) => Number(event.client_ms || 0) <= clientMs).slice(-6);
        } else {
          nearby = nearby.slice(-8);
        }
        return nearby.map(formatReplayEventSummary).join("\\n") || "-";
      }

      function formatReplayEventSummary(event) {
        const parts = [
          formatReplayClientMs(Number(event.client_ms || 0)),
          event.kind,
        ];
        if (event.action) parts.push("action=" + event.action.toUpperCase());
        if (event.event_id !== undefined) parts.push("event#" + event.event_id);
        if (event.chunk_index !== undefined) parts.push("chunk#" + event.chunk_index);
        if (Array.isArray(event.active_actions)) {
          parts.push("active=" + actionText(new Set(event.active_actions.map(String))));
        }
        return parts.join(" · ");
      }

      function referenceImageText() {
        if (!referenceImage) return "T2V / no reference image";
        const parts = [
          referenceImage.label || "reference image",
          referenceImage.source || "",
          referenceImage.mime || "",
          referenceImage.bytes ? String(referenceImage.bytes) + " bytes" : "",
        ].filter(Boolean);
        return parts.join(" · ");
      }

      function replayReferenceImageSrc(referenceImage) {
        return referenceImage?.data_url || referenceImage?.url || referenceImage?.source_url || referenceImage?.preset_url || "";
      }

      function replayDurationSeconds() {
        if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
        const frames = Number(recording.frames || 0);
        const fps = Number(recording.fps || request.fps || 0);
        return frames > 0 && fps > 0 ? frames / fps : 0;
      }

      function clampReplayRatio(value) {
        return Math.min(1, Math.max(0, value));
      }

      function replayClientMsFromPointer(event) {
        const rect = video.getBoundingClientRect();
        const ratio = clampReplayRatio((event.clientX - rect.left) / Math.max(1, rect.width));
        const durationSeconds = replayDurationSeconds();
        if (videoShell) videoShell.style.setProperty("--replay-cursor-x", (ratio * 100).toFixed(2) + "%");
        return recordingStartMs + durationSeconds * ratio * 1000;
      }

      function formatReplayClientMs(ms) {
        const relative = Math.max(0, ms - recordingStartMs);
        return (relative / 1000).toFixed(2) + "s";
      }

      function positionReplayInspector(event) {
        if (!inspector || !event) return;
        inspector.hidden = false;
        const left = event.clientX + REPLAY_INSPECTOR_OFFSET_PX;
        const top = event.clientY + REPLAY_INSPECTOR_OFFSET_PX;
        inspector.style.transform = "translate(" + Math.round(left) + "px, " + Math.round(top) + "px)";
      }

      function inspectReplayAt(clientMs) {
        if (!inspector) return;
        const userActions = userActionsAt(clientMs);
        const chunk = chunkAt(clientMs);
        const sglangActions = sglangActionsForEventId(chunk?.event_id);
        const prompt = promptAt(clientMs);
        inspector.hidden = false;
        videoShell?.classList.add("is-inspecting");
        if (inspectorTime) inspectorTime.textContent = formatReplayClientMs(clientMs);
        if (inspectorUserKeys) inspectorUserKeys.textContent = actionText(userActions);
        if (inspectorSglangKeys) inspectorSglangKeys.textContent = actionText(sglangActions);
        if (inspectorChunk) {
          inspectorChunk.textContent = chunk
            ? "chunk #" + (chunk.chunk_index ?? "-") + " · event #" + (chunk.event_id ?? "-")
            : "-";
        }
        if (inspectorPrompt) {
          inspectorPrompt.textContent = (prompt.kind || "prompt") + " " + formatReplayClientMs(Number(prompt.client_ms || 0)) + "\\n" + (prompt.prompt || "-");
        }
        if (inspectorEvents) inspectorEvents.textContent = eventSummaryAt(clientMs);
        if (inspectorImageMeta) inspectorImageMeta.textContent = referenceImageText();
        if (inspectorImage) {
          if (referenceSrc) {
            inspectorImage.src = referenceSrc;
            inspectorImage.classList.add("has-image");
          } else {
            inspectorImage.removeAttribute("src");
            inspectorImage.classList.remove("has-image");
          }
        }
      }

      function syncReplayControls() {
        const clientMs = recordingStartMs + video.currentTime * 1000;
        const active = replayActionsAt(clientMs);
        if (activeText) {
          const labels = Array.from(active).sort().map((action) => action.toUpperCase());
          activeText.textContent = labels.length ? "input " + labels.join(" + ") : "input idle";
        }
        if (!video.paused && !video.ended) requestAnimationFrame(syncReplayControls);
      }

      ["loadedmetadata", "timeupdate", "seeked", "play", "pause"].forEach((eventName) => {
        video.addEventListener(eventName, syncReplayControls);
      });
      video.addEventListener("mousemove", (event) => {
        positionReplayInspector(event);
        inspectReplayAt(replayClientMsFromPointer(event));
      });
      video.addEventListener("mouseleave", () => {
        if (inspector) inspector.hidden = true;
        videoShell?.classList.remove("is-inspecting");
        syncReplayControls();
      });
      syncReplayControls();
    })();
  </script>
</body>
</html>`;
}

function formatReplayMs(value) {
  const ms = Number(value || 0);
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
}

function escapeHtmlText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeHtmlAttribute(value) {
  return escapeHtmlText(value)
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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
  if (selectedPlaybackMode() === "timeline") return;
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

function drawFrame(image, { close = true, markRendered = true } = {}) {
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
  ctx.imageSmoothingQuality = "medium";
  ctx.drawImage(drawSource, 0, 0, sourceWidth, sourceHeight);
  if (markRendered) renderedPreviewFrames += 1;
  setPreviewState("live");
  if (close && !(image instanceof ImageData)) image.close?.();
}

function renderLoop(now) {
  renderLoopSamples.push(now);
  renderLoopSamples = renderLoopSamples.filter((t) => now - t < 1000);
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
    recordChunkFirstRendered(item.chunk, {
      render_loop: true,
      display_lag_ms: lastDisplayLagMs,
      decode_ms: item.decodeMs || lastDecodeMs,
    });
    updateStats();
  } else if (decision.action === "hold") {
    updateStats();
  }
  scheduleRenderLoop();
}

function scheduleRenderLoop() {
  if (
    document.visibilityState !== "hidden" &&
    typeof window.requestAnimationFrame === "function"
  ) {
    window.requestAnimationFrame(renderLoop);
    return;
  }
  const timerFps = Math.min(
    MAX_RENDER_TIMER_FPS,
    Math.max(MIN_RENDER_TIMER_FPS, previewPlaybackTargetFps() * 2),
  );
  window.setTimeout(() => renderLoop(performance.now()), 1000 / timerFps);
}

async function readFirstFrame() {
  const file = $("firstFrame").files[0];
  if (file) return new Uint8Array(await file.arrayBuffer());
  if (selectedReferenceBytes) return selectedReferenceBytes;
  if (selectedReferenceUrl) {
    selectedReferenceBytes = await fetchReferenceBytes(selectedReferenceUrl);
    return selectedReferenceBytes;
  }
  return undefined;
}

async function fetchReferenceBytes(url) {
  try {
    const response = await fetch(url, { cache: "force-cache", mode: "cors" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (!bytes.byteLength) {
      throw new Error("empty image");
    }
    return bytes;
  } catch (error) {
    throw new Error(
      `reference image fetch failed: ${error.message || String(error)}`
    );
  }
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
    previewCtx.fillStyle = "#11140f";
    previewCtx.fillRect(0, 0, preview.width, preview.height);
    previewCtx.fillStyle = "#8c9288";
    previewCtx.font = "14px ui-sans-serif, Avenir Next, Helvetica Neue, sans-serif";
    previewCtx.textAlign = "center";
    previewCtx.textBaseline = "middle";
    previewCtx.fillText("reference image unavailable", preview.width / 2, preview.height / 2);
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
  recordTrajectoryEvent(expectedClose ? "session_close_requested" : "session_abort_requested", {
    reason,
    clear_frames: clearFrames,
  });
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
  socket.close(expectedClose ? 1000 : 4000, reason.slice(0, 120));
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
    currentTrace = createClientTrace();
    traceHttpClient?.reset(currentTrace.traceId, $("serverUrl").value);
    resetTraceTopology(currentTrace.traceId);
    markClientTrace("client.generate_clicked", {
      generation_mode: selectedGenerationMode(),
      transport: $("transportFormat").value || "raw",
      fps: Number($("fps").value || DEFAULT_TARGET_FPS),
    });
    const generationMode = selectedGenerationMode();
    const continuousT2V = generationMode === "t2v" && $("continuous").checked;
    let firstFrame;
    let numFrames = Number($("numFrames").value);
    if (generationMode === "i2v") {
      if (!$("firstFrame").files[0] && !selectedReferenceBytes && !selectedReferenceUrl) {
        await setPresetReference(presets[0]);
      }
      firstFrame = await readFirstFrame();
      if (!firstFrame) {
        setStatus("Pick a reference", "error");
        setPreviewState("idle");
        addHistory("reference image required for I2V");
        $("connectBtn").disabled = false;
        return;
      }
    } else {
      numFrames = continuousT2V ? undefined : readT2VNumFrames();
    }
    const previewTransportParams = readPreviewTransportParams();
    const frameInterpolationParams = readFrameInterpolationParams();
    const superResolutionParams = readSuperResolutionParams();
    const init = compact({
      type: "init",
      generation_mode: generationMode,
      model: $("model").value,
      prompt: $("prompt").value,
      size: $("size").value,
      fps: Number($("fps").value || DEFAULT_TARGET_FPS),
      num_frames: continuousT2V ? undefined : numFrames,
      seed: Number($("seed").value),
      num_inference_steps: Number($("steps").value),
      guidance_scale: Number($("guidance").value),
      realtime_causal_sink_size: readOptionalInteger("sinkSize"),
      realtime_causal_kv_cache_num_frames: readOptionalInteger("windowFrames"),
      max_chunks: generationMode === "t2v" || $("continuous").checked
        ? undefined
        : 1,
      trace_id: currentTrace.traceId,
      first_frame: firstFrame,
      ...previewTransportParams,
      ...frameInterpolationParams,
      ...superResolutionParams,
    });
    const referenceImage = await createReferenceImageMeta(firstFrame);
    beginSessionArtifact(init, referenceImage);
    if (currentSessionArtifact && currentTrace) {
      currentSessionArtifact.trace_id = currentTrace.traceId;
    }
    document.activeElement?.blur?.();
    canvas.tabIndex = 0;
    canvas.focus();
    const socket = new WebSocket(traceWebSocketUrl($("serverUrl").value));
    ws = socket;
    socket.binaryType = "arraybuffer";
    socketHadError = false;
    socketCloseExpected = false;
    socketServerError = "";
    socket.onopen = () => {
      if (epoch !== streamEpoch) return;
      markClientTrace("client.ws_open", {
        url: traceWebSocketUrl($("serverUrl").value),
      });
      recordTrajectoryEvent("socket_open", { url: traceWebSocketUrl($("serverUrl").value) });
      const initPayload = pack(init);
      socket.send(initPayload);
      markClientTrace("client.init_sent", {
        generation_mode: generationMode,
        num_frames: init.num_frames,
        has_reference_image: Boolean(referenceImage),
        payload_bytes: initPayload.byteLength,
      });
      recordTrajectoryEvent("init_sent", {
        generation_mode: generationMode,
        num_frames: init.num_frames,
        has_reference_image: Boolean(referenceImage),
      });
      setStatus("Starting", "live");
      const source = generationMode === "t2v"
        ? `${numFrames} frames from text`
        : selectedReferenceLabel || "uploaded reference";
      addHistory(`${generationMode.toUpperCase()} session started · ${source}`);
    };
    socket.onclose = (event) => {
      if (epoch !== streamEpoch) return;
      if (ws === socket) ws = null;
      markClientTrace("client.ws_close", {
        code: event.code,
        reason: event.reason || "",
      });
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
      recordTrajectoryEvent("socket_close", {
        code: event.code,
        reason: event.reason || "",
        normal_close: normalClose,
        expected_close: socketCloseExpected,
      });
      void traceHttpClient?.flushClientEvents().catch(() => {});
      if (!renderedPreviewFrames) setPreviewState("idle");
      socketCloseExpected = false;
    };
    socket.onerror = () => {
      if (epoch !== streamEpoch) return;
      markClientTrace("client.ws_error");
      recordTrajectoryEvent("socket_error", { ready_state: socket.readyState });
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
    const receivedAt = performance.now();
    const message = unpack(new Uint8Array(data));
    message.__received_at = receivedAt;
    if (message.type === "error") {
      markClientTrace("client.server_error_received", {
        payload_bytes: data.byteLength || data.size || 0,
      });
      socketServerError = message.content || "unknown";
      setStatus(socketServerError, "error");
      addHistory(`server error: ${socketServerError}`);
      recordTrajectoryEvent("server_error", { content: socketServerError });
      if (ws && ws.readyState === WebSocket.OPEN) {
        socketCloseExpected = true;
        ws.close(1000, socketServerError.slice(0, 120));
      }
      $("connectBtn").disabled = false;
      if (!renderedPreviewFrames) setPreviewState("idle");
      return;
    }
    if (message.type === "frame_batch") {
      const payload = message.payload;
      delete message.payload;
      markClientTrace("client.frame_batch_received", {
        chunk_index: Number(message.chunk_index || 0),
        event_id: Number(message.event_id || 0),
        content_type: message.content_type || "",
        num_frames: Number(message.num_frames || 0),
        payload_bytes: payload?.byteLength || payload?.size || payload?.length || 0,
      });
      recordFrameBatchReceived(message, payload?.byteLength || payload?.size || payload?.length || 0);
      enqueueDecodeBatch(message, payload, epoch);
      if (!renderedPreviewFrames) setStatus("Receiving", "live");
      return;
    }
    if (message.type === "media_chunk_complete") {
      recordTrajectoryEvent("media_chunk_complete", {
        chunk_index: Number(message.chunk_index || 0),
        event_id: Number(message.event_id || 0),
        num_frames: Number(message.num_frames || 0),
      });
      return;
    }
    pendingHeader = message;
    if (pendingHeader && !renderedPreviewFrames) setStatus("Receiving", "live");
    return;
  }
  const header = pendingHeader;
  pendingHeader = null;
  header.__received_at = performance.now();
  markClientTrace("client.frame_batch_received", {
    chunk_index: Number(header.chunk_index || 0),
    event_id: Number(header.event_id || 0),
    content_type: header.content_type || "",
    num_frames: Number(header.num_frames || 0),
    payload_bytes: data.byteLength || data.size || data.length || 0,
  });
  recordFrameBatchReceived(header, data?.byteLength || data?.size || data?.length || 0);
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
  markClientTrace("client.decode_batch_done", {
    chunk_index: Number(header.chunk_index || 0),
    event_id: Number(header.event_id || 0),
    content_type: header.content_type || "",
    num_frames: decodedFrames.length,
    payload_bytes: payloadBytes,
    decode_ms: roundTraceNumber(lastDecodeMs),
  });
  const now = performance.now();
  if (!renderedPreviewFrames && decodedFrames.length) {
    drawFrame(decodedFrames[0].image, { close: false, markRendered: false });
    recordChunkFirstRendered(decodedFrames[0].chunk, {
      initial_preview: true,
      display_lag_ms: now - (decodedFrames[0].receivedAt || now),
      decode_ms: decodedFrames[0].decodeMs || lastDecodeMs,
    });
  }
  // record source frames before preview playback can hold or drop for latency
  recordDecodedFrameBatch(decodedFrames);
  const enqueueResult = playbackController.enqueueDecodedFrames(header, decodedFrames, now);
  closeFrames(enqueueResult.droppedFrames);
  const playback = enqueueResult.snapshot;
  lastSampledEventId = Number(header.event_id || lastSampledEventId);
  updateControlDebugText();
  $("chunkPayloadText").textContent = `${formatBytes(payloadBytes)} · ${chunkFrameCount}f`;
  const realtimeRatio = playback.targetFps > 0
    ? playback.sourceFps / playback.targetFps
    : 0;
  $("theoreticalFpsText").textContent = (
    `${playback.sourceFps.toFixed(1)} fps · ${realtimeRatio.toFixed(2)}x`
  );
  if (enqueueResult.cutover?.latencyMs) {
    const eventLatency = enqueueResult.cutover.latencyMs / 1000;
    $("latencyText").textContent = `${eventLatency.toFixed(1)}s · event`;
  }
  frames += chunkFrameCount;
  bytes += payloadBytes;
  $("payloadMode").textContent = payloadModeLabelFromHeader(header);
  updateOutputSizeFromHeader(header);
  setStatus("Live", "live");
  updateStats();
}

function recordFrameBatchReceived(header, payloadBytes) {
  recordTrajectoryEvent("frame_batch_received", {
    chunk_index: header.chunk_index,
    event_id: header.event_id,
    content_type: header.content_type,
    encoding: header.encoding,
    num_frames: header.num_frames,
    width: header.width,
    height: header.height,
    source_width: header.source_width,
    source_height: header.source_height,
    preview_width: header.preview_width,
    preview_height: header.preview_height,
    payload_bytes: payloadBytes,
  });
}

function recordChunkFirstRendered(chunkIndex, details = {}) {
  if (chunkIndex === undefined || chunkIndex === null) return;
  const key = String(chunkIndex);
  if (renderedTraceChunks.has(key)) return;
  renderedTraceChunks.add(key);
  const event = recordTrajectoryEvent("client.chunk_first_rendered", {
    chunk_index: chunkIndex,
    ...details,
  });
  markClientTrace("client.chunk_first_rendered", {
    chunk_index: Number(chunkIndex || 0),
    display_lag_ms: roundTraceNumber(details.display_lag_ms),
    decode_ms: roundTraceNumber(details.decode_ms),
  });
  if (event && currentSessionArtifact) {
    currentSessionArtifact.first_rendered_chunks.push(event);
    if (currentSessionArtifact.first_rendered_chunks.length > SESSION_ARTIFACT_EVENT_LIMIT) {
      currentSessionArtifact.first_rendered_chunks.splice(
        0,
        currentSessionArtifact.first_rendered_chunks.length - SESSION_ARTIFACT_EVENT_LIMIT,
      );
    }
  }
}

function sendEvent(kind, payload, historyText = null) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addHistory(`${historyText || `${kind} event`} · socket not open`);
    recordTrajectoryEvent(`${kind}_event_dropped`, {
      reason: "socket not open",
      payload,
    });
    return null;
  }
  const eventId = nextEventId++;
  const clientSentPerfMs = performance.now();
  const clientSentEpochMs = Date.now();
  ws.send(pack({
    type: "event",
    kind,
    payload,
    event_id: eventId,
    trace_id: currentTrace?.traceId,
    client_sent_perf_ms: roundTraceNumber(clientSentPerfMs),
    client_sent_epoch_ms: clientSentEpochMs,
  }));
  markClientTrace("client.event_sent", {
    kind,
    event_id: eventId,
    ws_buffered_amount: ws.bufferedAmount,
  });
  lastSentEventId = eventId;
  updateControlDebugText();
  if (kind === "prompt") {
    recordPromptHistory(payload, "prompt_update", eventId);
    recordTrajectoryEvent("prompt_update", { event_id: eventId, prompt: payload });
  } else if (kind === "camera_actions") {
    recordTrajectoryEvent("camera_actions_sent", {
      event_id: eventId,
      payload,
      active_actions: controlStateController
        ? Array.from(controlStateController.activeActions).sort()
        : [],
    });
  } else {
    recordTrajectoryEvent(`${kind}_event_sent`, { event_id: eventId, payload });
  }
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
  $("fps").value = UI_CONFIG.targetFps == null ? preset.fps : DEFAULT_TARGET_FPS;
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

function realtimeServerUrlFromLocation() {
  if (!window.location.host) return "";
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "";
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/v1/realtime_video/generate`;
}

function applyDefaultServerUrl() {
  const current = $("serverUrl").value.trim();
  const locationServerUrl = realtimeServerUrlFromLocation();
  if (!locationServerUrl) return;
  if (current.includes("127.0.0.1") || current.includes("localhost")) {
    $("serverUrl").value = locationServerUrl;
  }
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
  };
  const baseSize = parseSizeValue($("size").value);
  params.realtime_preview_max_width = previewMaxWidthForSize(baseSize);
  if (outputFormat === "webp" || outputFormat === "jpeg") {
    params.output_compression = outputQuality;
    if ($("superResolution").checked && $("frameInterpolation").checked) {
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

function previewMaxWidthForSize(baseSize) {
  const baseWidth = Number(baseSize?.width || 0);
  if (!baseWidth) return DEFAULT_PREVIEW_MAX_WIDTH;
  return Math.max(
    DEFAULT_PREVIEW_MAX_WIDTH,
    Math.min(baseWidth, MAX_AUTO_PREVIEW_WIDTH),
  );
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
  const requestSize = parseSizeValue($("size").value);
  const frameWidth = Number(header.width || 0);
  const frameHeight = Number(header.height || 0);
  const sourceWidth = Number(header.source_width || requestSize?.width || frameWidth || 0);
  const sourceHeight = Number(header.source_height || requestSize?.height || frameHeight || 0);
  if (!sourceWidth || !sourceHeight) return;
  updateOutputSizeText(sourceWidth, sourceHeight);
  const previewWidth = Number(header.preview_width || 0) || (
    frameWidth && frameWidth !== sourceWidth ? frameWidth : 0
  );
  const previewHeight = Number(header.preview_height || 0) || (
    frameHeight && frameHeight !== sourceHeight ? frameHeight : 0
  );
  if (previewWidth && previewHeight) {
    $("outputSizeText").textContent += ` · preview ${previewWidth}x${previewHeight}`;
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

function payloadModeLabelFromHeader(header) {
  if (header?.encoding) return header.encoding;
  const label = shortPayloadMode(header?.content_type || "");
  return label || selectedTransportLabel();
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
    const thumb = document.createElement("img");
    thumb.className = "preset-thumb";
    thumb.src = preset.referenceUrl;
    thumb.alt = "";
    thumb.loading = "lazy";
    thumb.onerror = () => thumb.replaceWith(createPresetThumbFallback(preset));
    const title = document.createElement("b");
    title.textContent = preset.name;
    const meta = document.createElement("span");
    meta.textContent = `${preset.source} · ${preset.size} · ${preset.fps}fps`;
    btn.append(thumb, title, meta);
    btn.onclick = () => applyPreset(preset).catch(showError);
    $("presetList").appendChild(btn);
  });
}

function createPresetThumbFallback(preset) {
  const fallback = document.createElement("span");
  fallback.className = "preset-thumb preset-thumb-fallback";
  fallback.textContent = preset.name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0] || "")
    .join("")
    .toUpperCase();
  fallback.title = `${preset.name} reference image unavailable`;
  return fallback;
}

async function applyQueryParams() {
  const params = new URLSearchParams(window.location.search);
  const server = params.get("server");
  if (server) $("serverUrl").value = server;
  else applyDefaultServerUrl();
  const model = params.get("model");
  if (model) $("model").value = model;
  const generationMode = String(params.get("mode") || "").toLowerCase();
  if (ENABLED_GENERATION_MODES.includes(generationMode)) {
    $("generationMode").value = generationMode;
    updateGenerationModeUi();
  }
  $("transportFormat").value = params.get("transport") || DEFAULT_PREVIEW_OUTPUT_FORMAT;
  $("transportQuality").value = params.get("quality") || String(DEFAULT_PREVIEW_OUTPUT_QUALITY);
  const playbackParam = params.get("playback");
  if (playbackParam === "live" || playbackParam === "timeline") {
    $("playbackMode").value = playbackParam;
  }
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
  syncPlaybackMode({ addToHistory: false });

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
  const readU32 = () => (
    (buf[i++] * 16777216) + (buf[i++] << 16) + (buf[i++] << 8) + buf[i++]
  );
  const readI32 = () => {
    const value = readU32();
    return value > 0x7fffffff ? value - 0x100000000 : value;
  };
  const readU64 = () => {
    const hi = readU32();
    const lo = readU32();
    const value = BigInt(hi) * 4294967296n + BigInt(lo);
    return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value.toString();
  };
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
    if (b === 0xce) return readU32();
    if (b === 0xcf) return readU64();
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
    if (b === 0xc6) return readBin(readU32());
    if (b === 0xd2) return readI32();
    if (b === 0xd3) {
      const hi = readI32();
      const lo = readU32();
      return hi * 4294967296 + lo;
    }
    if (b === 0xdc) return Array.from({ length: (buf[i++] << 8) | buf[i++] }, read);
    if (b === 0xdd) return Array.from({ length: readU32() }, read);
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

applyRuntimeUiConfig();
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
scheduleRenderLoop();
renderTraceTopology();
updateRecordButton();
updateRecordFolderButton();
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
$("recordFolderBtn").onclick = () => {
  chooseRecordingDirectory().catch((error) => {
    addHistory(error.message || "record folder selection failed");
  });
};
$("firstFrame").onchange = () => drawReferencePreview($("firstFrame").files[0]);
$("generationMode").addEventListener("change", updateGenerationModeUi);
$("continuous").addEventListener("change", updateGenerationModeUi);
$("numFrames").addEventListener("input", updateT2VFrameHint);
$("size").addEventListener("input", () => updateOutputSizeText());
$("fps").addEventListener("input", () => {
  syncPlaybackTargetFps();
  updateT2VFrameHint();
});
$("playbackMode").addEventListener("change", () => syncPlaybackMode());
$("superResolution").addEventListener("change", updateSuperResolutionControls);
$("upscalingScale").addEventListener("change", () => updateOutputSizeText());
$("frameInterpolation").addEventListener("change", () => {
  tunePreviewQualityForPostprocess();
  syncPlaybackTargetFps();
});
$("superResolution").addEventListener("change", tunePreviewQualityForPostprocess);
$("previewScale").addEventListener("input", () => setPreviewScale($("previewScale").value));
canvas.addEventListener("pointerdown", () => canvas.focus({ preventScroll: true }));
$("serverUrl").addEventListener("change", () => {
  queryServerModelInfo({ applyPresetForModel: true }).catch(showError);
});
document.querySelectorAll("[data-workspace-view]").forEach((button) => {
  button.addEventListener("click", () => setWorkspaceView(button.dataset.workspaceView));
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
    if (controlStateController.setAction(action, true)) {
      recordTrajectoryEvent("control_button_down", { action });
    }
  });
  ["pointerup", "pointercancel", "pointerleave", "blur"].forEach((eventName) => {
    btn.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (controlStateController.setAction(action, false)) {
        recordTrajectoryEvent("control_button_up", { action, event: eventName });
      }
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
    this.stateHeartbeatTimer = 0;
  }

  reset({ sendRelease = false } = {}) {
    const hadActions = this.activeActions.size > 0;
    this.activeActions.clear();
    this.pendingTransitions = [];
    this.clearFlushTimer();
    this.clearStateHeartbeatTimer();
    this.updateButtons();
    if (sendRelease && hadActions) {
      this.enqueueTransition();
    }
  }

  setAction(action, active) {
    const hadAction = this.activeActions.has(action);
    if (active === hadAction) return false;
    if (active) {
      this.activeActions.add(action);
    } else {
      this.activeActions.delete(action);
    }
    this.updateButtons();
    this.enqueueTransition();
    if (this.activeActions.size) this.scheduleStateHeartbeat();
    else this.clearStateHeartbeatTimer();
    return true;
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

  scheduleStateHeartbeat() {
    if (this.stateHeartbeatTimer || !this.activeActions.size) return;
    this.stateHeartbeatTimer = window.setTimeout(() => {
      this.stateHeartbeatTimer = 0;
      if (!this.activeActions.size) return;
      sendCameraControlTransitions([{
        actions: Array.from(this.activeActions).sort(),
        clientTsMs: Math.round(performance.now()),
      }]);
      this.scheduleStateHeartbeat();
    }, CONTROL_HELD_STATE_HEARTBEAT_MS);
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
    updateControlDebugText();
  }

  sameActions(left, right) {
    return left.length === right.length && left.every((item, idx) => item === right[idx]);
  }

  clearFlushTimer() {
    if (!this.flushTimer) return;
    window.clearTimeout(this.flushTimer);
    this.flushTimer = 0;
  }

  clearStateHeartbeatTimer() {
    if (!this.stateHeartbeatTimer) return;
    window.clearTimeout(this.stateHeartbeatTimer);
    this.stateHeartbeatTimer = 0;
  }
}

const CONTROL_ACTION_META_KEYS = Object.keys(CONTROL_ACTION_META);
controlStateController = new ControlStateController();
updateControlDebugText();
window.setInterval(() => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const eventId = nextEventId++;
  ws.send(pack({
    type: "event",
    kind: "heartbeat",
    payload: {
      active_actions: Array.from(controlStateController.activeActions).sort(),
    },
    event_id: eventId,
    trace_id: currentTrace?.traceId,
    client_sent_perf_ms: roundTraceNumber(performance.now()),
    client_sent_epoch_ms: Date.now(),
  }));
}, SESSION_HEARTBEAT_MS);

document.addEventListener("keydown", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  if (event.repeat) {
    recordTrajectoryEvent("key_repeat_ignored", {
      key: event.key,
      code: event.code,
      action,
    });
    return;
  }
  if (controlStateController.setAction(action, true)) {
    recordTrajectoryEvent("key_down", {
      key: event.key,
      code: event.code,
      action,
    });
  }
});

document.addEventListener("keyup", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  if (controlStateController.setAction(action, false)) {
    recordTrajectoryEvent("key_up", {
      key: event.key,
      code: event.code,
      action,
    });
  }
});

window.addEventListener("blur", () => {
  controlStateController.releaseAll();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    controlStateController.releaseAll();
  }
});

window.__sglangRealtimeDebug = () => ({
  activeActions: controlStateController
    ? Array.from(controlStateController.activeActions).sort()
    : [],
  bytes,
  decodeInProgress,
  decodeQueueLength: decodeQueue.length,
  droppedDecodeFrames,
  frames,
  lastDecodeMs,
  lastDisplayLagMs,
  lastSampledEventId,
  lastSentEventId,
  pendingDecodeBatches,
  pendingHeader: Boolean(pendingHeader),
  playback: playbackController.snapshot(),
  renderedFps: fpsSamples.length,
  renderedPreviewFrames,
  renderLoopFps: renderLoopSamples.length,
  recordingArtifact: recordingArtifact ? {
    events: recordingArtifact.events.length,
    firstRenderedChunks: recordingArtifact.first_rendered_chunks.length,
    promptHistory: recordingArtifact.prompt_history.length,
    traceId: recordingArtifact.trace_id,
  } : null,
  currentSessionArtifact: currentSessionArtifact ? {
    events: currentSessionArtifact.events.length,
    firstRenderedChunks: currentSessionArtifact.first_rendered_chunks.length,
    promptHistory: currentSessionArtifact.prompt_history.length,
    traceId: currentSessionArtifact.trace_id,
  } : null,
  socketBufferedAmount: ws ? ws.bufferedAmount : 0,
  socketCloseExpected,
  socketHadError,
  socketReadyState: ws ? ws.readyState : null,
  socketServerError,
  status: $("statusText").textContent,
  streamEpoch,
  visibilityState: document.visibilityState,
});
