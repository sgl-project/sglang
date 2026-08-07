const assert = require("assert");
const fs = require("fs");
const path = require("path");

const webuiDir = __dirname;
const sglangRoot = path.resolve(webuiDir, "../../..");

function read(...parts) {
  return fs.readFileSync(path.join(sglangRoot, ...parts), "utf8");
}

const appJs = read("multimodal_gen/apps/realtime_webui/app.js");
const indexHtml = read("multimodal_gen/apps/realtime_webui/index.html");
const outputAdapterPy = read(
  "multimodal_gen/runtime/entrypoints/openai/realtime/realtime_output_adapter.py",
);

assert.match(
  outputAdapterPy,
  /DEFAULT_REALTIME_OUTPUT_FORMAT\s*=\s*"webp"/,
  "backend realtime transport should default to compressed WebP preview",
);
assert.match(
  outputAdapterPy,
  /ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE\s*=\s*1/,
  "backend should send each encoded preview frame as soon as it is available",
);
assert.match(
  outputAdapterPy,
  /DEFAULT_REALTIME_PREVIEW_MAX_WIDTH\s*=\s*480/,
  "backend realtime preview should downscale by default for browser latency",
);
assert.match(
  outputAdapterPy,
  /normalize_realtime_output_format\(.*?\)/s,
  "backend should normalize omitted realtime_output_format instead of falling back to raw RGB",
);

assert.match(
  appJs,
  /const DEFAULT_TARGET_FPS\s*=\s*configuredNumber\("targetFps", 24\);/,
  "webui should preserve its fallback while accepting a deployment target FPS",
);
assert.match(
  indexHtml,
  /<script src="\.\/runtime-config\.js"><\/script>/,
  "webui should load the deployment profile before app.js",
);
assert.match(
  appJs,
  /const DEFAULT_PREVIEW_OUTPUT_QUALITY\s*=\s*55;/,
  "8-GPU webui profile should favor steady public websocket playback",
);
assert.match(
  appJs,
  /const DEFAULT_PREVIEW_MAX_WIDTH\s*=\s*configuredNumber\("previewMaxWidth", 832\);/,
  "8-GPU webui profile should show the model-native 832px preview by default",
);
assert.match(
  appJs,
  /const MAX_AUTO_PREVIEW_WIDTH\s*=\s*configuredNumber\("maxAutoPreviewWidth", 1280\);/,
  "720p webui sessions should be able to request a 1280px preview without hard-coding 560px",
);
assert.match(
  appJs,
  /function previewMaxWidthForSize\(baseSize\)[\s\S]*Math\.min\(baseWidth, MAX_AUTO_PREVIEW_WIDTH\)/,
  "webui preview width should scale with the requested Size field",
);
assert.match(
  appJs,
  /params\.realtime_preview_max_width\s*=\s*previewMaxWidthForSize\(baseSize\);[\s\S]*if \(outputFormat === "webp" \|\| outputFormat === "jpeg"\)/,
  "webui should send the Size-derived preview width for raw and encoded transports",
);
assert.match(
  appJs,
  /lowLatencyPlayback:\s*true/,
  "webui should render at the requested cadence without accumulating a smoothing queue",
);
assert.match(
  appJs,
  /holdForTargetLead:\s*false/,
  "webui should start from the first decoded frame",
);
assert.match(
  appJs,
  /targetLeadChunkRatio:\s*0\.55/,
  "24 fps playback should keep enough jitter lead for sub-24fps backend delivery",
);
assert.match(
  appJs,
  /minTargetLeadMs:\s*260/,
  "24 fps playback should avoid chasing a too-small buffer when backend delivery is bursty",
);
assert.match(
  appJs,
  /maxTargetLeadMs:\s*640/,
  "24 fps playback should trade a bounded sub-second lead for smoother display",
);
assert.match(
  appJs,
  /maxDeliveryLeadBoostMs:\s*240/,
  "webui should bound adaptive jitter buffering",
);
assert.match(
  appJs,
  /lowLatencyMaxLeadFrames:\s*8/,
  "live playback should retain a small 24 fps frame cushion before dropping stale frames",
);
assert.match(
  appJs,
  /requestAnimationFrame\(renderLoop\)/,
  "visible playback should render on the browser refresh clock",
);
assert.doesNotMatch(
  appJs,
  /realtime_output_pacing:\s*true/,
  "webui should not ask the backend to pace output in low-latency mode",
);
assert.match(
  appJs,
  /realtime_output_pacing:\s*false/,
  "webui should explicitly request immediate chunk delivery",
);
assert.match(
  indexHtml,
  /id="fps"[^>]*value="24"/,
  "HTML fallback defaults should match app.js target fps",
);
assert.match(
  indexHtml,
  /id="transportQuality"[^>]*value="55"/,
  "HTML fallback transport quality should keep 560px preview payloads smooth on public websocket",
);

console.log("realtime low-latency defaults ok");
