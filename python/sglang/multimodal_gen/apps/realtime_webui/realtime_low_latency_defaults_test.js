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
  appJs,
  /const DEFAULT_T2V_NUM_FRAMES\s*=\s*9;/,
  "T2V should default to a short user-editable 9-frame request",
);
assert.doesNotMatch(
  appJs,
  /configuredNumber\("t2vDefaultNumFrames"/,
  "deployment runtime config should not override the user-visible T2V Frames default",
);
assert.match(
  appJs,
  /let savedT2VNumFrames\s*=\s*String\(DEFAULT_T2V_NUM_FRAMES\);/,
  "T2V should remember the user's Frames value across mode switches",
);
assert.match(
  appJs,
  /let savedT2VContinuous\s*=\s*true;/,
  "T2V should start in continuous mode so Generate keeps the realtime session open",
);
assert.match(
  appJs,
  /savedT2VNumFrames\s*=\s*\$\("numFrames"\)\.value;/,
  "leaving T2V should save the user-edited Frames value",
);
assert.match(
  appJs,
  /\$\("numFrames"\)\.value\s*=\s*savedT2VNumFrames;/,
  "entering T2V should restore the user-edited Frames value instead of resetting it",
);
assert.match(
  indexHtml,
  /<script src="\.\/runtime-config\.js"><\/script>/,
  "webui should load the deployment profile before app.js",
);
assert.match(
  indexHtml,
  /<input id="size" value="1280x704" \/>/,
  "webui should default to the 720p-ish realtime profile",
);
assert.match(
  indexHtml,
  /<span class="stage-stat">output <b id="outputSizeText">1280x704<\/b><\/span>/,
  "webui should show the default output size before the first server response",
);
assert.match(
  indexHtml,
  /<option value="adaptive">Adaptive \(buffered, fast input\)<\/option>/,
  "webui should keep adaptive playback available",
);
assert.match(
  indexHtml,
  /<option value="smooth_timeline" selected>Smooth timeline \(catch-up, no skip\)<\/option>/,
  "webui should default to the no-skip catch-up timeline mode for smoother playback",
);
assert.match(
  appJs,
  /playbackParam === "live" \|\| playbackParam === "timeline" \|\| playbackParam === "adaptive" \|\| playbackParam === "smooth_timeline"/,
  "webui should accept playback=adaptive and playback=smooth_timeline from the URL",
);
assert.match(
  appJs,
  /const preservesTimeline[\s\S]*selectedPlaybackMode\(\) === "timeline"[\s\S]*selectedPlaybackMode\(\) === "smooth_timeline"/,
  "webui should avoid browser decode drops in smooth timeline except at the byte cap",
);
assert.match(
  appJs,
  /const DECODE_QUEUE_SECONDS\s*=\s*configuredNumber\("decodeQueueSeconds", 5\);/,
  "webui should default to a browser-side decode queue that drains the backend websocket",
);
assert.match(
  appJs,
  /const MAX_DECODE_QUEUE_BYTES\s*=\s*configuredNumber\([\s\S]*192 \* 1024 \* 1024/,
  "webui should bound browser-side decode buffering by bytes",
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
  "webui should keep low-latency backlog trimming enabled",
);
assert.match(
  appJs,
  /holdForTargetLead:\s*true/,
  "webui should hold a small jitter lead before rendering public websocket media",
);
assert.match(
  appJs,
  /targetLeadChunkRatio:\s*0\.75/,
  "24 fps playback should keep enough jitter lead for sub-24fps backend delivery",
);
assert.match(
  appJs,
  /minTargetLeadMs:\s*360/,
  "24 fps playback should avoid chasing a too-small buffer when backend delivery is bursty",
);
assert.match(
  appJs,
  /maxTargetLeadMs:\s*900/,
  "24 fps playback should trade a bounded sub-second lead for smoother display",
);
assert.match(
  appJs,
  /maxDeliveryLeadBoostMs:\s*360/,
  "webui should bound adaptive jitter buffering",
);
assert.match(
  appJs,
  /smoothTimelinePlaybackRateMax:\s*2\.5/,
  "smooth timeline should catch up quickly without dropping old backlog",
);
assert.match(
  appJs,
  /lowLatencyMaxLeadFrames:\s*12/,
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
  "webui should not ask the backend to pace output at the playback fps",
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
