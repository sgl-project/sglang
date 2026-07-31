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
  /ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE\s*=\s*3/,
  "backend should send tiny encoded preview batches to reduce websocket message overhead",
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
  /const DEFAULT_TARGET_FPS\s*=\s*configuredNumber\("targetFps", 16\);/,
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
  /const DEFAULT_PREVIEW_MAX_WIDTH\s*=\s*560;/,
  "8-GPU webui profile should request a clearer preview width without overwhelming public websocket transport",
);
assert.match(
  appJs,
  /holdForTargetLead:\s*true/,
  "webui should keep a tiny playback lead to absorb public websocket jitter",
);
assert.match(
  appJs,
  /maxDeliveryLeadBoostMs:\s*220/,
  "webui should adapt its tiny playback lead when frame delivery is bursty",
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
  /id="fps"[^>]*value="16"/,
  "HTML fallback defaults should match app.js target fps",
);
assert.match(
  indexHtml,
  /id="transportQuality"[^>]*value="55"/,
  "HTML fallback transport quality should keep 560px preview payloads smooth on public websocket",
);

console.log("realtime low-latency defaults ok");
