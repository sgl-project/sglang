const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const webuiDir = __dirname;
const indexHtml = fs.readFileSync(path.join(webuiDir, "index.html"), "utf8");
const appJs = fs.readFileSync(path.join(webuiDir, "app.js"), "utf8");

assert.match(
  indexHtml,
  /id="tracePaneButton"/,
  "WebUI should expose a Trace workspace tab next to the preview",
);
assert.match(
  indexHtml,
  /id="traceTopology"/,
  "WebUI should have a live topology container",
);
assert.match(
  indexHtml,
  /trace_topology\.js\?v=/,
  "WebUI should load the trace topology reducer before app.js",
);
assert.match(
  appJs,
  /const traceTopologyApi = window\.SGLangRealtimeTraceTopology \|\| \{\};/,
  "app.js should connect to the trace topology reducer",
);
assert.match(
  appJs,
  /function currentTracePayload\(/,
  "generate init payload should include the client trace envelope",
);
assert.match(
  appJs,
  /function traceWebSocketUrl\(/,
  "WebSocket URL should carry trace_id so server trace events join the same request",
);
assert.match(
  appJs,
  /message\.type === "trace_event"/,
  "client should consume server trace_event messages",
);
assert.match(
  appJs,
  /message\.type === "trace_events"/,
  "client should consume batched server trace events",
);
assert.match(
  appJs,
  /recordTraceTopologyEvent\(\{ event: "server\.chunk_complete", \.\.\.stats \}/,
  "chunk stats should feed both dump artifacts and live trace topology",
);
assert.match(
  appJs,
  /recordTrajectoryEvent\("trace_event", \{ trace: traceEvent \}\)/,
  "dump artifacts should retain the same realtime trace events shown in the topology",
);
assert.match(
  appJs,
  /trace_id: currentTrace\?\.traceId/,
  "runtime input events should include the current trace id",
);
assert.match(
  appJs,
  /client_trace: currentTracePayload\(\)/,
  "init request should include recent client-side trace marks",
);
assert.match(
  appJs,
  /currentSessionArtifact\.trace_id = currentTrace\.traceId/,
  "dump artifact and live trace should use the same trace id",
);

console.log("realtime trace dump integration ok");
