const assert = require("assert");
const fs = require("fs");
const path = require("path");

const appJs = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");

assert.match(
  appJs,
  /url\.searchParams\.set\("user_id", browserUserId\)/,
  "webui should carry a stable browser identity for per-user admission",
);
assert.match(
  appJs,
  /const CONTROL_HELD_STATE_HEARTBEAT_MS = 100;/,
  "held actions should be refreshed every 100ms",
);
assert.match(
  appJs,
  /actions: Array\.from\(this\.activeActions\)\.sort\(\)/,
  "each held-key refresh should send the complete key state",
);
assert.match(
  appJs,
  /kind: "heartbeat"/,
  "idle connected clients should keep their lease alive explicitly",
);

console.log("realtime multi-user lifecycle ok");
