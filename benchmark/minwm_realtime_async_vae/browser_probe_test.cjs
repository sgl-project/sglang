const assert = require("node:assert/strict");
const { summarize, traceHttpUrl } = require("./browser_probe.cjs");

assert.deepEqual(summarize([80, 100, 120, 200]), {
  count: 4,
  min: 80,
  p50: 110,
  p95: 188,
  max: 200,
  mean: 125,
});
assert.equal(
  traceHttpUrl("http://example.test/app", "trace-a"),
  "http://example.test/v1/realtime_video/traces/trace-a",
);
console.log("browser probe unit tests passed");
