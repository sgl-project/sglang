const assert = require("node:assert/strict");
const fs = require("node:fs");
const { parseArgs, summarize, traceHttpUrl } = require("./browser_probe.cjs");

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
assert.equal(
  parseArgs(["node", "probe", "--url", "http://example.test", "--output", "out.json"]).continuous,
  false,
);
assert.equal(
  parseArgs([
    "node", "probe", "--url", "http://example.test", "--output", "out.json",
    "--continuous", "true",
  ]).continuous,
  true,
);
assert.equal(
  parseArgs([
    "node", "probe", "--url", "http://example.test", "--output", "out.json",
    "--send-action", "false",
  ]).sendAction,
  false,
);
const probeSource = fs.readFileSync(require.resolve("./browser_probe.cjs"), "utf8");
assert.match(
  probeSource,
  /if \(args\.continuous\)[\s\S]+await page\.click\("#stopBtn"\)/,
);
assert.match(probeSource, /browser did not reach requested frames/);
assert.match(probeSource, /pageErrors/);
assert.match(probeSource, /historyList/);
assert.match(probeSource, /limit: "100"/);
assert.match(probeSource, /timeout: 30000/);
console.log("browser probe unit tests passed");
