const assert = require("assert");
const fs = require("fs");
const path = require("path");

const webuiDir = __dirname;
const appJs = fs.readFileSync(path.join(webuiDir, "app.js"), "utf8");

function extractFunction(name) {
  const start = appJs.indexOf(`function ${name}(`);
  assert.notStrictEqual(start, -1, `${name} should exist`);
  let depth = 0;
  let end = -1;
  for (let i = start; i < appJs.length; i++) {
    const ch = appJs[i];
    if (ch === "{") depth++;
    if (ch === "}") {
      depth--;
      if (depth === 0) {
        end = i + 1;
        break;
      }
    }
  }
  assert.notStrictEqual(end, -1, `${name} body should be complete`);
  return appJs.slice(start, end);
}

const unpack = new Function(`${extractFunction("unpack")}; return unpack;`)();

function utf8(value) {
  return Array.from(new TextEncoder().encode(value));
}

function uint64Bytes(value) {
  let n = BigInt(value);
  const bytes = new Array(8).fill(0);
  for (let i = 7; i >= 0; i--) {
    bytes[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  return bytes;
}

const key = "server_epoch_ms";
const epochMs = 1785496472440;
const payload = new Uint8Array([
  0x81,
  0xa0 | key.length,
  ...utf8(key),
  0xcf,
  ...uint64Bytes(epochMs),
]);

assert.deepStrictEqual(
  unpack(payload),
  { server_epoch_ms: epochMs },
  "trace msgpack payloads should decode uint64 timestamps emitted by Python",
);

assert.doesNotMatch(
  appJs,
  /socket\.close\(expectedClose \? 1000 : 1011/,
  "browser WebSocket.close must not use reserved server close code 1011 from client JS",
);
assert.doesNotMatch(
  appJs,
  /socket\.close\(expectedClose \? 1000 : 1008/,
  "browser WebSocket.close must not use reserved server close code 1008 from client JS",
);
assert.match(
  appJs,
  /socket\.close\(expectedClose \? 1000 : 4000/,
  "browser aborts should use an application-defined close code",
);

console.log("realtime msgpack codec ok");
