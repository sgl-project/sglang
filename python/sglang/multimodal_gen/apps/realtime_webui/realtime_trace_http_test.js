const assert = require("node:assert/strict");
const {
  RealtimeTraceHttpClient,
  traceHttpBaseUrl,
} = require("./trace_transport.js");
const fs = require("node:fs");

async function flushesClientEventsInOneHttpBatch() {
  const requests = [];
  const client = new RealtimeTraceHttpClient({
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, json: async () => ({ accepted: 2 }) };
    },
  });
  client.reset("abc", "ws://example.test/v1/realtime_video/generate");
  client.enqueueClientEvent({ name: "client.ws_open", seq: 1 });
  client.enqueueClientEvent({ name: "client.init_sent", seq: 2 });

  await client.flushClientEvents();

  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "http://example.test/v1/realtime_video/traces/abc/client-events");
  assert.equal(requests[0].options.method, "POST");
  assert.deepEqual(JSON.parse(requests[0].options.body).events.map((event) => event.seq), [1, 2]);
}

async function pollsIncrementallyOnlyWhenRequested() {
  const requests = [];
  const received = [];
  const client = new RealtimeTraceHttpClient({
    fetchImpl: async (url) => {
      requests.push(url);
      return {
        ok: true,
        json: async () => ({
          events: [{ event: "server.chunk_complete", trace_seq: 7 }],
          next_cursor: 7,
        }),
      };
    },
    onServerEvents: (events) => received.push(...events),
  });
  client.reset("abc", "wss://example.test/v1/realtime_video/generate");

  await client.pollOnce();
  await client.pollOnce();

  assert.equal(requests[0], "https://example.test/v1/realtime_video/traces/abc?after=0&limit=220");
  assert.equal(requests[1], "https://example.test/v1/realtime_video/traces/abc?after=7&limit=220");
  assert.equal(received.length, 2);
}

assert.equal(
  traceHttpBaseUrl("ws://127.0.0.1:30000/v1/realtime_video/generate"),
  "http://127.0.0.1:30000/v1/realtime_video/traces",
);

const appSource = fs.readFileSync(require.resolve("./app.js"), "utf8");
assert.equal(appSource.includes('kind: "client_trace"'), false);
assert.equal(appSource.includes('message.type === "trace_events"'), false);

Promise.resolve()
  .then(flushesClientEventsInOneHttpBatch)
  .then(pollsIncrementallyOnlyWhenRequested)
  .then(() => console.log("realtime trace HTTP transport ok"));
