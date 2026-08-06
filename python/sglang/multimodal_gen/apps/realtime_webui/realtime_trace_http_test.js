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

async function preservesLastAggregateAcrossTransientFailure() {
  let call = 0;
  const aggregates = [];
  const client = new RealtimeTraceHttpClient({
    fetchImpl: async () => {
      call += 1;
      if (call === 2) throw new Error("temporary network failure");
      return {
        ok: true,
        json: async () => ({
          trace_id: "abc",
          stale: false,
          observed_at: "2026-08-06T00:00:00Z",
          window: { seconds: 300 },
          stages: [{ id: "denoise", count: 2, p50_ms: 40, p95_ms: 60 }],
          events: [],
          next_cursor: 0,
        }),
      };
    },
    onAggregate: (result) => aggregates.push(result),
  });
  client.reset("abc", "wss://example.test/v1/realtime_video/generate");

  const first = await client.pollOnce();
  const second = await client.pollOnce();

  assert.equal(first.stale, false);
  assert.equal(second.stale, true);
  assert.equal(second.stale_reason, "client_query_failed");
  assert.deepEqual(second.stages, first.stages);
  assert.equal(aggregates.length, 2);
  assert.equal(aggregates[1].stages[0].p95_ms, 60);
}

async function previewDoesNotPollAndTraceDoes() {
  let requests = 0;
  const client = new RealtimeTraceHttpClient({
    fetchImpl: async () => {
      requests += 1;
      return {
        ok: true,
        json: async () => ({ events: [], next_cursor: 0, stages: [] }),
      };
    },
  });
  client.reset("abc", "wss://example.test/v1/realtime_video/generate");

  client.setActive(false, 10);
  await new Promise((resolve) => setTimeout(resolve, 25));
  assert.equal(requests, 0);

  client.setActive(true, 10);
  await new Promise((resolve) => setTimeout(resolve, 25));
  client.setActive(false);
  assert.ok(requests >= 1);
}

async function resetKeepsVisibleTracePollingAndIgnoresOldResponses() {
  const resolvers = [];
  const aggregates = [];
  const client = new RealtimeTraceHttpClient({
    fetchImpl: (url) =>
      new Promise((resolve) => {
        resolvers.push({ url, resolve });
      }),
    onAggregate: (result) => aggregates.push(result.trace_id),
  });
  client.reset("old", "wss://example.test/v1/realtime_video/generate");
  client.setActive(true, 1000);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(resolvers.length, 1);

  client.reset("new", "wss://example.test/v1/realtime_video/generate");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(resolvers.length, 2);
  assert.ok(resolvers[1].url.includes("/new?"));

  resolvers[0].resolve({
    ok: true,
    json: async () => ({
      trace_id: "old",
      stages: [{ id: "denoise", count: 1 }],
      events: [],
      next_cursor: 5,
    }),
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(aggregates, []);

  resolvers[1].resolve({
    ok: true,
    json: async () => ({
      trace_id: "new",
      stages: [{ id: "denoise", count: 1 }],
      events: [],
      next_cursor: 6,
    }),
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(aggregates, ["new"]);
  client.setActive(false);
}

assert.equal(
  traceHttpBaseUrl("ws://127.0.0.1:30000/v1/realtime_video/generate"),
  "http://127.0.0.1:30000/v1/realtime_video/traces",
);

const appSource = fs.readFileSync(require.resolve("./app.js"), "utf8");
assert.equal(appSource.includes('kind: "client_trace"'), false);
assert.equal(appSource.includes('message.type === "trace_events"'), false);
assert.equal(appSource.includes('message.type === "chunk_stats"'), false);

Promise.resolve()
  .then(flushesClientEventsInOneHttpBatch)
  .then(pollsIncrementallyOnlyWhenRequested)
  .then(preservesLastAggregateAcrossTransientFailure)
  .then(previewDoesNotPollAndTraceDoes)
  .then(resetKeepsVisibleTracePollingAndIgnoresOldResponses)
  .then(() => console.log("realtime trace HTTP transport ok"));
