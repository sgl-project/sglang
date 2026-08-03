const assert = require("node:assert/strict");
const {
  TRACE_TOPOLOGY_STAGES,
  createRealtimeTraceTopology,
  formatTraceDuration,
} = require("./trace_topology.js");

function recordsChunkCriticalPathAndAsyncEstimate() {
  const topology = createRealtimeTraceTopology({ maxEvents: 16 });
  topology.reset("trace-a");

  topology.addEvent({
    name: "client.generate_clicked",
    trace_id: "trace-a",
    client_perf_ms: 1000,
  });
  topology.addEvent({
    event: "server.chunk_complete",
    trace_id: "trace-a",
    chunk_index: 3,
    request_prepare_ms: 12,
    scheduler_forward_ms: 820,
    raw_payload_build_ms: 34,
    ws_write_ms: 8,
    chunk_total_ms: 910,
    num_frames: 9,
    ws_payload_bytes: 500000,
    content_type: "image/webp",
  });
  topology.addEvent({
    event: "server.model_denoise_complete",
    trace_id: "trace-a",
    chunk_index: 3,
    duration_ms: 620,
  });
  topology.addEvent({
    event: "server.vae_decode_complete",
    trace_id: "trace-a",
    chunk_index: 3,
    cuda_ms: 180,
  });
  topology.addEvent({
    name: "client.decode_batch_done",
    trace_id: "trace-a",
    chunk_index: 3,
    decode_ms: 9,
  });
  topology.addEvent({
    name: "client.chunk_first_rendered",
    trace_id: "trace-a",
    chunk_index: 3,
    display_lag_ms: 240,
  });

  const summary = topology.summary();
  const chunk = summary.latestChunk;
  assert.equal(chunk.chunkIndex, 3);
  assert.equal(chunk.denoiseMs, 620);
  assert.equal(chunk.vaeDecodeMs, 180);
  assert.equal(chunk.schedulerForwardMs, 820);
  assert.equal(chunk.chunkTotalMs, 910);
  assert.equal(chunk.displayLagMs, 240);
  assert.equal(summary.asyncEstimate.syncComputeMs, 800);
  assert.equal(summary.asyncEstimate.asyncCriticalMs, 620);
  assert.equal(summary.asyncEstimate.savedMs, 180);
  assert.equal(summary.asyncEstimate.speedup, 1.29);
  assert.ok(summary.nodes.find((node) => node.id === "denoise").metric.includes("620ms"));
  const decodeEdge = summary.edges.find(
    (edge) => edge.from === "denoise" && edge.to === "vae_decode",
  );
  assert.ok(decodeEdge);
  assert.ok(decodeEdge.label.includes("180ms"));
}

function keepsOnlyCurrentTraceAndRecentEvents() {
  const topology = createRealtimeTraceTopology({ maxEvents: 3 });
  topology.reset("trace-b");
  topology.addEvent({ event: "server.ws_accepted", trace_id: "other", server_elapsed_ms: 0 });
  topology.addEvent({ event: "server.ws_accepted", trace_id: "trace-b", server_elapsed_ms: 0 });
  topology.addEvent({ event: "server.init_received", trace_id: "trace-b", server_elapsed_ms: 11 });
  topology.addEvent({ event: "server.adapter_init_done", trace_id: "trace-b", server_elapsed_ms: 22 });
  topology.addEvent({ event: "server.init_ready", trace_id: "trace-b", server_elapsed_ms: 33 });

  const summary = topology.summary();
  assert.equal(summary.eventCount, 3);
  assert.equal(summary.traceId, "trace-b");
  assert.deepEqual(summary.recentEvents.map((event) => event.event), [
    "server.init_received",
    "server.adapter_init_done",
    "server.init_ready",
  ]);
}

function usesLatestCompletedChunkWhenNextChunkIsInFlight() {
  const topology = createRealtimeTraceTopology({ maxEvents: 16 });
  topology.reset("trace-c");

  topology.addEvent({
    event: "server.chunk_complete",
    trace_id: "trace-c",
    chunk_index: 206,
    request_prepare_ms: 11,
    scheduler_forward_ms: 491,
    raw_payload_build_ms: 5,
    ws_write_ms: 1,
    chunk_total_ms: 508,
    num_frames: 9,
  });
  topology.addEvent({
    event: "client.decode_batch_done",
    trace_id: "trace-c",
    chunk_index: 206,
    decode_ms: 8,
  });
  topology.addEvent({
    event: "server.scheduler_forward_start",
    trace_id: "trace-c",
    chunk_index: 207,
    server_elapsed_ms: 94070,
  });

  const summary = topology.summary();
  assert.equal(summary.latestObservedChunk.chunkIndex, 207);
  assert.equal(summary.latestChunk.chunkIndex, 206);
  assert.equal(summary.latestChunk.chunkTotalMs, 508);
  assert.ok(summary.nodes.find((node) => node.id === "scheduler").metric.includes("491ms"));
}

function retainsLastKnownStageMetricsForPartiallyReportedChunks() {
  const topology = createRealtimeTraceTopology({ maxEvents: 32 });
  topology.reset("trace-sticky");

  topology.addEvent({
    event: "server.chunk_complete",
    trace_id: "trace-sticky",
    chunk_index: 16,
    request_prepare_ms: 6,
    scheduler_forward_ms: 450,
    raw_payload_build_ms: 40,
    ws_write_ms: 4,
    chunk_total_ms: 520,
  });
  topology.addEvent({
    event: "server.model_denoise_complete",
    trace_id: "trace-sticky",
    chunk_index: 16,
    duration_ms: 390,
  });
  topology.addEvent({
    event: "server.vae_decode_complete",
    trace_id: "trace-sticky",
    chunk_index: 16,
    duration_ms: 88,
  });

  topology.addEvent({
    event: "client.decode_batch_done",
    trace_id: "trace-sticky",
    chunk_index: 17,
    decode_ms: 4,
  });

  const summary = topology.summary();
  assert.equal(summary.latestObservedChunk.chunkIndex, 17);
  assert.equal(summary.latestChunk.chunkIndex, 17);
  assert.equal(summary.latestChunk.denoiseMs, 390);
  assert.equal(summary.latestChunk.vaeDecodeMs, 88);
  assert.equal(summary.latestChunk.clientDecodeMs, 4);
  assert.equal(summary.nodes.find((node) => node.id === "denoise").metric, "390ms");
  assert.equal(summary.nodes.find((node) => node.id === "vae_decode").metric, "88ms");
  assert.equal(summary.nodes.find((node) => node.id === "frontend").metric, "4ms");
}

function mapsGenericPipelineStageEventsToChunkMetrics() {
  const topology = createRealtimeTraceTopology({ maxEvents: 16 });
  topology.reset("trace-d");

  topology.addEvent({
    event: "server.pipeline_stage_complete",
    trace_id: "trace-d",
    chunk_index: 9,
    stage: "RealtimeImageVAEEncodingStage",
    duration_ms: 75,
  });
  topology.addEvent({
    event: "server.pipeline_stage_complete",
    trace_id: "trace-d",
    chunk_index: 9,
    stage: "LingBotWorldCausalDMDDenoisingStage",
    duration_ms: 630,
  });
  topology.addEvent({
    event: "server.pipeline_stage_complete",
    trace_id: "trace-d",
    chunk_index: 9,
    stage: "CausalVaeDecodingStage",
    duration_ms: 185,
  });

  const summary = topology.summary();
  assert.equal(summary.latestChunk.chunkIndex, 9);
  assert.equal(summary.latestChunk.vaeEncodeMs, 75);
  assert.equal(summary.latestChunk.denoiseMs, 630);
  assert.equal(summary.latestChunk.vaeDecodeMs, 185);
  assert.equal(summary.asyncEstimate.savedMs, 185);
}

function separatesVaeEncodeAndDecodeInTopologyOrder() {
  assert.deepEqual(TRACE_TOPOLOGY_STAGES.map((stage) => stage.id), [
    "browser",
    "gateway",
    "api",
    "scheduler",
    "vae_encode",
    "denoise",
    "vae_decode",
    "transport",
    "frontend",
  ]);

  const topology = createRealtimeTraceTopology({ maxEvents: 16 });
  topology.reset("trace-e");
  topology.addEvent({
    event: "server.vae_encode_complete",
    trace_id: "trace-e",
    chunk_index: 4,
    duration_ms: 75,
  });
  topology.addEvent({
    event: "server.model_denoise_complete",
    trace_id: "trace-e",
    chunk_index: 4,
    duration_ms: 630,
  });
  topology.addEvent({
    event: "server.vae_decode_complete",
    trace_id: "trace-e",
    chunk_index: 4,
    duration_ms: 185,
  });

  const summary = topology.summary();
  assert.equal(summary.nodes.find((node) => node.id === "vae_encode").metric, "75ms");
  assert.equal(summary.nodes.find((node) => node.id === "denoise").metric, "630ms");
  assert.equal(summary.nodes.find((node) => node.id === "vae_decode").metric, "185ms");
  assert.ok(summary.edges.find((edge) => edge.from === "vae_encode" && edge.to === "denoise"));
  assert.ok(summary.edges.find((edge) => edge.from === "denoise" && edge.to === "vae_decode"));
}

function keepsLatestCompleteRemoteVaeMetricsDuringNextChunk() {
  const topology = createRealtimeTraceTopology({ maxEvents: 32 });
  topology.reset("trace-remote");
  topology.addEvent({
    event: "server.remote_vae_complete",
    trace_id: "trace-remote",
    chunk_index: 7,
    vae_queue_wait_ms: 4,
    vae_decode_ms: 82,
    frame_encode_ms: 13,
    latent_serialize_ms: 2,
    latent_send_ms: 5,
    vae_credit_wait_ms: 3,
    latent_to_gateway_complete_ms: 111,
    overlap_with_next_denoise_ms: 70,
    overlap_ratio: 0.72,
  });
  topology.addEvent({
    event: "server.scheduler_forward_start",
    trace_id: "trace-remote",
    chunk_index: 8,
  });

  const summary = topology.summary();
  assert.equal(summary.latestChunk.vaeQueueWaitMs, 4);
  assert.equal(summary.latestChunk.vaeDecodeMs, 82);
  assert.equal(summary.latestChunk.frameEncodeMs, 13);
  assert.equal(summary.latestChunk.latentTransferMs, 8);
  assert.equal(summary.latestChunk.overlapMs, 70);
  assert.equal(summary.nodes.find((node) => node.id === "vae_decode").metric, "86ms");
}

function formatsReadableDurations() {
  assert.equal(formatTraceDuration(0), "0ms");
  assert.equal(formatTraceDuration(12.4), "12ms");
  assert.equal(formatTraceDuration(1250), "1.25s");
  assert.equal(formatTraceDuration(null), "-");
}

recordsChunkCriticalPathAndAsyncEstimate();
keepsOnlyCurrentTraceAndRecentEvents();
usesLatestCompletedChunkWhenNextChunkIsInFlight();
retainsLastKnownStageMetricsForPartiallyReportedChunks();
mapsGenericPipelineStageEventsToChunkMetrics();
separatesVaeEncodeAndDecodeInTopologyOrder();
keepsLatestCompleteRemoteVaeMetricsDuringNextChunk();
formatsReadableDurations();

console.log("trace topology tests ok");
