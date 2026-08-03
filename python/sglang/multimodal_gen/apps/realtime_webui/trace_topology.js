(function initTraceTopology(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.SGLangRealtimeTraceTopology = api;
})(typeof globalThis !== "undefined" ? globalThis : window, function traceTopologyFactory() {
  const DEFAULT_MAX_EVENTS = 160;
  const DEFAULT_TRANSFER_MS = 10;

  const STAGES = [
    { id: "browser", title: "Browser", subtitle: "input + render" },
    { id: "gateway", title: "Gateway", subtitle: "WS edge" },
    { id: "api", title: "Realtime API", subtitle: "session + batch" },
    { id: "scheduler", title: "Scheduler", subtitle: "forward" },
    { id: "vae_encode", title: "VAE Encode", subtitle: "input latents" },
    { id: "denoise", title: "Denoising", subtitle: "DiT / model" },
    { id: "vae_decode", title: "VAE Decode", subtitle: "output frames" },
    { id: "transport", title: "Transport", subtitle: "WebP + WS" },
    { id: "frontend", title: "Frontend", subtitle: "decode + canvas" },
  ];

  function createRealtimeTraceTopology(options = {}) {
    const maxEvents = Math.max(1, Number(options.maxEvents || DEFAULT_MAX_EVENTS));
    const transferBudgetMs = Math.max(0, Number(options.transferBudgetMs ?? DEFAULT_TRANSFER_MS));
    let traceId = "";
    let events = [];
    let eventKeys = new Set();
    let chunks = new Map();

    function reset(nextTraceId = "") {
      traceId = String(nextTraceId || "");
      events = [];
      eventKeys = new Set();
      chunks = new Map();
    }

    function addEvent(rawEvent, receivedPerfMs = 0) {
      const event = normalizeTraceEvent(rawEvent, receivedPerfMs);
      if (!event.event) return null;
      if (traceId && event.trace_id && event.trace_id !== traceId) return null;
      if (!traceId && event.trace_id) traceId = event.trace_id;

      const key = traceEventKey(event);
      if (eventKeys.has(key)) return event;
      eventKeys.add(key);
      events.push(event);
      if (events.length > maxEvents) {
        const removed = events.shift();
        eventKeys.delete(traceEventKey(removed));
      }
      rebuildChunks();
      return event;
    }

    function rebuildChunks() {
      chunks = new Map();
      for (const event of events) {
        const chunkIndex = event.chunk_index;
        if (chunkIndex === null || chunkIndex === undefined || Number.isNaN(chunkIndex)) {
          continue;
        }
        const chunk = ensureChunk(chunks, chunkIndex);
        chunk.events.push(event);
        applyEventToChunk(chunk, event);
      }
    }

    function summary() {
      const sortedChunks = Array.from(chunks.values()).sort(
        (left, right) => Number(left.chunkIndex) - Number(right.chunkIndex),
      );
      const latestObservedChunk = sortedChunks[sortedChunks.length - 1] || null;
      const latestChunk = latestTimedChunk(sortedChunks) || latestObservedChunk;
      const nodes = buildNodes(latestChunk, events);
      const edges = buildEdges(latestChunk, events);
      return {
        traceId,
        eventCount: events.length,
        recentEvents: events.slice(-10),
        chunks: sortedChunks,
        latestObservedChunk,
        latestChunk,
        nodes,
        edges,
        asyncEstimate: estimateAsyncVae(latestChunk, transferBudgetMs),
      };
    }

    reset(options.traceId || "");
    return { addEvent, reset, summary };
  }

  function normalizeTraceEvent(rawEvent, receivedPerfMs = 0) {
    const input = rawEvent && typeof rawEvent === "object" ? rawEvent : {};
    const event = String(input.event || input.name || input.type || "");
    const chunkIndex = numericOrNull(input.chunk_index);
    return {
      ...input,
      event,
      chunk_index: chunkIndex,
      event_id: numericOrNull(input.event_id),
      client_received_perf_ms: numericOrNull(receivedPerfMs),
    };
  }

  function traceEventKey(event) {
    return [
      event.trace_id || "",
      event.event || "",
      event.stage || event.component || "",
      event.chunk_index ?? "",
      event.event_id ?? "",
      event.server_elapsed_ms ?? "",
      event.client_perf_ms ?? "",
      event.duration_ms ?? event.cuda_ms ?? "",
      event.seq ?? "",
    ].join(":");
  }

  function ensureChunk(chunks, chunkIndex) {
    const key = Number(chunkIndex);
    if (!chunks.has(key)) {
      chunks.set(key, {
        chunkIndex: key,
        events: [],
        requestPrepareMs: null,
        schedulerForwardMs: null,
        denoiseMs: null,
        vaeEncodeMs: null,
        vaeDecodeMs: null,
        postDecodeMs: null,
        rawPayloadBuildMs: null,
        wsWriteMs: null,
        chunkTotalMs: null,
        clientDecodeMs: null,
        displayLagMs: null,
        numFrames: null,
        payloadBytes: null,
        contentType: "",
      });
    }
    return chunks.get(key);
  }

  function applyEventToChunk(chunk, event) {
    if (event.event === "server.chunk_complete" || event.type === "chunk_stats") {
      assignNumber(chunk, "requestPrepareMs", event.request_prepare_ms);
      assignNumber(chunk, "schedulerForwardMs", event.scheduler_forward_ms);
      assignNumber(chunk, "rawPayloadBuildMs", event.raw_payload_build_ms);
      assignNumber(chunk, "wsWriteMs", event.ws_write_ms);
      assignNumber(chunk, "chunkTotalMs", event.chunk_total_ms);
      assignNumber(chunk, "numFrames", event.num_frames);
      assignNumber(chunk, "payloadBytes", event.ws_payload_bytes);
      if (event.content_type) chunk.contentType = String(event.content_type);
    } else if (event.event === "server.model_denoise_complete") {
      assignNumber(chunk, "denoiseMs", preferredDuration(event));
    } else if (event.event === "server.vae_encode_complete") {
      assignNumber(chunk, "vaeEncodeMs", preferredDuration(event));
    } else if (event.event === "server.vae_decode_complete") {
      assignNumber(chunk, "vaeDecodeMs", preferredDuration(event));
    } else if (event.event === "server.post_decode_complete") {
      assignNumber(chunk, "postDecodeMs", preferredDuration(event));
    } else if (event.event === "server.pipeline_stage_complete") {
      applyPipelineStageDuration(chunk, event);
    } else if (event.event === "client.decode_batch_done") {
      assignNumber(chunk, "clientDecodeMs", event.decode_ms);
    } else if (event.event === "client.chunk_first_rendered") {
      assignNumber(chunk, "displayLagMs", event.display_lag_ms);
    }
  }

  function buildNodes(latestChunk, events) {
    const latestByStage = latestEventsByStage(events);
    return STAGES.map((stage) => {
      const event = latestByStage.get(stage.id);
      const metric = stageMetric(stage.id, latestChunk, event);
      return {
        ...stage,
        metric,
        status: event ? "active" : "idle",
      };
    });
  }

  function buildEdges(latestChunk, events) {
    const edges = [];
    for (let i = 0; i < STAGES.length - 1; i += 1) {
      const from = STAGES[i].id;
      const to = STAGES[i + 1].id;
      edges.push({ from, to, label: edgeMetric(from, to, latestChunk, events) });
    }
    return edges;
  }

  function latestEventsByStage(events) {
    const latest = new Map();
    for (const event of events) {
      const stage = stageForEvent(event);
      if (!stage) continue;
      latest.set(stage, event);
    }
    return latest;
  }

  function stageForEvent(eventOrName) {
    const eventName = typeof eventOrName === "string"
      ? eventOrName
      : String(eventOrName?.event || "");
    if (!eventName) return "";
    if (eventName.startsWith("client.")) {
      if (
        eventName === "client.frame_batch_received" ||
        eventName === "client.decode_batch_done" ||
        eventName === "client.chunk_first_rendered"
      ) return "frontend";
      return "browser";
    }
    if (eventName === "server.ws_accepted" || eventName === "server.init_received") return "gateway";
    if (
      eventName === "server.adapter_init_start" ||
      eventName === "server.adapter_init_done" ||
      eventName === "server.init_ready" ||
      eventName === "server.chunk_prepare_start"
    ) return "api";
    if (
      eventName === "server.scheduler_forward_start" ||
      eventName === "server.scheduler_forward_done"
    ) return "scheduler";
    if (eventName === "server.pipeline_stage_complete") {
      return pipelineStageGroup(eventOrName) || "scheduler";
    }
    if (eventName === "server.vae_encode_complete") return "vae_encode";
    if (eventName === "server.model_denoise_complete") return "denoise";
    if (
      eventName === "server.vae_decode_complete" ||
      eventName === "server.post_decode_complete"
    ) return "vae_decode";
    if (
      eventName === "server.output_send_start" ||
      eventName === "server.chunk_complete" ||
      eventName === "server.chunk_stats_sent"
    ) return "transport";
    return "";
  }

  function stageMetric(stageId, chunk, event) {
    if (!chunk && event) return elapsedMetric(event);
    if (!chunk) return "-";
    if (stageId === "api") return formatTraceDuration(chunk.requestPrepareMs);
    if (stageId === "scheduler") return formatTraceDuration(chunk.schedulerForwardMs);
    if (stageId === "vae_encode") return formatTraceDuration(chunk.vaeEncodeMs);
    if (stageId === "denoise") return formatTraceDuration(chunk.denoiseMs);
    if (stageId === "vae_decode") return formatTraceDuration(sumNumbers(chunk.vaeDecodeMs, chunk.postDecodeMs));
    if (stageId === "transport") {
      return formatTraceDuration(sumNumbers(chunk.rawPayloadBuildMs, chunk.wsWriteMs));
    }
    if (stageId === "frontend") {
      return formatTraceDuration(sumNumbers(chunk.clientDecodeMs, chunk.displayLagMs));
    }
    if (stageId === "browser") return event ? elapsedMetric(event) : "-";
    if (stageId === "gateway") return event ? elapsedMetric(event) : "-";
    return "-";
  }

  function edgeMetric(from, to, chunk, events) {
    if (from === "api" && to === "scheduler" && chunk) return formatTraceDuration(chunk.requestPrepareMs);
    if (from === "scheduler" && to === "vae_encode" && chunk) return formatTraceDuration(chunk.schedulerForwardMs);
    if (from === "vae_encode" && to === "denoise" && chunk) return formatTraceDuration(chunk.vaeEncodeMs);
    if (from === "denoise" && to === "vae_decode" && chunk) {
      return formatTraceDuration(sumNumbers(chunk.vaeDecodeMs, chunk.postDecodeMs));
    }
    if (from === "vae_decode" && to === "transport" && chunk) {
      return formatTraceDuration(sumNumbers(chunk.rawPayloadBuildMs, chunk.wsWriteMs));
    }
    if (from === "transport" && to === "frontend" && chunk) return formatTraceDuration(chunk.displayLagMs);

    const fromEvent = lastEventForStage(events, from);
    const toEvent = lastEventForStage(events, to);
    const delta = eventDeltaMs(fromEvent, toEvent);
    return formatTraceDuration(delta);
  }

  function lastEventForStage(events, stage) {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      if (stageForEvent(events[i]) === stage) return events[i];
    }
    return null;
  }

  function eventDeltaMs(fromEvent, toEvent) {
    const fromMs = eventRelativeMs(fromEvent);
    const toMs = eventRelativeMs(toEvent);
    if (fromMs === null || toMs === null || toMs < fromMs) return null;
    return toMs - fromMs;
  }

  function eventRelativeMs(event) {
    if (!event) return null;
    if (isFiniteNumber(event.server_elapsed_ms)) return Number(event.server_elapsed_ms);
    if (isFiniteNumber(event.client_perf_ms)) return Number(event.client_perf_ms);
    if (isFiniteNumber(event.client_received_perf_ms)) return Number(event.client_received_perf_ms);
    return null;
  }

  function elapsedMetric(event) {
    const value = eventRelativeMs(event);
    return value === null ? "-" : `t+${formatTraceDuration(value)}`;
  }

  function estimateAsyncVae(chunk, transferBudgetMs) {
    if (!chunk || !isFiniteNumber(chunk.denoiseMs) || !isFiniteNumber(chunk.vaeDecodeMs)) {
      return null;
    }
    const denoiseMs = Number(chunk.denoiseMs);
    const vaeDecodeMs = Number(chunk.vaeDecodeMs);
    const syncComputeMs = roundTraceValue(denoiseMs + vaeDecodeMs);
    const asyncCriticalMs = roundTraceValue(Math.max(denoiseMs, transferBudgetMs + vaeDecodeMs));
    const savedMs = roundTraceValue(Math.max(0, syncComputeMs - asyncCriticalMs));
    const speedup = asyncCriticalMs > 0 ? roundTraceValue(syncComputeMs / asyncCriticalMs, 2) : 0;
    return {
      chunkIndex: chunk.chunkIndex,
      denoiseMs: roundTraceValue(denoiseMs),
      vaeDecodeMs: roundTraceValue(vaeDecodeMs),
      transferBudgetMs,
      syncComputeMs,
      asyncCriticalMs,
      savedMs,
      speedup,
    };
  }

  function preferredDuration(event) {
    return isFiniteNumber(event.cuda_ms) ? Number(event.cuda_ms) : numericOrNull(event.duration_ms);
  }

  function latestTimedChunk(sortedChunks) {
    for (let i = sortedChunks.length - 1; i >= 0; i -= 1) {
      if (hasChunkTiming(sortedChunks[i])) return sortedChunks[i];
    }
    return null;
  }

  function hasChunkTiming(chunk) {
    return [
      chunk.requestPrepareMs,
      chunk.schedulerForwardMs,
      chunk.denoiseMs,
      chunk.vaeEncodeMs,
      chunk.vaeDecodeMs,
      chunk.postDecodeMs,
      chunk.rawPayloadBuildMs,
      chunk.wsWriteMs,
      chunk.chunkTotalMs,
      chunk.clientDecodeMs,
      chunk.displayLagMs,
    ].some(isFiniteNumber);
  }

  function applyPipelineStageDuration(chunk, event) {
    const durationMs = preferredDuration(event);
    if (!isFiniteNumber(durationMs)) return;
    const stage = String(event.stage || event.component || "").toLowerCase();
    if (stage.includes("denois")) {
      assignNumber(chunk, "denoiseMs", durationMs);
    } else if (stage.includes("vae") && stage.includes("encod")) {
      assignNumber(chunk, "vaeEncodeMs", durationMs);
    } else if (stage.includes("vae") && stage.includes("decod")) {
      assignNumber(chunk, "vaeDecodeMs", durationMs);
    } else if (stage.includes("post") && stage.includes("decod")) {
      assignNumber(chunk, "postDecodeMs", durationMs);
    }
  }

  function pipelineStageGroup(event) {
    const stage = String(event?.stage || event?.component || "").toLowerCase();
    if (stage.includes("denois")) return "denoise";
    if (stage.includes("vae") && stage.includes("encod")) return "vae_encode";
    if (stage.includes("vae") && stage.includes("decod")) return "vae_decode";
    if (stage.includes("post") && stage.includes("decod")) return "vae_decode";
    return "";
  }

  function assignNumber(target, key, value) {
    if (isFiniteNumber(value)) target[key] = Number(value);
  }

  function numericOrNull(value) {
    if (value === "" || value === null || value === undefined) return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function isFiniteNumber(value) {
    return value !== null && value !== undefined && Number.isFinite(Number(value));
  }

  function sumNumbers(...values) {
    let total = 0;
    let seen = false;
    for (const value of values) {
      if (!isFiniteNumber(value)) continue;
      total += Number(value);
      seen = true;
    }
    return seen ? total : null;
  }

  function roundTraceValue(value, digits = 3) {
    const multiplier = 10 ** digits;
    return Math.round(Number(value || 0) * multiplier) / multiplier;
  }

  function formatTraceDuration(value) {
    if (!isFiniteNumber(value)) return "-";
    const ms = Number(value);
    if (Math.abs(ms) >= 1000) return `${(ms / 1000).toFixed(2)}s`;
    return `${Math.round(ms)}ms`;
  }

  return {
    TRACE_TOPOLOGY_STAGES: STAGES,
    createRealtimeTraceTopology,
    formatTraceDuration,
  };
});
