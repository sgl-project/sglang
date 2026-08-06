(function initRealtimeTraceTransport(root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.SGLangRealtimeTraceTransport = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function createApi() {
  function traceHttpBaseUrl(serverUrl) {
    const fallbackBase = typeof window !== "undefined" ? window.location.href : undefined;
    const url = new URL(serverUrl, fallbackBase);
    if (url.protocol === "ws:") url.protocol = "http:";
    if (url.protocol === "wss:") url.protocol = "https:";
    url.pathname = "/v1/realtime_video/traces";
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/$/, "");
  }

  class RealtimeTraceHttpClient {
    constructor({
      fetchImpl = globalThis.fetch?.bind(globalThis),
      onServerEvents = () => {},
      onAggregate = () => {},
      flushBatchSize = 32,
      maxBufferedEvents = 128,
    } = {}) {
      if (!fetchImpl) throw new Error("fetch is required for trace HTTP transport");
      this.fetchImpl = fetchImpl;
      this.onServerEvents = onServerEvents;
      this.onAggregate = onAggregate;
      this.flushBatchSize = Math.max(1, flushBatchSize);
      this.maxBufferedEvents = Math.max(this.flushBatchSize, maxBufferedEvents);
      this.traceId = "";
      this.baseUrl = "";
      this.cursor = 0;
      this.pending = [];
      this.pollTimer = 0;
      this.flushTimer = 0;
      this.lastSuccessfulResult = null;
      this.active = false;
      this.pollIntervalMs = 5000;
      this.generation = 0;
    }

    reset(traceId, serverUrl) {
      const restartPolling = this.active;
      this.stopPolling();
      if (this.flushTimer) clearTimeout(this.flushTimer);
      this.generation += 1;
      this.traceId = String(traceId || "");
      this.baseUrl = traceHttpBaseUrl(serverUrl);
      this.cursor = 0;
      this.pending = [];
      this.lastSuccessfulResult = null;
      this.flushTimer = 0;
      if (restartPolling) this.startPolling(this.pollIntervalMs);
    }

    enqueueClientEvent(event) {
      if (!this.traceId || !event) return;
      this.pending.push(event);
      if (this.pending.length > this.maxBufferedEvents) {
        this.pending.splice(0, this.pending.length - this.maxBufferedEvents);
      }
      if (this.pending.length >= this.flushBatchSize) {
        void this.flushClientEvents();
      } else if (!this.flushTimer) {
        this.flushTimer = setTimeout(() => {
          this.flushTimer = 0;
          void this.flushClientEvents().catch(() => {});
        }, 1000);
      }
    }

    async flushClientEvents() {
      if (this.flushTimer) {
        clearTimeout(this.flushTimer);
        this.flushTimer = 0;
      }
      if (!this.traceId || !this.pending.length) return { accepted: 0 };
      const generation = this.generation;
      const traceId = this.traceId;
      const baseUrl = this.baseUrl;
      const events = this.pending.splice(0, this.flushBatchSize);
      try {
        const response = await this.fetchImpl(
          `${baseUrl}/${encodeURIComponent(traceId)}/client-events`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ events }),
            keepalive: true,
          },
        );
        if (!response.ok) throw new Error(`trace event upload failed: ${response.status}`);
        return await response.json();
      } catch (error) {
        if (generation === this.generation) {
          this.pending.unshift(...events);
          if (this.pending.length > this.maxBufferedEvents) this.pending.length = this.maxBufferedEvents;
        }
        throw error;
      }
    }

    async pollOnce() {
      if (!this.traceId) return this.lastSuccessfulResult;
      const generation = this.generation;
      const url = `${this.baseUrl}/${encodeURIComponent(this.traceId)}?after=${this.cursor}&limit=220`;
      try {
        const response = await this.fetchImpl(url, { cache: "no-store" });
        if (!response.ok) throw new Error(`trace query failed: ${response.status}`);
        const result = await response.json();
        if (generation !== this.generation) return this.lastSuccessfulResult;
        const events = Array.isArray(result.events) ? result.events : [];
        this.cursor = Math.max(this.cursor, Number(result.next_cursor || 0));
        if (Array.isArray(result.stages) && result.stages.length) {
          this.lastSuccessfulResult = result;
          this.onAggregate(result);
        } else if (!this.lastSuccessfulResult) {
          this.lastSuccessfulResult = result;
        }
        if (events.length) this.onServerEvents(events, result);
        return Array.isArray(result.stages) && result.stages.length
          ? result
          : this.lastSuccessfulResult;
      } catch (error) {
        if (generation !== this.generation) return this.lastSuccessfulResult;
        if (!this.lastSuccessfulResult) throw error;
        const fallback = {
          ...this.lastSuccessfulResult,
          events: [],
          stale: true,
          stale_reason: "client_query_failed",
        };
        if (Array.isArray(fallback.stages) && fallback.stages.length) {
          this.onAggregate(fallback);
        }
        return fallback;
      }
    }

    setActive(active, intervalMs = 5000) {
      this.active = Boolean(active);
      this.pollIntervalMs = Math.max(1000, intervalMs);
      if (this.active) this.startPolling(this.pollIntervalMs);
      else this.stopPolling();
    }

    startPolling(intervalMs = 5000) {
      if (this.pollTimer || !this.traceId) return;
      this.pollIntervalMs = Math.max(1000, intervalMs);
      void this.pollOnce().catch(() => {});
      this.pollTimer = setInterval(() => {
        void this.pollOnce().catch(() => {});
      }, this.pollIntervalMs);
    }

    stopPolling() {
      if (!this.pollTimer) return;
      clearInterval(this.pollTimer);
      this.pollTimer = 0;
    }
  }

  return { RealtimeTraceHttpClient, traceHttpBaseUrl };
});
