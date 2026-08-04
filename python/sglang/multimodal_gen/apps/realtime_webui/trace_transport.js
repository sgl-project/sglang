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
      flushBatchSize = 32,
      maxBufferedEvents = 128,
    } = {}) {
      if (!fetchImpl) throw new Error("fetch is required for trace HTTP transport");
      this.fetchImpl = fetchImpl;
      this.onServerEvents = onServerEvents;
      this.flushBatchSize = Math.max(1, flushBatchSize);
      this.maxBufferedEvents = Math.max(this.flushBatchSize, maxBufferedEvents);
      this.traceId = "";
      this.baseUrl = "";
      this.cursor = 0;
      this.pending = [];
      this.pollTimer = 0;
      this.flushTimer = 0;
      this.lastSuccessfulResult = null;
    }

    reset(traceId, serverUrl) {
      this.stopPolling();
      this.traceId = String(traceId || "");
      this.baseUrl = traceHttpBaseUrl(serverUrl);
      this.cursor = 0;
      this.pending = [];
      this.lastSuccessfulResult = null;
      this.flushTimer = 0;
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
      const events = this.pending.splice(0, this.flushBatchSize);
      try {
        const response = await this.fetchImpl(
          `${this.baseUrl}/${encodeURIComponent(this.traceId)}/client-events`,
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
        this.pending.unshift(...events);
        if (this.pending.length > this.maxBufferedEvents) this.pending.length = this.maxBufferedEvents;
        throw error;
      }
    }

    async pollOnce() {
      if (!this.traceId) return this.lastSuccessfulResult;
      const url = `${this.baseUrl}/${encodeURIComponent(this.traceId)}?after=${this.cursor}&limit=220`;
      const response = await this.fetchImpl(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`trace query failed: ${response.status}`);
      const result = await response.json();
      const events = Array.isArray(result.events) ? result.events : [];
      this.cursor = Math.max(this.cursor, Number(result.next_cursor || 0));
      this.lastSuccessfulResult = result;
      if (events.length) this.onServerEvents(events, result);
      return result;
    }

    startPolling(intervalMs = 5000) {
      if (this.pollTimer || !this.traceId) return;
      void this.pollOnce().catch(() => {});
      this.pollTimer = setInterval(() => {
        void this.pollOnce().catch(() => {});
      }, Math.max(1000, intervalMs));
    }

    stopPolling() {
      if (!this.pollTimer) return;
      clearInterval(this.pollTimer);
      this.pollTimer = 0;
    }
  }

  return { RealtimeTraceHttpClient, traceHttpBaseUrl };
});
