(function attachRealtimePlaybackController(global) {
  const DEFAULT_CONFIG = {
    targetFps: 25,
    minSourceFps: 1,
    serverFpsAlphaUp: 0.28,
    serverFpsAlphaDown: 0.2,
    deliveryFpsAlphaUp: 0.08,
    deliveryFpsAlphaDown: 0.55,
    targetLeadChunkRatio: 1.5,
    minTargetLeadMs: 1500,
    maxTargetLeadMs: 2600,
    maxLeadExtraChunkRatio: 8.0,
    startLeadChunkRatio: 1.85,
    minStartLeadMs: 1700,
    resumeLeadChunkRatio: 2.5,
    minResumeLeadMs: 1000,
    maxResumeLeadMs: 1800,
    rebufferLeadBoostMs: 250,
    rebufferLeadBoostDecayMsPerSecond: 120,
    deliveryLeadBoostDecayMsPerSecond: 80,
    maxDeliveryLeadBoostMs: 2000,
    deliveryStallExpectedMultiplier: 1.25,
    receiveStallPlaybackRateMin: 0.65,
    receiveStallPlaybackRateSlewPerSecond: 0.5,
    lowWaterRatio: 0.4,
    playbackRateGain: 0.14,
    playbackRateMin: 0.92,
    playbackRateMax: 1.08,
    emergencyPlaybackRateMin: 0.9,
    emergencyPlaybackRateMax: 1.12,
    playbackRateSlewPerSecond: 0.08,
    eventCutoverMaxMs: 420,
    eventCutoverMaxFrames: 10,
    settleEventCutoverMaxMs: 720,
    settleEventCutoverMaxFrames: 18,
    startupWarmupMinMs: 1500,
    startupWarmupExpectedMultiplier: 3,
  };

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function finitePositive(value) {
    return Number.isFinite(value) && value > 0;
  }

  class RealtimePlaybackController {
    constructor(config = {}) {
      this.config = { ...DEFAULT_CONFIG, ...config };
      this.reset({ targetFps: this.config.targetFps });
    }

    reset({ targetFps } = {}) {
      this.targetFps = Math.max(1, Number(targetFps || this.config.targetFps));
      this.sourceFps = this.targetFps;
      this.serverFps = this.targetFps;
      this.deliveryFps = this.targetFps;
      this.hasServerSample = false;
      this.hasDeliverySample = false;
      this.latestChunkDurationMs = 1000 / this.targetFps;
      this.latestChunkFrames = 1;
      this.playbackRate = 1;
      this.renderFps = this.targetFps;
      this.queue = [];
      this.lastDrawAt = 0;
      this.lastRateUpdateAt = 0;
      this.renderedFrames = 0;
      this.droppedFrames = 0;
      this.buffering = true;
      this.pendingEventId = 0;
      this.pendingEventSentAt = 0;
      this.pendingEventCutoverMode = "motion";
      this.lastDropReason = "";
      this.lastDropAt = 0;
      this.lastDropCount = 0;
      this.rebufferLeadBoostMs = 0;
      this.deliveryLeadBoostMs = 0;
      this.chunkReceives = new Map();
      this.serverStatChunks = new Set();
      this.lastFinalReceiveAt = 0;
      this.receiveStalled = false;
    }

    setTargetFps(targetFps) {
      const nextTargetFps = Math.max(1, Number(targetFps || this.config.targetFps));
      this.targetFps = nextTargetFps;
      if (!this.hasServerSample && !this.hasDeliverySample) {
        this.serverFps = nextTargetFps;
        this.deliveryFps = nextTargetFps;
        this.sourceFps = nextTargetFps;
        this.renderFps = nextTargetFps;
      } else {
        this.serverFps = clamp(this.serverFps, this.config.minSourceFps, nextTargetFps);
        this.deliveryFps = clamp(this.deliveryFps, this.config.minSourceFps, nextTargetFps);
        this.sourceFps = clamp(this.sourceFps, this.config.minSourceFps, nextTargetFps);
        this.renderFps = this.sourceFps * this.playbackRate;
      }
      this.latestChunkDurationMs = Math.max(this.latestChunkDurationMs, 1000 / this.targetFps);
    }

    clear() {
      const frames = this.queue.splice(0);
      this.lastDrawAt = 0;
      this.buffering = true;
      return frames;
    }

    noteInputEvent(eventId, now, { cutoverMode = "motion" } = {}) {
      this.pendingEventId = Number(eventId || 0);
      this.pendingEventSentAt = Number(now || 0);
      this.pendingEventCutoverMode = cutoverMode;
    }

    observeServerStats(stats, now) {
      const chunkIndex = Number(stats.chunk_index || 0);
      const numFrames = Number(stats.num_frames || 0);
      const chunkTotalMs = Number(stats.chunk_total_ms || 0);
      if (numFrames > 0 && chunkTotalMs > 0) {
        this.serverStatChunks.add(chunkIndex);
        if (this.serverStatChunks.size > 128) {
          this.serverStatChunks.delete(this.serverStatChunks.values().next().value);
        }
        const expectedMs = numFrames / Math.max(1, this.targetFps) * 1000;
        const isStartupWarmup =
          chunkIndex === 0 &&
          chunkTotalMs > Math.max(
            this.config.startupWarmupMinMs,
            expectedMs * this.config.startupWarmupExpectedMultiplier,
          );
        if (isStartupWarmup) return this.snapshot();
        this.#observeFpsSample("server", {
          fps: numFrames / (chunkTotalMs / 1000),
          frameCount: numFrames,
          durationMs: chunkTotalMs,
          now,
        });
      }
      return this.snapshot();
    }

    enqueueDecodedFrames(header, frames, now) {
      const chunkIndex = Number(header.chunk_index || 0);
      const eventId = Number(header.event_id || 0);
      const receivedAt = Number(header.__received_at || now);
      const preparedFrames = frames.map((frame) => ({
        ...frame,
        chunk: Number(frame.chunk ?? chunkIndex),
        chunkIndex,
        eventId,
      }));
      const droppedFrames = [];
      let cutover = null;

      if (this.pendingEventId && eventId >= this.pendingEventId) {
        const oldEventFrameCount = this.#oldEventFrameCount(eventId);
        const graceFrames = this.#eventGraceFrames();
        const dropCount = Math.max(0, oldEventFrameCount - graceFrames);
        if (dropCount > 0) {
          droppedFrames.push(...this.queue.splice(graceFrames, dropCount));
          this.#recordDrop(dropCount, "event cutover", now);
        }
        cutover = {
          eventId,
          latencyMs: this.pendingEventSentAt ? now - this.pendingEventSentAt : 0,
        };
        this.pendingEventId = 0;
        this.pendingEventSentAt = 0;
        this.pendingEventCutoverMode = "motion";
      }

      this.queue.push(...preparedFrames);
      this.#observeChunkArrival(header, preparedFrames.length, receivedAt, now);
      droppedFrames.push(...this.#trimBacklog(now));
      return { droppedFrames, cutover, snapshot: this.snapshot() };
    }

    render(now, { hasPendingInput = true } = {}) {
      this.#decayRebufferBoost(now);
      this.#updateReceiveStallGuard(now);
      const droppedFrames = this.#trimBacklog(now);
      if (!this.queue.length) {
        if (this.renderedFrames && hasPendingInput && !this.buffering) {
          this.buffering = true;
          this.rebufferLeadBoostMs = Math.max(
            this.rebufferLeadBoostMs,
            this.config.rebufferLeadBoostMs,
          );
        }
        return { action: "hold", droppedFrames, snapshot: this.snapshot() };
      }

      const bufferMs = this.bufferDurationMs;
      if (
        hasPendingInput &&
        this.receiveStalled &&
        this.renderedFrames &&
        bufferMs < this.targetLeadMs
      ) {
        this.buffering = true;
        this.lastDrawAt = 0;
        return { action: "hold", droppedFrames, snapshot: this.snapshot() };
      }

      if (
        hasPendingInput &&
        this.buffering &&
        bufferMs < (this.renderedFrames ? this.#resumeLeadMs() : this.#startLeadMs())
      ) {
        this.buffering = true;
        this.lastDrawAt = 0;
        return { action: "hold", droppedFrames, snapshot: this.snapshot() };
      }

      if (this.buffering) {
        this.buffering = false;
        this.lastDrawAt = 0;
      }

      this.#updatePlaybackRate(now);
      const targetMs = 1000 / Math.max(1, this.renderFps);
      const elapsedMs = this.lastDrawAt ? now - this.lastDrawAt : targetMs;
      if (elapsedMs < targetMs) {
        return { action: "wait", droppedFrames, snapshot: this.snapshot() };
      }

      const frame = this.queue.shift();
      this.renderedFrames += 1;
      this.lastDrawAt = !this.lastDrawAt || elapsedMs > targetMs * 4
        ? now
        : now - (elapsedMs % targetMs);
      return { action: "draw", frame, droppedFrames, snapshot: this.snapshot() };
    }

    get queuedFrames() {
      return this.queue.length;
    }

    get bufferDurationMs() {
      return this.queue.length / Math.max(1, this.sourceFps) * 1000;
    }

    get targetLeadMs() {
      const base = clamp(
        this.latestChunkDurationMs * this.config.targetLeadChunkRatio,
        this.config.minTargetLeadMs,
        this.config.maxTargetLeadMs,
      );
      return clamp(
        base + this.rebufferLeadBoostMs + this.deliveryLeadBoostMs,
        this.config.minTargetLeadMs,
        this.config.maxTargetLeadMs +
          this.config.rebufferLeadBoostMs +
          this.config.maxDeliveryLeadBoostMs,
      );
    }

    get maxLeadMs() {
      return this.targetLeadMs + this.latestChunkDurationMs * this.config.maxLeadExtraChunkRatio;
    }

    snapshot() {
      return {
        queueFrames: this.queue.length,
        bufferMs: this.bufferDurationMs,
        targetLeadMs: this.targetLeadMs,
        maxLeadMs: this.maxLeadMs,
        sourceFps: this.sourceFps,
        serverFps: this.serverFps,
        deliveryFps: this.deliveryFps,
        targetFps: this.targetFps,
        renderFps: this.renderFps,
        playbackRate: this.playbackRate,
        droppedFrames: this.droppedFrames,
        lastDropAt: this.lastDropAt,
        lastDropCount: this.lastDropCount,
        buffering: this.buffering,
        lastDropReason: this.lastDropReason,
      };
    }

    #observeFpsSample(kind, { fps, frameCount, durationMs, now }) {
      if (!finitePositive(fps)) return;
      const cappedFps = clamp(fps, this.config.minSourceFps, this.targetFps);
      const isDelivery = kind === "delivery";
      const currentFps = isDelivery ? this.deliveryFps : this.serverFps;
      const hasSample = isDelivery ? this.hasDeliverySample : this.hasServerSample;
      let nextFps;
      if (!hasSample) {
        nextFps = cappedFps;
      } else {
        const alpha = cappedFps > currentFps
          ? (isDelivery ? this.config.deliveryFpsAlphaUp : this.config.serverFpsAlphaUp)
          : (isDelivery ? this.config.deliveryFpsAlphaDown : this.config.serverFpsAlphaDown);
        nextFps = currentFps * (1 - alpha) + cappedFps * alpha;
      }
      if (isDelivery) {
        this.deliveryFps = nextFps;
        this.hasDeliverySample = true;
        this.#observeDeliveryJitter(frameCount, durationMs);
      } else {
        this.serverFps = nextFps;
        this.hasServerSample = true;
      }
      const effectiveFps = this.hasServerSample
        ? this.serverFps
        : (this.hasDeliverySample ? this.deliveryFps : this.targetFps);
      this.sourceFps = clamp(effectiveFps, this.config.minSourceFps, this.targetFps);
      if (!isDelivery || !this.hasServerSample) {
        this.latestChunkFrames = Math.max(1, Number(frameCount || this.latestChunkFrames));
        this.latestChunkDurationMs = clamp(
          Number(durationMs || (this.latestChunkFrames / Math.max(1, this.sourceFps) * 1000)),
          1000 / Math.max(1, this.targetFps),
          2500,
        );
      }
      this.#updatePlaybackRate(now);
    }

    #observeDeliveryJitter(frameCount, durationMs) {
      if (!this.hasServerSample || !finitePositive(durationMs)) return;
      const expectedMs = Number(frameCount || 0) / Math.max(1, this.serverFps) * 1000;
      if (expectedMs <= 0) return;
      if (durationMs <= expectedMs * this.config.deliveryStallExpectedMultiplier) return;
      const boostMs = clamp(
        durationMs - expectedMs,
        0,
        this.config.maxDeliveryLeadBoostMs,
      );
      this.deliveryLeadBoostMs = Math.max(this.deliveryLeadBoostMs, boostMs);
    }

    #updateReceiveStallGuard(now) {
      this.receiveStalled = false;
      if (!this.lastFinalReceiveAt || !this.hasServerSample) return;
      const elapsedMs = now - this.lastFinalReceiveAt;
      const expectedMs = Math.max(
        this.latestChunkDurationMs,
        this.latestChunkFrames / Math.max(1, this.serverFps) * 1000,
      );
      if (elapsedMs <= expectedMs * this.config.deliveryStallExpectedMultiplier) return;
      this.receiveStalled = true;
      this.deliveryLeadBoostMs = Math.max(
        this.deliveryLeadBoostMs,
        clamp(elapsedMs - expectedMs, 0, this.config.maxDeliveryLeadBoostMs),
      );
    }

    #observeChunkArrival(header, frameCount, receivedAt, now) {
      const chunkIndex = Number(header.chunk_index || 0);
      const state = this.chunkReceives.get(chunkIndex) || {
        firstAt: receivedAt,
        frames: 0,
      };
      state.frames += Number(frameCount || 0);
      state.lastAt = receivedAt;
      this.chunkReceives.set(chunkIndex, state);

      const frameBatchIndex = Number(header.frame_batch_index || 0);
      const numFrameBatches = Number(header.num_frame_batches || 1);
      const isFinalFrameBatch =
        Boolean(header.is_final_frame_batch) ||
        frameBatchIndex + 1 >= numFrameBatches;
      if (!isFinalFrameBatch) return;
      const durationMs = this.lastFinalReceiveAt
        ? receivedAt - this.lastFinalReceiveAt
        : 0;
      this.lastFinalReceiveAt = receivedAt;
      if (state.frames > 0 && durationMs > 0) {
        this.#observeFpsSample("delivery", {
          fps: state.frames / (durationMs / 1000),
          frameCount: state.frames,
          durationMs,
          now,
        });
      }
      this.chunkReceives.delete(chunkIndex);
    }

    #updatePlaybackRate(now) {
      const bufferMs = this.bufferDurationMs;
      const targetLeadMs = Math.max(1, this.targetLeadMs);
      const error = (bufferMs - targetLeadMs) / targetLeadMs;
      const emergency =
        bufferMs > this.maxLeadMs ||
        bufferMs < targetLeadMs * this.config.lowWaterRatio ||
        (this.receiveStalled && bufferMs < targetLeadMs);
      const minRate = emergency
        ? (
            this.receiveStalled
              ? this.config.receiveStallPlaybackRateMin
              : this.config.emergencyPlaybackRateMin
          )
        : this.config.playbackRateMin;
      const maxRate = this.receiveStalled && bufferMs < targetLeadMs
        ? 1
        : emergency
        ? this.config.emergencyPlaybackRateMax
        : this.config.playbackRateMax;
      const desiredRate = clamp(
        1 + error * this.config.playbackRateGain,
        minRate,
        maxRate,
      );

      if (!this.lastRateUpdateAt) {
        this.playbackRate = desiredRate;
      } else {
        const dtSeconds = Math.max(0.001, (now - this.lastRateUpdateAt) / 1000);
        const slewPerSecond = this.receiveStalled
          ? this.config.receiveStallPlaybackRateSlewPerSecond
          : this.config.playbackRateSlewPerSecond;
        const maxDelta = slewPerSecond * dtSeconds;
        this.playbackRate = clamp(
          desiredRate,
          this.playbackRate - maxDelta,
          this.playbackRate + maxDelta,
        );
      }
      this.lastRateUpdateAt = now;
      this.renderFps = clamp(
        this.sourceFps * this.playbackRate,
        this.config.minSourceFps,
        this.targetFps * this.config.emergencyPlaybackRateMax,
      );
    }

    #trimBacklog(now) {
      const droppedFrames = [];
      while (this.queue.length && this.bufferDurationMs > this.maxLeadMs) {
        const firstChunk = this.queue[0].chunkIndex;
        let dropCount = 0;
        while (
          dropCount < this.queue.length &&
          this.queue[dropCount].chunkIndex === firstChunk
        ) {
          dropCount += 1;
        }
        if (!dropCount || dropCount >= this.queue.length) break;
        droppedFrames.push(...this.queue.splice(0, dropCount));
        this.#recordDrop(dropCount, "backlog", now);
      }
      return droppedFrames;
    }

    #oldEventFrameCount(nextEventId) {
      let count = 0;
      while (count < this.queue.length && this.queue[count].eventId < nextEventId) {
        count += 1;
      }
      return count;
    }

    #eventGraceFrames() {
      const byTime = Math.max(
        1,
        Math.round(this.sourceFps * this.#eventCutoverMaxMs() / 1000),
      );
      const byChunkRatio = this.pendingEventCutoverMode === "settle" ? 1.5 : 0.85;
      const byChunk = Math.max(1, Math.round(this.latestChunkFrames * byChunkRatio));
      return Math.min(this.#eventCutoverMaxFrames(), byTime, byChunk);
    }

    #eventCutoverMaxMs() {
      return this.pendingEventCutoverMode === "settle"
        ? this.config.settleEventCutoverMaxMs
        : this.config.eventCutoverMaxMs;
    }

    #eventCutoverMaxFrames() {
      return this.pendingEventCutoverMode === "settle"
        ? this.config.settleEventCutoverMaxFrames
        : this.config.eventCutoverMaxFrames;
    }

    #startLeadMs() {
      return Math.max(
        this.config.minStartLeadMs,
        this.latestChunkDurationMs * this.config.startLeadChunkRatio,
        this.targetLeadMs,
      );
    }

    #resumeLeadMs() {
      return clamp(
        this.latestChunkDurationMs * this.config.resumeLeadChunkRatio,
        this.config.minResumeLeadMs,
        this.config.maxResumeLeadMs,
      );
    }

    #decayRebufferBoost(now) {
      if ((!this.rebufferLeadBoostMs && !this.deliveryLeadBoostMs) || !this.lastRateUpdateAt) return;
      const dtSeconds = Math.max(0, (now - this.lastRateUpdateAt) / 1000);
      this.rebufferLeadBoostMs = Math.max(
        0,
        this.rebufferLeadBoostMs - dtSeconds * this.config.rebufferLeadBoostDecayMsPerSecond,
      );
      this.deliveryLeadBoostMs = Math.max(
        0,
        this.deliveryLeadBoostMs - dtSeconds * this.config.deliveryLeadBoostDecayMsPerSecond,
      );
    }

    #recordDrop(count, reason, now) {
      this.droppedFrames += count;
      this.lastDropAt = Number(now || 0);
      this.lastDropCount = count;
      this.lastDropReason = reason;
    }
  }

  global.RealtimePlaybackController = RealtimePlaybackController;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { RealtimePlaybackController };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
