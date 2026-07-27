const assert = require("node:assert/strict");
const { RealtimePlaybackController } = require("./playback_controller.js");

function frames(count, chunk) {
  return Array.from({ length: count }, (_, index) => ({
    image: { close() {} },
    chunk,
    index,
  }));
}

function enqueueChunk(controller, {
  chunk,
  eventId = 0,
  frameCount = 12,
  durationMs = 480,
  now,
}) {
  controller.observeServerStats({
    chunk_index: chunk,
    num_frames: frameCount,
    chunk_total_ms: durationMs,
  }, now);
  return controller.enqueueDecodedFrames({
    chunk_index: chunk,
    event_id: eventId,
    num_frames: frameCount,
    __received_at: now,
    is_final_frame_batch: true,
  }, frames(frameCount, chunk), now);
}

function renderFor(controller, startMs, durationMs) {
  let rendered = 0;
  for (let now = startMs; now <= startMs + durationMs; now += 16.67) {
    const decision = controller.render(now, { hasPendingInput: true });
    if (decision.action === "draw") rendered += 1;
  }
  return rendered;
}

function stableSourceDoesNotDrop() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  let now = 0;
  for (let chunk = 0; chunk < 8; chunk += 1) {
    now += 480;
    enqueueChunk(controller, { chunk, now });
    renderFor(controller, now, 480);
  }
  const snapshot = controller.snapshot();
  assert.equal(snapshot.droppedFrames, 0);
  assert.ok(snapshot.sourceFps > 24 && snapshot.sourceFps <= 25);
}

function slowServerCapsRenderFps() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  let now = 0;
  for (let chunk = 0; chunk < 8; chunk += 1) {
    now += 1360;
    enqueueChunk(controller, { chunk, durationMs: 1360, now });
    renderFor(controller, now, 1360);
  }
  const snapshot = controller.snapshot();
  assert.ok(snapshot.sourceFps > 8 && snapshot.sourceFps < 10);
  assert.ok(snapshot.renderFps <= 10);
}

function backlogDropsContiguousOldFrames() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  let now = 100;
  for (let chunk = 0; chunk < 13; chunk += 1) {
    enqueueChunk(controller, { chunk, now, durationMs: 480 });
    now += 20;
  }
  const snapshot = controller.snapshot();
  assert.ok(snapshot.droppedFrames > 0);
  assert.equal(snapshot.lastDropReason, "backlog");
}

function eventCutoverKeepsShortGrace() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  enqueueChunk(controller, { chunk: 1, frameCount: 24, durationMs: 960, now: 1000 });
  controller.noteInputEvent(5, 1050);
  const result = enqueueChunk(controller, {
    chunk: 2,
    eventId: 5,
    frameCount: 12,
    durationMs: 480,
    now: 1150,
  });
  assert.ok(result.cutover);
  assert.ok(result.droppedFrames.length >= 14);
  assert.equal(controller.queue[0].chunk, 1);
  assert.equal(controller.queue[0].index, 0);
}

function settleEventCutoverKeepsWiderGrace() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  enqueueChunk(controller, { chunk: 1, frameCount: 24, durationMs: 960, now: 1000 });
  controller.noteInputEvent(5, 1050, { cutoverMode: "settle" });
  const result = enqueueChunk(controller, {
    chunk: 2,
    eventId: 5,
    frameCount: 12,
    durationMs: 480,
    now: 1150,
  });
  assert.ok(result.cutover);
  assert.ok(result.droppedFrames.length <= 12);
}

stableSourceDoesNotDrop();
slowServerCapsRenderFps();
backlogDropsContiguousOldFrames();
eventCutoverKeepsShortGrace();
settleEventCutoverKeepsWiderGrace();
