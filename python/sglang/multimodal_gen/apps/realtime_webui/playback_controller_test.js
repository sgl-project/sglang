const assert = require("node:assert/strict");
const { RealtimePlaybackController } = require("./playback_controller.js");

function frames(count, chunk, receivedAt) {
  return Array.from({ length: count }, (_, index) => ({
    image: { close() {} },
    chunk,
    index,
    receivedAt,
  }));
}

function enqueueChunk(controller, {
  chunk,
  eventId = 0,
  frameCount = 12,
  durationMs = 480,
  now,
  receivedAt = now,
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
    __received_at: receivedAt,
    is_final_frame_batch: true,
  }, frames(frameCount, chunk, receivedAt), now);
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

function slowServerPacesAtSourceFps() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  let now = 0;
  for (let chunk = 0; chunk < 8; chunk += 1) {
    now += 1360;
    enqueueChunk(controller, { chunk, durationMs: 1360, now });
    renderFor(controller, now, 1360);
  }
  const snapshot = controller.snapshot();
  assert.ok(snapshot.sourceFps > 8 && snapshot.sourceFps < 10);
  assert.ok(snapshot.renderFps < snapshot.sourceFps);
  assert.ok(snapshot.renderFps > snapshot.sourceFps * 0.9);
}

function smallBufferStartsFromFirstChunk() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  enqueueChunk(controller, {
    chunk: 0,
    durationMs: 750,
    now: 1000,
  });

  const decision = controller.render(1000, { hasPendingInput: true });

  assert.equal(decision.action, "draw");
  assert.equal(decision.frame.chunk, 0);
  assert.ok(decision.snapshot.targetLeadMs <= 450);
}

function smallBufferDrawsSingleFrameBatch() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  enqueueChunk(controller, {
    chunk: 0,
    frameCount: 1,
    durationMs: 63,
    now: 1000,
  });

  const decision = controller.render(1000, { hasPendingInput: true });

  assert.equal(decision.action, "draw");
  assert.equal(decision.frame.chunk, 0);
}

function jitterLeadWaitsForMoreThanOneTinyFrame() {
  const controller = new RealtimePlaybackController({
    targetFps: 12,
    holdForTargetLead: true,
    minTargetLeadMs: 220,
    maxTargetLeadMs: 420,
    minStartLeadMs: 160,
  });
  enqueueChunk(controller, {
    chunk: 0,
    frameCount: 1,
    durationMs: 83,
    now: 1000,
  });

  assert.equal(controller.render(1000, { hasPendingInput: true }).action, "hold");

  enqueueChunk(controller, {
    chunk: 0,
    frameCount: 2,
    durationMs: 166,
    now: 1030,
  });
  const decision = controller.render(1030, { hasPendingInput: true });

  assert.equal(decision.action, "draw");
  assert.equal(decision.frame.chunk, 0);
}

function smallBufferContinuesWhenMoreFramesArrive() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  enqueueChunk(controller, {
    chunk: 0,
    frameCount: 1,
    durationMs: 40,
    now: 1000,
  });
  assert.equal(controller.render(1000, { hasPendingInput: true }).action, "draw");

  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 6,
    durationMs: 240,
    now: 1100,
  });
  const decision = controller.render(1100, { hasPendingInput: true });

  assert.equal(decision.action, "draw");
  assert.equal(decision.frame.chunk, 1);
}

function smallBufferPacesSlowChunksAtSourceFps() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  enqueueChunk(controller, {
    chunk: 0,
    durationMs: 750,
    now: 1000,
  });
  assert.equal(controller.render(1000, { hasPendingInput: true }).action, "draw");

  const decision = controller.render(1041, { hasPendingInput: true });

  assert.equal(decision.action, "wait");
  assert.equal(controller.render(1063, { hasPendingInput: true }).action, "wait");
  assert.equal(controller.render(1073, { hasPendingInput: true }).action, "draw");
}

function burstySubTargetSourceKeepsWarmBuffer() {
  const controller = new RealtimePlaybackController({ targetFps: 16 });
  let now = 1000;
  let emptySnapshots = 0;
  for (let chunk = 0; chunk < 12; chunk += 1) {
    enqueueChunk(controller, {
      chunk,
      frameCount: 12,
      durationMs: 850,
      now,
    });
    renderFor(controller, now, 850);
    const snapshot = controller.snapshot();
    if (chunk > 3 && snapshot.queueFrames === 0) emptySnapshots += 1;
    now += 850;
  }

  const snapshot = controller.snapshot();
  assert.ok(snapshot.sourceFps > 13.5 && snapshot.sourceFps < 15);
  assert.equal(emptySnapshots, 0);
  assert.ok(snapshot.bufferMs > 0);
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

function eventCutoverKeepsOnlySmallOldFrameGrace() {
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
  assert.equal(result.droppedFrames.length, 21);
  assert.equal(controller.queue[0].chunk, 1);
  assert.equal(controller.queue[0].index, 0);
  assert.equal(controller.queue[3].chunk, 2);
}

function settleEventCutoverKeepsOnlySmallOldFrameGrace() {
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
  assert.equal(result.droppedFrames.length, 24);
  assert.equal(controller.queue[0].chunk, 2);
  assert.equal(controller.queue[0].index, 0);
}

function staleFramesAfterWallClockPauseResumeAtFreshestChunk() {
  const controller = new RealtimePlaybackController({ targetFps: 25 });
  for (let chunk = 0; chunk < 5; chunk += 1) {
    enqueueChunk(controller, {
      chunk,
      now: 1000 + chunk * 480,
      receivedAt: 1000 + chunk * 480,
    });
  }

  const resumedAt = 82000;
  let droppedFrames = 0;
  for (let chunk = 5; chunk < 9; chunk += 1) {
    const result = enqueueChunk(controller, {
      chunk,
      now: resumedAt + (chunk - 5) * 480,
      receivedAt: resumedAt + (chunk - 5) * 480,
    });
    droppedFrames += result.droppedFrames.length;
  }

  const decision = controller.render(resumedAt + 4 * 480, { hasPendingInput: true });
  droppedFrames += decision.droppedFrames.length;
  assert.equal(decision.action, "draw");
  assert.equal(decision.frame.chunk, 8);
  assert.ok(droppedFrames >= 48);
  assert.equal(controller.snapshot().lastDropReason, "backlog");
}

function timelineModeNeverDropsBacklog() {
  const controller = new RealtimePlaybackController({
    mode: "timeline",
    targetFps: 25,
  });
  let now = 100;
  for (let chunk = 0; chunk < 20; chunk += 1) {
    enqueueChunk(controller, { chunk, now, durationMs: 480 });
    now += 20;
  }
  const snapshot = controller.snapshot();
  assert.equal(snapshot.mode, "timeline");
  assert.equal(snapshot.droppedFrames, 0);
  assert.equal(snapshot.queueFrames, 240);
}

function timelineModePreservesFramesAcrossEventCutover() {
  const controller = new RealtimePlaybackController({
    mode: "timeline",
    targetFps: 25,
  });
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
  assert.equal(result.droppedFrames.length, 0);
  assert.equal(controller.snapshot().droppedFrames, 0);
  assert.equal(controller.queue.length, 36);
  assert.equal(controller.queue[24].eventId, 5);
}

function switchingBackToLiveTrimsTimelineBacklog() {
  const controller = new RealtimePlaybackController({
    mode: "timeline",
    targetFps: 25,
  });
  let now = 100;
  for (let chunk = 0; chunk < 20; chunk += 1) {
    enqueueChunk(controller, { chunk, now, durationMs: 480 });
    now += 20;
  }
  controller.setMode("live");
  const decision = controller.render(now, { hasPendingInput: true });
  assert.ok(decision.droppedFrames.length > 0);
  assert.equal(decision.snapshot.mode, "live");
}

function lowLatencyModeFollowsMeasuredSourceInsteadOfDrainingAtTargetFps() {
  const controller = new RealtimePlaybackController({
    targetFps: 24,
    lowLatencyPlayback: true,
    minTargetLeadMs: 80,
    maxTargetLeadMs: 180,
  });
  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 8,
    durationMs: 1000,
    now: 1000,
  });
  controller.render(1000, { hasPendingInput: true });
  const snapshot = controller.snapshot();
  assert.ok(snapshot.sourceFps >= 7.5 && snapshot.sourceFps <= 8.5);
  assert.ok(snapshot.renderFps <= 9, `render fps ${snapshot.renderFps}`);
}

function lowLatencyModeBoundsSingleChunkBacklogAndCutsOldActionImmediately() {
  const controller = new RealtimePlaybackController({
    targetFps: 16,
    lowLatencyPlayback: true,
    holdForTargetLead: false,
    minTargetLeadMs: 0,
    maxTargetLeadMs: 80,
    maxDeliveryLeadBoostMs: 30,
    lowLatencyMaxLeadFrames: 1,
  });
  enqueueChunk(controller, {
    chunk: 0,
    eventId: 0,
    frameCount: 12,
    durationMs: 750,
    now: 1000,
  });
  assert.ok(controller.snapshot().bufferMs <= controller.snapshot().maxLeadMs + 1);
  assert.ok(
    controller.snapshot().maxLeadMs <=
      controller.snapshot().targetLeadMs + 1000 / controller.snapshot().sourceFps + 1,
  );
  controller.noteInputEvent(5, 1010);
  const result = enqueueChunk(controller, {
    chunk: 1,
    eventId: 5,
    frameCount: 3,
    durationMs: 188,
    now: 1100,
  });
  assert.ok(result.cutover);
  assert.equal(controller.queue.some((frame) => frame.eventId < 5), false);
}

stableSourceDoesNotDrop();
slowServerPacesAtSourceFps();
smallBufferStartsFromFirstChunk();
smallBufferDrawsSingleFrameBatch();
jitterLeadWaitsForMoreThanOneTinyFrame();
smallBufferContinuesWhenMoreFramesArrive();
smallBufferPacesSlowChunksAtSourceFps();
burstySubTargetSourceKeepsWarmBuffer();
backlogDropsContiguousOldFrames();
eventCutoverKeepsOnlySmallOldFrameGrace();
settleEventCutoverKeepsOnlySmallOldFrameGrace();
staleFramesAfterWallClockPauseResumeAtFreshestChunk();
timelineModeNeverDropsBacklog();
timelineModePreservesFramesAcrossEventCutover();
switchingBackToLiveTrimsTimelineBacklog();
lowLatencyModeFollowsMeasuredSourceInsteadOfDrainingAtTargetFps();
lowLatencyModeBoundsSingleChunkBacklogAndCutsOldActionImmediately();
