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

function timelineModeDrainsBacklogOnEveryRenderTick() {
  const controller = new RealtimePlaybackController({
    mode: "timeline",
    targetFps: 50,
  });
  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 10,
    durationMs: 1000,
    now: 1000,
  });

  const first = controller.render(1000, { hasPendingInput: true });
  const second = controller.render(1016, { hasPendingInput: true });

  assert.equal(first.action, "draw");
  assert.equal(second.action, "draw");
  assert.equal(second.frame.index, 1);
  assert.equal(controller.snapshot().queueFrames, 8);
  assert.equal(controller.snapshot().renderFps, 50);
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

function smoothTimelineModePreservesBacklogAndCatchesUp() {
  const controller = new RealtimePlaybackController({
    mode: "smooth_timeline",
    targetFps: 25,
    minTargetLeadMs: 200,
    maxTargetLeadMs: 400,
    maxLeadExtraChunkRatio: 0.2,
  });
  let now = 100;
  for (let chunk = 0; chunk < 20; chunk += 1) {
    enqueueChunk(controller, { chunk, now, durationMs: 480 });
    now += 20;
  }
  const snapshot = controller.snapshot();
  assert.equal(snapshot.mode, "smooth_timeline");
  assert.equal(snapshot.droppedFrames, 0);
  assert.equal(snapshot.queueFrames, 240);

  const decision = controller.render(now, { hasPendingInput: true });
  const catchUp = controller.snapshot();
  assert.equal(decision.action, "draw");
  assert.equal(catchUp.droppedFrames, 0);
  assert.ok(catchUp.playbackRate > 1.5, `playback rate ${catchUp.playbackRate}`);
  assert.ok(catchUp.renderFps > 40, `render fps ${catchUp.renderFps}`);
}

function smoothTimelineModePreservesFramesAcrossEventCutover() {
  const controller = new RealtimePlaybackController({
    mode: "smooth_timeline",
    targetFps: 25,
    minTargetLeadMs: 1600,
    maxTargetLeadMs: 2400,
    maxLeadExtraChunkRatio: 1.0,
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
  assert.equal(controller.queue[0].eventId, 0);
  assert.equal(controller.queue[24].eventId, 5);
}

function smoothTimelineModePacesInsteadOfDrainingEveryRenderTick() {
  const controller = new RealtimePlaybackController({
    mode: "smooth_timeline",
    targetFps: 50,
  });
  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 10,
    durationMs: 1000,
    now: 1000,
  });

  const first = controller.render(1000, { hasPendingInput: true });
  const second = controller.render(1016, { hasPendingInput: true });

  assert.equal(first.action, "draw");
  assert.equal(second.action, "wait");
  assert.equal(controller.snapshot().queueFrames, 9);
  assert.ok(controller.snapshot().renderFps < 50);
}

function smoothTimelineModeSpeedsUpToCatchBacklogWithoutDropping() {
  const controller = new RealtimePlaybackController({
    mode: "smooth_timeline",
    targetFps: 24,
    holdForTargetLead: true,
    targetLeadChunkRatio: 0.75,
    minTargetLeadMs: 600,
    maxTargetLeadMs: 1200,
  });
  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 48,
    durationMs: 2000,
    now: 1000,
  });

  const decision = controller.render(1000, { hasPendingInput: true });
  const snapshot = controller.snapshot();

  assert.equal(decision.action, "draw");
  assert.equal(snapshot.droppedFrames, 0);
  assert.ok(snapshot.playbackRate > 1, `playback rate ${snapshot.playbackRate}`);
  assert.ok(snapshot.playbackRate <= 2.5, `playback rate ${snapshot.playbackRate}`);
}

function adaptiveModeKeepsBoundedBacklogWithoutLowLatencyJump() {
  const controller = new RealtimePlaybackController({
    mode: "adaptive",
    targetFps: 25,
    lowLatencyPlayback: true,
    holdForTargetLead: true,
    minTargetLeadMs: 220,
    maxTargetLeadMs: 420,
    lowLatencyMaxLeadFrames: 1,
  });
  let now = 100;
  for (let chunk = 0; chunk < 13; chunk += 1) {
    enqueueChunk(controller, { chunk, now, durationMs: 480 });
    now += 20;
  }
  const snapshot = controller.snapshot();
  assert.equal(snapshot.mode, "adaptive");
  assert.ok(snapshot.droppedFrames > 0);
  assert.equal(snapshot.lastDropReason, "backlog");
}

function adaptiveModeCutsActiveInputWithoutOldFrameGrace() {
  const controller = new RealtimePlaybackController({
    mode: "adaptive",
    targetFps: 25,
    lowLatencyPlayback: true,
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
  assert.equal(result.droppedFrames.length, 24);
  assert.equal(controller.queue[0].chunk, 2);
  assert.equal(controller.queue[0].eventId, 5);
}

function adaptiveModeDropsBufferedFramesForActiveInputCutover() {
  const controller = new RealtimePlaybackController({
    mode: "adaptive",
    targetFps: 24,
    lowLatencyPlayback: true,
    holdForTargetLead: true,
    minTargetLeadMs: 420,
    minStartLeadMs: 420,
    minResumeLeadMs: 420,
  });
  enqueueChunk(controller, {
    chunk: 1,
    eventId: 0,
    frameCount: 16,
    durationMs: 1000,
    now: 1000,
  });
  controller.noteInputEvent(7, 1050, { cutoverMode: "motion" });
  const result = enqueueChunk(controller, {
    chunk: 2,
    eventId: 7,
    frameCount: 1,
    durationMs: 63,
    now: 1150,
  });

  assert.ok(result.cutover);
  assert.equal(result.droppedFrames.length, 16);
  assert.equal(controller.queue.length, 1);
  assert.equal(controller.queue[0].eventId, 7);
}

function adaptiveModeRendersCutoverFrameWithoutWaitingForBufferLead() {
  const controller = new RealtimePlaybackController({
    mode: "adaptive",
    targetFps: 24,
    lowLatencyPlayback: true,
    holdForTargetLead: true,
    minTargetLeadMs: 420,
    minStartLeadMs: 420,
    minResumeLeadMs: 420,
  });
  enqueueChunk(controller, {
    chunk: 1,
    eventId: 0,
    frameCount: 16,
    durationMs: 1000,
    now: 1000,
  });
  controller.noteInputEvent(7, 1050, { cutoverMode: "motion" });
  enqueueChunk(controller, {
    chunk: 2,
    eventId: 7,
    frameCount: 1,
    durationMs: 63,
    now: 1150,
  });

  const decision = controller.render(1150, { hasPendingInput: true });

  assert.equal(decision.action, "draw");
  assert.equal(decision.frame.eventId, 7);
}

function deliveryFpsCapsOptimisticServerFps() {
  const controller = new RealtimePlaybackController({
    mode: "adaptive",
    targetFps: 24,
    holdForTargetLead: true,
  });
  enqueueChunk(controller, {
    chunk: 0,
    frameCount: 16,
    durationMs: 670,
    now: 1000,
    receivedAt: 1000,
  });
  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 16,
    durationMs: 670,
    now: 3400,
    receivedAt: 3400,
  });

  const snapshot = controller.snapshot();

  assert.ok(snapshot.serverFps > 22, `server fps ${snapshot.serverFps}`);
  assert.ok(snapshot.deliveryFps < 7, `delivery fps ${snapshot.deliveryFps}`);
  assert.ok(snapshot.sourceFps < 7, `source fps ${snapshot.sourceFps}`);
}

function deliveryCadenceExpandsAdaptiveLeadWindow() {
  const controller = new RealtimePlaybackController({
    mode: "adaptive",
    targetFps: 24,
    holdForTargetLead: true,
    targetLeadChunkRatio: 0.75,
    minTargetLeadMs: 360,
    maxTargetLeadMs: 900,
  });
  enqueueChunk(controller, {
    chunk: 0,
    frameCount: 16,
    durationMs: 670,
    now: 1000,
    receivedAt: 1000,
  });
  enqueueChunk(controller, {
    chunk: 1,
    frameCount: 16,
    durationMs: 670,
    now: 3400,
    receivedAt: 3400,
  });

  const snapshot = controller.snapshot();

  assert.ok(snapshot.targetLeadMs >= 850, `target lead ${snapshot.targetLeadMs}`);
  assert.ok(snapshot.maxLeadMs > 1800, `max lead ${snapshot.maxLeadMs}`);
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

function lowLatencyModePreservesNewestChunkAndCutsOldActionImmediately() {
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
    frameCount: 16,
    durationMs: 750,
    now: 1000,
  });
  assert.equal(controller.snapshot().droppedFrames, 0);
  assert.equal(controller.snapshot().queueFrames, 16);
  controller.noteInputEvent(5, 1010);
  const result = enqueueChunk(controller, {
    chunk: 1,
    eventId: 5,
    frameCount: 3,
    durationMs: 188,
    now: 1100,
  });
  assert.ok(result.cutover);
  assert.equal(result.droppedFrames.length, 16);
  assert.equal(controller.snapshot().queueFrames, 3);
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
timelineModeDrainsBacklogOnEveryRenderTick();
timelineModePreservesFramesAcrossEventCutover();
smoothTimelineModePreservesBacklogAndCatchesUp();
smoothTimelineModePreservesFramesAcrossEventCutover();
smoothTimelineModePacesInsteadOfDrainingEveryRenderTick();
smoothTimelineModeSpeedsUpToCatchBacklogWithoutDropping();
adaptiveModeKeepsBoundedBacklogWithoutLowLatencyJump();
adaptiveModeCutsActiveInputWithoutOldFrameGrace();
adaptiveModeDropsBufferedFramesForActiveInputCutover();
adaptiveModeRendersCutoverFrameWithoutWaitingForBufferLead();
deliveryFpsCapsOptimisticServerFps();
deliveryCadenceExpandsAdaptiveLeadWindow();
switchingBackToLiveTrimsTimelineBacklog();
lowLatencyModeFollowsMeasuredSourceInsteadOfDrainingAtTargetFps();
lowLatencyModePreservesNewestChunkAndCutsOldActionImmediately();
