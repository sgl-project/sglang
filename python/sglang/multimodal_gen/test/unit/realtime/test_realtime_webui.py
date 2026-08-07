# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_realtime_webui_uses_same_origin_server_by_default():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    proxy_server = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/server.py"
    ).read_text()

    assert "`${protocol}//${window.location.host}/v1/realtime_video/generate`" in app_js
    assert 'app.router.add_get("/v1/realtime_video/generate"' in proxy_server
    assert 'app.router.add_route("*", "/v1/{path:.*}"' in proxy_server


def test_realtime_webui_supports_deployment_runtime_config():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    index_html = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/index.html"
    ).read_text()
    proxy_server = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/server.py"
    ).read_text()

    assert 'os.environ.get("REALTIME_UI_CONFIG_JSON", "{}")' in proxy_server
    assert 'app.router.add_get("/runtime-config.js", _runtime_config)' in proxy_server
    assert '<script src="./runtime-config.js"></script>' in index_html
    assert 'configuredNumber("targetFps", 16)' in app_js
    assert "UI_CONFIG.targetFps == null ? preset.fps : DEFAULT_TARGET_FPS" in app_js


def test_realtime_webui_supports_explicit_minwm_t2v_sessions():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    index_html = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/index.html"
    ).read_text()

    assert 'id="generationMode"' in index_html
    assert '<option value="t2v">Text to video (T2V)</option>' in index_html
    assert 'id="referenceSection"' in index_html
    assert 'id="t2vFrameHint"' in index_html
    assert "styles.css?v=realtime-t2v-dump-trace-v1" in index_html
    assert "playback_controller.js?v=realtime-playback-v22" in index_html
    assert "app.js?v=realtime-production-gateway-v4" in index_html
    assert "UI_CONFIG.generationModes" in app_js
    assert "UI_CONFIG.generationMode" in app_js
    assert "CONFIGURED_DEFAULT_GENERATION_MODE" in app_js
    assert "generation_mode: generationMode" in app_js
    assert 'generationMode === "i2v"' in app_js
    assert 'const continuousT2V = generationMode === "t2v"' in app_js
    assert "numFrames = continuousT2V ? undefined : readT2VNumFrames()" in app_js
    assert "num_frames: continuousT2V ? undefined : numFrames" in app_js
    assert 'max_chunks: generationMode === "t2v"' in app_js
    assert '$(' + '"continuous"' + ').disabled = false' in app_js
    assert "let savedT2VContinuous = true" in app_js
    assert '$(' + '"continuous"' + ').checked = savedT2VContinuous' in app_js
    assert '"Continuous T2V session"' in app_js
    assert '$("referenceSection").hidden = isT2V' in app_js


def test_realtime_webui_presets_do_not_emit_camera_scripts():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    index_html = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/index.html"
    ).read_text()
    styles_css = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/styles.css"
    ).read_text()

    assert "preset.actions" not in app_js
    assert "repeatActions" not in app_js
    assert 'id="eventFrames"' not in index_html
    assert "ControlStateController" in app_js
    assert 'const DEFAULT_PREVIEW_OUTPUT_FORMAT = "webp";' in app_js
    assert 'id="transportFormat"' in index_html
    assert 'id="fps" type="number" value="16"' in index_html
    assert 'id="superResolution" type="checkbox"' in index_html
    assert 'id="upscalingScale"' in index_html
    assert 'class="workspace"' in index_html
    assert 'class="preview-frame"' in index_html
    assert 'id="previewOverlay" class="preview-overlay"' in index_html
    assert 'id="previewScale" type="range" min="80" max="170" value="100"' in index_html
    assert 'id="previewScaleText"' in index_html
    assert 'id="outputSizeText"' in index_html
    assert 'id="frameInterpolation" type="checkbox" />' in index_html
    assert (
        'id="serverUrl" value="ws://127.0.0.1:30000/v1/realtime_video/generate"'
        in index_html
    )
    assert '<option value="webp" selected>WebP preview</option>' in index_html
    assert 'id="serverSendText"' in index_html
    assert 'id="theoreticalFpsText"' in index_html
    assert 'id="renderFps"' in index_html
    assert 'id="stageRenderFps"' not in index_html
    assert "sglang-diffusion Realtime Studio" in index_html
    assert "SGLD" not in index_html
    assert 'class="tabs"' not in index_html
    assert "Recordings" not in index_html
    assert "API" not in index_html
    assert "Info" not in index_html
    assert 'id="steps" type="number" value="4"' in index_html
    assert 'id="guidance" type="number" value="1"' in index_html
    assert "styles.css?v=realtime-t2v-dump-trace-v1" in index_html
    assert "app.js?v=realtime-production-gateway-v4" in index_html
    assert 'const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v10";' in app_js
    assert 'const DEFAULT_TARGET_FPS = configuredNumber("targetFps", 16);' in app_js
    assert "const DEFAULT_FRAME_INTERPOLATION_EXP = 1;" in app_js
    assert "const DEFAULT_FRAME_INTERPOLATION_SCALE = 1.0;" in app_js
    assert "const DEFAULT_UPSCALING_SCALE = 2;" in app_js
    assert "const DEFAULT_PREVIEW_SCALE = 100;" in app_js
    assert 'setPreviewState("waiting")' in app_js
    assert "stage.dataset.previewState = state" in app_js
    assert "previewProgressSpin" in styles_css
    assert "previewDotPulse" not in styles_css
    assert 'document.querySelector(".preview-frame")' in app_js
    assert 'previewFrame.style.setProperty("--preview-scale"' in app_js
    assert "cancelAnimationFrame(previewScaleFrame)" in app_js
    assert "enable_frame_interpolation: true" in app_js
    assert "frame_interpolation_exp: DEFAULT_FRAME_INTERPOLATION_EXP" in app_js
    assert "frame_interpolation_scale: DEFAULT_FRAME_INTERPOLATION_SCALE" in app_js
    assert "readSuperResolutionParams()" in app_js
    assert "enable_upscaling: true" in app_js
    assert "upscaling_scale: readUpscalingScale()" in app_js
    assert "updateOutputSizeFromHeader(header)" in app_js
    assert "setPreviewScale(DEFAULT_PREVIEW_SCALE)" in app_js
    assert "preview_scale" in app_js
    assert "sr_scale" in app_js
    assert "playbackController.render(now" in app_js
    assert "playbackController.enqueueDecodedFrames(header, decodedFrames, now)" in app_js
    assert (
        'const REACTOR_PRESET_BASE_URL = "https://www.reactor.inc/lingbot-world-fast-v1";'
        in app_js
    )
    assert "Dragon Dolly" in app_js
    assert "no creature morphing" in app_js
    assert "A static album-cover view matching the reference image" in app_js
    assert "lighthouse remains on the left with its white reflection path" in app_js
    assert "subtle star twinkle" in app_js
    assert "Ziggy Stardust" in app_js
    assert "blue K. West sign" in app_js
    assert "wet pavement reflecting a yellow streetlamp" in app_js
    assert "ZiggyStardust.jpg" in app_js
    assert "A slow aerial orbit around a pastel floating island hotel" not in app_js
    assert app_js.index("Dragon Ride") < app_js.index("Dragon Dolly")
    assert app_js.index("Ziggy Stardust") < app_js.index("Plastic Beach")
    assert app_js.index("Dragon Dolly") < app_js.index("Kid A")
    assert "dragon-ride.jpg" in app_js
    assert 'referenceUrl: "./assets/dragon-ride.jpg"' in app_js
    assert "function createPresetThumbFallback" in app_js
    assert "thumb.onerror = () => thumb.replaceWith(createPresetThumbFallback(preset))" in app_js
    assert "reference image unavailable" in app_js
    assert ".preset-thumb-fallback" in styles_css
    assert (
        repo_root
        / "python/sglang/multimodal_gen/apps/realtime_webui/assets/dragon-ride.jpg"
    ).stat().st_size > 0
    assert "stageRenderFps" not in app_js
    assert 'setStatus("Receiving", "live")' in app_js
    assert "pumpDecodeQueue()" in app_js
    assert "receiveChain" not in app_js
    assert 'message.type === "chunk_stats"' not in app_js
    assert "function updateServerChunkStats" not in app_js
    assert ".stage-stat" in styles_css
    assert ".workspace" in styles_css
    assert ".preview-frame" in styles_css
    assert ".preview-overlay" in styles_css
    assert "@keyframes previewProgressSpin" in styles_css
    assert ".preview-scale-control" in styles_css
    assert "--preview-scale" in styles_css


def test_realtime_webui_exports_replayable_recording_artifacts_on_t2v_branch():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    index_html = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/index.html"
    ).read_text()
    styles_css = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/styles.css"
    ).read_text()

    assert '<option value="t2v">Text to video (T2V)</option>' in index_html
    assert 'id="recordFolderBtn"' in index_html
    assert "let currentSessionArtifact = null;" in app_js
    assert "function createSessionArtifact" in app_js
    assert "function recordTrajectoryEvent" in app_js
    assert "function saveRecordingArtifactFiles" in app_js
    assert "function buildReplayHtml" in app_js
    assert "function drawRecordingStageFrame" in app_js
    assert "function drawRecordingControls" in app_js
    assert "MediaRecorder" in app_js
    assert "mediarecorder-webm" in app_js
    assert "generation_mode: generationMode" in app_js
    assert "reference_image" in app_js
    assert "artifact.reference_image = referenceImage || null;" in app_js
    assert "first_frame_sha256" in app_js
    assert "prompt_history" in app_js
    assert "prompt_update" in app_js
    assert "key_down" in app_js
    assert "key_up" in app_js
    assert "camera_actions_sent" in app_js
    assert "server_chunk_stats" in app_js
    assert "frame_batch_received" in app_js
    assert "client.chunk_first_rendered" in app_js
    assert "video/mp4" in app_js
    assert "webcodecs-mp4" in app_js
    assert "video/webm" in app_js
    assert ".json" in app_js
    assert ".html" in app_js
    assert "recordingAssetBaseUrl" in app_js
    assert ".record-folder-button" in styles_css


def test_realtime_webui_exposes_live_trace_topology_with_dump_trace_id():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    index_html = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/index.html"
    ).read_text()
    styles_css = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/styles.css"
    ).read_text()

    assert 'id="tracePaneButton"' in index_html
    assert 'id="traceTopology"' in index_html
    assert "trace_topology.js?v=" in index_html
    assert 'id="traceVaeEncodeText"' in index_html
    assert 'id="traceDenoiseText"' in index_html
    assert 'id="traceVaeDecodeText"' in index_html
    assert "const traceTopologyApi = window.SGLangRealtimeTraceTopology || {};" in app_js
    assert "function traceWebSocketUrl" in app_js
    assert 'message.type === "chunk_stats"' not in app_js
    assert "currentSessionArtifact.trace_id = currentTrace.traceId" in app_js
    assert "traceHttpClient?.enqueueClientEvent(event)" in app_js
    assert "traceHttpClient?.setActive(true, 5000)" in app_js
    assert "traceHttpClient?.setActive(false)" in app_js
    assert "traceTopology?.setAggregate?.(aggregate)" in app_js
    assert 'id="traceObservedText"' in index_html
    assert "client_trace:" not in app_js
    assert 'message.type === "trace_event"' not in app_js
    assert "trace-panel" in styles_css
    assert ".trace-node" in styles_css


def test_realtime_webui_uses_frame_metadata_for_live_business_status():
    repo_root = Path(__file__).resolve().parents[6]
    app_js = (
        repo_root / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()

    assert "lastSampledEventId = Number(header.event_id || lastSampledEventId)" in app_js
    assert "formatBytes(payloadBytes)" in app_js
    assert "playback.sourceFps.toFixed(1)" in app_js
