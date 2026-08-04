# MinWM Runtime Transport And Latency Optimization Design

## Goal

Remove runtime dependency installation, move diagnostic trace traffic off the
video WebSocket, and reduce browser receive-to-display latency without changing
the generated video or the asynchronous Denoiser-to-VAE protocol.

## Dependency Image

Build one immutable MinWM realtime runtime image from the existing CUDA/PyTorch
training base. The image contains the checked-out SGLang source, diffusion
dependencies, TAEHV package, and the verified `taew2_2.pth` checkpoint. Both the
Denoiser and VAE worker use this image. Model checkpoints remain external S3
artifacts because they are large and change independently from runtime code.

The image is layered in this order:

1. Existing CUDA, PyTorch, and system runtime base.
2. TAEHV package and its checksum-verified small checkpoint.
3. SGLang Python dependencies and source.
4. Role-neutral runtime entrypoints.

Pods must not clone Git repositories, install Python packages, or download
TAEHV during startup. A build-and-push helper emits an immutable ECR tag based
on the Git SHA, and Kubernetes manifests reference a replaceable image URI.

## Trace Transport

Full server trace events continue to be emitted as structured
`realtime_trace` log records. Production log collection sends these records to
CloudWatch with the existing five-day retention policy.

For the WebUI, each API process also keeps a bounded, five-minute in-memory
trace window. The buffer has explicit maximum trace and event counts and is
not durable. A read-only HTTP endpoint returns incremental events by cursor.
The WebUI polls it only while the Trace tab is visible.

Browser trace events use a separate batched HTTP POST endpoint. They are never
written to the video WebSocket. The video WebSocket retains only initialization,
controls, acknowledgements, chunk statistics, and frame payloads. `trace_id`
remains lightweight correlation metadata on the generation connection.

The HTTP trace path is diagnostic. Failures must not close or delay video
generation, and retries are bounded. Existing WebSocket trace messages remain
accepted by the client for compatibility but the server no longer emits them.

## Display Latency

The default live playback profile targets a 100-200 ms queue rather than the
current 220-420 ms smoothing lead. It starts on the first decodable frame,
does not wait for a target lead, trims stale event frames immediately, and
caps adaptive jitter growth. Timeline mode remains lossless and unchanged.

The profile uses:

- `lowLatencyPlayback: true`
- `holdForTargetLead: false`
- target lead range of 80-180 ms
- startup/resume lead of one frame
- maximum delivery jitter boost of 60 ms
- zero old-event grace frames

This intentionally favors action responsiveness over perfectly uniform frame
cadence. It cannot make display latency zero because WebP decoding, browser
animation scheduling, network jitter, and at least one decoded frame remain.

## Verification

Local verification covers the bounded trace store, HTTP trace endpoints,
absence of server WebSocket trace delivery, client HTTP polling/batching,
low-latency playback behavior, and image policy manifests.

End-to-end verification uses the existing disposable Spot topology: one H100
Denoiser and one low-cost L4 VAE worker, falling back to L40S only if needed.
The report records image pull/startup time, video WebSocket message types,
trace endpoint correctness, warm display-lag percentiles, generation latency,
and dropped-frame behavior. All temporary GPU resources and the public load
balancer are deleted after the run.
