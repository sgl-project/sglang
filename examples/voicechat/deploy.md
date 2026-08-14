# Deploy and run

The realtime adapter uses two processes on the same GPU:

- The NeMo sidecar loads the causal perception encoder and audio codec from
  the unified checkpoint and listens only on `127.0.0.1:18081`.
- The SGLang server loads the converted Duplex and EarTTS stages and exposes
  HTTP/WebSocket endpoints on port `18080`.

Both containers use host networking so the SGLang server can reach the
localhost-only sidecar. The deployment is intentionally limited to one active
conversation.

## Prerequisites

- Complete [Prerequisites](prerequisites.md).
- Complete [Generate SGLang stages](generate-model-repo.md).
- Run commands from the SGLang checkout containing this integration.
- Replace `<gpu-uuid>` with the UUID reported by
  `nvidia-smi --query-gpu=uuid --format=csv,noheader`.
- Export the same absolute artifact path used during conversion, for example
  `export VOICECHAT_DATA=/absolute/path/to/voicechat-data`.

## Step 1: Start the NeMo audio sidecar

```bash
docker run -d --rm \
  --name sglang-voicechat-audio \
  --gpus 'device=<gpu-uuid>' \
  --network host \
  --shm-size=8g \
  -v "$PWD:/workspace/sglang:ro" \
  -v "$VOICECHAT_DATA/checkpoint:/checkpoint:ro" \
  -w /workspace/sglang \
  --entrypoint python \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest \
  examples/voicechat/nemo_audio_sidecar.py /checkpoint
```

Verify that the sidecar loaded perception and codec weights:

```bash
curl --fail http://127.0.0.1:18081/health
```

The response includes `loaded_tensors`, `perception_cudagraph`, and the active
audio session, which should initially be `null`.

## Step 2: Start the SGLang server

```bash
docker run -d --rm \
  --name sglang-voicechat-server \
  --gpus 'device=<gpu-uuid>' \
  --network host \
  --shm-size=8g \
  -e PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang \
  -v "$VOICECHAT_DATA/converted:/models:ro" \
  -w /sgl-workspace/sglang \
  --entrypoint python \
  sglang-voicechat \
  examples/voicechat/online_server.py \
    --duplex-model /models/duplex \
    --eartts-model /models/eartts \
    --host 0.0.0.0 \
    --port 18080
```

The Duplex thinker defaults to `bfloat16` for interactive latency while its
Mamba recurrent state remains fp32. Pass `--duplex-dtype float32` when
comparing against fp32 NeMo reference outputs. EarTTS and the audio sidecar
remain fp32 in both modes.

Startup includes a disposable two-frame warm-up. The HTTP socket becomes ready
after the first MaskGIT compilation and all temporary warm-up sessions have
been released. Poll readiness:

```bash
until curl --fail --silent http://127.0.0.1:18080/v1/realtime/health; do
  sleep 5
done
```

Do not disable warm-up for normal serving. `--skip-warmup` is intended only for
diagnosing startup behavior.

## Microphone conversation

Install the laptop dependencies described in [Prerequisites](prerequisites.md),
then list audio devices if the defaults are not correct:

```bash
python examples/voicechat/client.py --list-devices
```

Start microphone capture and speaker playback:

```bash
python examples/voicechat/client.py \
  --url ws://<voicechat-host>:18080/v1/realtime \
  --output-wav response.wav
```

Speak normally, then press Enter to stop capture. The client sends trailing
silence, waits for every queued frame, and saves the complete 22.05 kHz response
to `response.wav`. Use `--input-device-index` and `--output-device-index` to
select non-default devices, `--no-playback` to disable speakers, or
`--microphone-seconds <seconds>` for a fixed-duration capture.
The client opens each device at its advertised native sample rate, resamples at
the model boundary, and prints an input level about once per second. A level
that remains `rms=0 peak=0` indicates a device or operating-system permission
problem before audio reaches the server.

For best results, use headphones and a quiet room. Speaker feedback into the
microphone and background noise can degrade a full-duplex model.

## WAV conversation

WAV input must be PCM16. The client downmixes multichannel input to mono, resamples
it to 16 kHz, streams fixed 80 ms frames, and writes 22.05 kHz PCM16 output.

```bash
python examples/voicechat/client.py \
  --url ws://<voicechat-host>:18080/v1/realtime \
  --input-wav question.wav \
  --output-wav response.wav \
  --no-playback
```

The default two seconds of trailing silence is sufficient for the tested short
prompt. Increase `--trailing-silence` when evaluating longer responses because
the full-duplex model emits output only while input frames continue arriving.
The client prints a truncation warning if non-padding text tokens occur in the
final 12 frames; repeat with more trailing silence when this warning appears.

## Observe and stop

Inspect service logs:

```bash
docker logs --follow sglang-voicechat-audio
docker logs --follow sglang-voicechat-server
```

Stop only the two named containers:

```bash
docker stop sglang-voicechat-server sglang-voicechat-audio
```

## Troubleshooting

### Server is not ready

- Confirm `curl http://127.0.0.1:18081/health` succeeds first.
- Inspect both named container logs.
- Allow additional time for first-run MaskGIT compilation.
- Confirm both containers use `--network host`.

### GPU out of memory

- Stop unrelated GPU workloads.
- Confirm only one copy of each service is running.
- Keep the defaults of `0.45` Duplex and `0.20` EarTTS static memory.
- As a diagnostic, lower `--duplex-memory-fraction` or
  `--eartts-memory-fraction` slightly; too little static memory can fail later
  when the KV cache grows.

### Audio backlog exceeded

The input source is outrunning the causal pipeline. Keep WAV realtime pacing
enabled, verify the GPU is not shared with another heavy workload, and inspect
the health and per-frame queue timing. `--no-realtime-pacing` is a benchmark
option, not an interactive setting.

### Microphone or playback cannot open

Run `--list-devices`, select explicit device indexes, and verify that the
microphone level changes while speaking. Native 24 kHz and 48 kHz CoreAudio
devices are resampled automatically.

### Model shard is reported missing

If the named safetensors shard exists on the host, verify that the image user
can read it and that the backing filesystem supports memory-mapped files:

```bash
docker run --rm \
  -v "$VOICECHAT_DATA/converted:/models:ro" \
  --entrypoint python \
  sglang-voicechat -c '
from pathlib import Path
from safetensors import safe_open
shard = next(Path("/models/duplex").glob("*.safetensors"))
safe_open(str(shard), framework="pt", device="cpu")
'
```

Correct permissions only within the explicit artifact directory. If files are
on CIFS or another incompatible shared filesystem, copy `converted/` to local
POSIX storage and update `VOICECHAT_DATA`; changing the container user does not
fix a filesystem that safetensors cannot open or memory-map.

Next: [API reference](api-reference.md)
