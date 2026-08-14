# Offline WAV inference

The offline runner processes one prerecorded WAV through direct SGLang
`Engine` sessions and writes the same primary artifacts as NVIDIA's NeMo
offline example. It does not start or call the SGLang WebSocket server.

NeMo perception and codec operations still run in the localhost sidecar because
they are not SGLang autoregressive stages. Duplex and EarTTS run directly in
the offline process through `SGLangVoiceChatSession`.
Perception, Thinker, Talker, and codec work in four ordered async pipeline stages.

## Prerequisites

Complete [Prerequisites](prerequisites.md) and
[Generate SGLang stages](generate-model-repo.md). Then set these absolute host
paths:

```bash
# SGLang checkout containing examples/voicechat/offline_inference.py.
export SGLANG_REPO=/absolute/path/to/sglang-fork

# Parent directory containing checkpoint/ and converted/:
#   $VOICECHAT_DATA/checkpoint/        unified NVIDIA checkpoint
#   $VOICECHAT_DATA/converted/duplex/  converted Duplex stage
#   $VOICECHAT_DATA/converted/eartts/  converted EarTTS stage
export VOICECHAT_DATA=/absolute/path/to/voicechat-data

# Input directory and basename are separate so the Docker mount is explicit.
export VOICECHAT_INPUT_DIR=/absolute/path/to/input-wavs
export VOICECHAT_WAV=sample.wav

# Existing writable directory for generated outputs.
export VOICECHAT_OUTPUT_DIR=/absolute/path/to/voicechat-outputs

# Allocated GPU UUID and local SGLang image tag.
export VOICECHAT_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)"
export SGLANG_VOICECHAT_IMAGE=sglang-voicechat
```

`VOICECHAT_DATA` points to the parent of both checkpoint layouts, not directly
to the unified checkpoint or either converted stage.

The input must be 16-bit PCM WAV. Multichannel input is downmixed to mono and
the audio is resampled to 16 kHz when necessary.

## Start the NeMo audio sidecar

```bash
cd "$SGLANG_REPO"

docker run -d --rm \
  --name sglang-voicechat-audio \
  --gpus "device=$VOICECHAT_GPU_UUID" \
  --network host \
  --shm-size=8g \
  -v "$SGLANG_REPO:/workspace/sglang:ro" \
  -v "$VOICECHAT_DATA/checkpoint:/checkpoint:ro" \
  -w /workspace/sglang \
  --entrypoint python \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest \
  examples/voicechat/nemo_audio_sidecar.py /checkpoint

curl --fail http://127.0.0.1:18081/health
```

## Run direct offline inference

```bash
cd "$SGLANG_REPO"

docker run --rm \
  --gpus "device=$VOICECHAT_GPU_UUID" \
  --network host \
  --shm-size=8g \
  -e PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang \
  -v "$SGLANG_REPO:/sgl-workspace/sglang:ro" \
  -v "$VOICECHAT_DATA/converted:/models:ro" \
  -v "$VOICECHAT_INPUT_DIR:/input:ro" \
  -v "$VOICECHAT_OUTPUT_DIR:/output" \
  -w /sgl-workspace/sglang \
  --entrypoint python \
  "$SGLANG_VOICECHAT_IMAGE" \
  examples/voicechat/offline_inference.py \
    --duplex-model /models/duplex \
    --eartts-model /models/eartts \
    --wav "/input/$VOICECHAT_WAV" \
    --output-dir /output
```

Offline mode skips the disposable warm-up session by default because a one-shot
job does not reuse its compiled session. Pass `--warmup` to run the configured
`--warmup-frames` before the actual file when diagnosing or benchmarking startup.
The runner appends two seconds of silence by default so the full-duplex model has
time to finish a short reply. Increase this for longer replies with
`--trailing-silence <seconds>`.

For an input named `sample.wav`, the output directory receives:

- `sample_output.txt`: decoded text-channel output.
- `sample_output.wav`: mono 22.05 kHz agent audio.
- `sample_combined.wav`: stereo 22.05 kHz audio with the user on the left and
  agent on the right.
- `sample_output.json`: text/function token channels and frame/sample-rate
  metadata for debugging.

Stop the named sidecar when finished:

```bash
docker stop sglang-voicechat-audio
```

## Current scope

This initial path handles one WAV per invocation and greedy frame-locked
generation. It preserves function-channel predictions in the JSON metadata,
but does not yet implement NVIDIA's separate two-pass function-response
injection flow. The implementation is confined to the VoiceChat example and
uses the existing persistent-session API; it does not change scheduling or
sampling behavior for unrelated SGLang models.
