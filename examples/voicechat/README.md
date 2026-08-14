# NVIDIA NemotronLabs VoiceChat online inference

This integration follows NVIDIA's online architecture: NeMo runs the causal
Conformer perception encoder and EarTTS audio codec, while two persistent
SGLang engines serve the autoregressive Duplex and EarTTS stages. The engines
exchange one text token and one 31-codebook codec frame every 80 ms step.

Convert the unified checkpoint into `duplex/` and `eartts/` with
`convert_duplex_stage.py` and `convert_eartts_stage.py`. The second conversion
must run in NVIDIA's VoiceChat NeMo environment and uses a GPU to precompute
the character-aware subword lookup.

Both engines currently require eager execution. EarTTS must use float32 to
match NVIDIA's audio-quality requirement, so SGLang conservatively selects its
native PyTorch attention backend while preserving other models' defaults. On a
single H100, start Duplex with approximately 45% static memory and EarTTS with
20%; leave the remainder for NeMo perception, the codec, and temporary MaskGIT
buffers.
The intended programmatic setup is:

```python
from sglang import Engine
from online_session import SGLangVoiceChatSession

duplex = Engine(
    model_path="duplex",
    dtype="bfloat16",
    mem_fraction_static=0.45,
    max_running_requests=2,
    enable_streaming_session=True,
)
eartts = Engine(
    model_path="eartts",
    dtype="float32",
    mem_fraction_static=0.20,
    max_running_requests=2,
    enable_streaming_session=True,
)

session = SGLangVoiceChatSession(duplex, eartts)
session.start(system_prompt_ids, speaker_latent)
for microphone_frame in microphone_frames:
    acoustic_embedding = nemo_perception_step(microphone_frame)
    result = session.step(acoustic_embedding)
    pcm_22050 = nemo_codec_decode(result.audio_codes)
```

VoiceChat architectures automatically disable CUDA graphs and overlap
scheduling; EarTTS also disables chunked prefill. Keep at least two running
request slots: a streaming session retains one slot for its KV state while the
next frame is admitted. Each post-prefill turn uses an empty `input_ids` append
so SGLang forwards exactly the prior sampled placeholder token once.

## End-to-end server

The included adapter is split into two processes so that each component runs
in its supported pinned environment:

1. Run `nemo_audio_sidecar.py` in NVIDIA's VoiceChat NIM image with the unified
   checkpoint. It listens only on localhost by default and loads the causal
   perception encoder and audio codec.
2. Run `online_server.py` in the SGLang image with the converted Duplex and
   EarTTS directories. It exposes `GET /health` and WebSocket
   `/v1/realtime` (default port 18080).

At startup the server runs two silent frames through an isolated disposable
audio/model session. This warms first-frame and feedback kernels before the
server becomes ready, then releases all temporary KV and sidecar state. The
health response reports the warm-up duration. Use `--warmup-frames` to increase
the coverage or `--skip-warmup` only for debugging startup behavior.

The WebSocket accepts `session.update`, `input_audio_buffer.append`,
`input_audio_buffer.commit`, and `session.close`. Each append contains exactly
1280 mono PCM16 samples at 16 kHz. Output arrives as
`response.output_audio.delta`, mono PCM16 at 22.05 kHz. This is the audio
streaming subset of NVIDIA's Realtime-style protocol; transcription, function
calling, and server-side resampling are not yet implemented.

On a laptop, install `websockets` and run a mono 16-bit WAV (16 kHz or 24 kHz):

```bash
python -m pip install websockets
python examples/voicechat/client.py \
  --url ws://<voicechat-host>:18080/v1/realtime \
  --input-wav question.wav \
  --output-wav response.wav
```

The client resamples 24 kHz WAV files to the model's 16 kHz input rate, sends
80 ms frames on a fixed clock while receiving output concurrently, appends two
seconds of silence for turn completion, and writes the streamed response to a
22.05 kHz WAV. It also reports connection-to-first-audio latency and client-side
output interval mean/p95. The server uses a bounded input queue, preserving the causal
order within each stage while pipelining adjacent frames across perception,
Duplex, EarTTS, and codec. Each output frame reports queue, stage, total, and
output-interval latency; the commit event reports mean and p95 values.

NVIDIA's reference configuration enables CUDA graphs for NeMo perception and
TF32 for float32 EarTTS. The sidecar and server use those settings by default.
`--disable-perception-cudagraph` and
`--eartts-attention-backend torch_native` provide controlled comparison points.
On a multi-GPU node, use `--duplex-base-gpu-id` and `--eartts-base-gpu-id` to
place the two SGLang engines separately from the NeMo sidecar. Realtime means
that steady-state output cadence keeps up with the 80 ms source, total latency
remains bounded, and queue latency does not grow. Total latency can exceed
80 ms because adjacent frames occupy different causal pipeline stages.

The adapter remains intentionally single-session. The sample client currently
uses WAV input/output; microphone capture/playback is not yet implemented.
