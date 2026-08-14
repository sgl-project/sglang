# NVIDIA NemotronLabs VoiceChat inference with SGLang

This directory contains direct offline inference and realtime deployment
instructions for [NVIDIA NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).
Both paths run the Duplex and EarTTS autoregressive stages with SGLang on one
80 GB NVIDIA GPU.

The runtime follows NVIDIA's online architecture while replacing the two
autoregressive vLLM stages with persistent SGLang streaming sessions:

```text
microphone/WAV
      |
      v
NeMo perception -> SGLang Duplex -> SGLang EarTTS -> NeMo codec
      ^                  |                 |
      |                  + text/function state  + codec state
      +---------------------- one 80 ms frame -------------------+
```

NeMo remains responsible for the causal perception encoder and audio codec.
SGLang serves Duplex and EarTTS and retains their KV state across audio frames.
The adapter is currently single-session and supports audio streaming only.
Transcription events, function calling, Opus, and server-side resampling are
not implemented.

## Contents

| Document | Description |
| --- | --- |
| [Prerequisites](prerequisites.md) | GPU, software, checkpoint, image, and networking requirements |
| [Generate SGLang stages](generate-model-repo.md) | Download and convert the unified Hugging Face checkpoint |
| [Offline WAV inference](offline-inference.md) | Run a prerecorded WAV through direct SGLang engine sessions |
| [Deploy and run](deploy.md) | Start both services and use microphone or WAV clients |
| [API reference](api-reference.md) | HTTP/WebSocket endpoints, events, audio formats, errors, and limits |

Start with [Prerequisites](prerequisites.md). After converting the checkpoint
into `duplex/` and `eartts/`, choose [Offline WAV inference](offline-inference.md)
for a bounded file or [Deploy and run](deploy.md) for a realtime WebSocket.

## Programmatic integration

Applications that already provide NeMo perception and codec steps can use
`SGLangVoiceChatSession` directly:

```python
from examples.voicechat.online_session import SGLangVoiceChatSession

session = SGLangVoiceChatSession(duplex_engine, eartts_engine)
session.start(system_prompt_ids, speaker_latent, duplex_config.pad_token_id)
for microphone_frame in microphone_frames:
    acoustic_embedding = nemo_perception_step(microphone_frame)
    result = session.step(acoustic_embedding)
    pcm_22050 = nemo_codec_decode(result.audio_codes)
```

See [Deploy and run](deploy.md) for the complete two-container deployment.
