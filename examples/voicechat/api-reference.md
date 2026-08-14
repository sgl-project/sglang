# API reference

The SGLang VoiceChat adapter exposes a realtime-style JSON WebSocket protocol.
Audio payloads are base64-encoded mono PCM16. This is an intentionally small
audio-streaming subset, not a complete OpenAI Realtime API implementation.

## HTTP endpoints

### `GET /`

Returns service discovery metadata:

```json
{
  "service": "sglang-nemotron-voicechat",
  "websocket": "/v1/realtime",
  "websocket_alias": "/realtime",
  "health": "/v1/realtime/health",
  "input_sample_rate": 16000,
  "output_sample_rate": 22050
}
```

### `GET /health`

### `GET /v1/realtime/health`

Both paths return the same readiness and runtime information:

```json
{
  "ready": true,
  "input_format": "pcm16",
  "input_sample_rate": 16000,
  "output_format": "pcm16",
  "output_sample_rate": 22050,
  "frame_samples": 1280,
  "single_session": true,
  "max_audio_queue_frames": 256,
  "context_length": 8192,
  "warmup": {
    "enabled": true,
    "frames": 2,
    "duration_ms": 61000.0
  },
  "audio_sidecar": {
    "ready": true
  }
}
```

The server does not accept WebSocket connections until application startup and
warm-up complete. Treat `ready: true` as the readiness condition.

## WebSocket endpoints

The two paths are aliases:

| Path | Description |
| --- | --- |
| `/v1/realtime` | Primary endpoint used by the bundled client |
| `/realtime` | Compatibility alias |

Only one connection can own the model sessions. A concurrent connection
receives an error event and closes with WebSocket code `1013`.

## Audio format

| Parameter | Value |
| --- | --- |
| Input | mono signed PCM16 little-endian |
| Input sample rate | 16 kHz |
| Input frame | exactly 1,280 samples / 2,560 bytes / 80 ms |
| Output | mono signed PCM16 little-endian |
| Output sample rate | 22,050 Hz |
| Typical output frame | 1,764 samples / 80 ms |

The server performs no resampling. The bundled client converts mono PCM16 WAV
files from their source rate to the required 16 kHz input. Microphone capture
opens the selected device at its advertised native rate and converts each frame
to 16 kHz. Playback is likewise converted to the output device's native rate.

## Session lifecycle

```text
Client                                      Server
  |---- connect ----------------------------->|
  |<--- session.created ----------------------|
  |---- session.update ---------------------->|
  |<--- session.updated ----------------------|
  |---- input_audio_buffer.append ----------->|  repeated
  |<--- response.output_audio.delta ----------|  repeated
  |---- input_audio_buffer.commit ----------->|
  |<--- input_audio_buffer.committed ---------|
  |---- session.close ------------------------>|
  |<--- WebSocket close ----------------------|
```

Input and output are concurrent. Output for earlier frames can arrive while the
client is still sending later microphone frames.

## Server events

### `session.created`

Sent immediately after the WebSocket is accepted:

```json
{
  "type": "session.created",
  "session": {
    "input_audio_format": "pcm16",
    "input_sample_rate": 16000,
    "output_audio_format": "pcm16",
    "output_sample_rate": 22050,
    "frame_samples": 1280
  }
}
```

### `session.updated`

Confirms the effective instructions, sample rates, and `max_input_frames` after
`session.update`.

### `response.output_audio.delta`

Returns one decoded audio frame and diagnostic model state:

```json
{
  "type": "response.output_audio.delta",
  "delta": "<base64 PCM16>",
  "sample_rate": 22050,
  "samples": 1764,
  "text_token": 123,
  "function_token": 456,
  "timing_ms": {
    "queue": 0.02,
    "perception": 8.0,
    "duplex": 6.0,
    "eartts": 41.0,
    "models": 47.0,
    "codec": 7.0,
    "total": 104.0,
    "output_interval": 80.0
  }
}
```

Token fields are implementation diagnostics, not transcript text. Timing values
are observational and depend on hardware and workload.

### `input_audio_buffer.committed`

Sent after all audio preceding the commit marker has crossed every pipeline
stage. It includes frame count plus mean and p95 timing for the queue and each
stage. If any non-padding text token occurred in the final 12 frames, the event
also reports that the client may have stopped supplying frames while the reply
was still active:

```json
{
  "type": "input_audio_buffer.committed",
  "truncation_warning": true,
  "warning": "The reply was still emitting in the final 12 frames; send more trailing silence to avoid truncation.",
  "timing_ms": {
    "frames": 100
  }
}
```

The bundled client prints this warning to stderr. Increase
`--trailing-silence` and repeat the request when it appears.

### `error`

```json
{
  "type": "error",
  "error": {
    "message": "<description>"
  }
}
```

Worker failures close with code `1011`. Invalid client events and payloads close
with code `1008`. A busy server closes with code `1013`.

## Client events

### `session.update`

Send before the first audio frame:

```json
{
  "type": "session.update",
  "session": {
    "instructions": "You are a helpful, concise voice assistant."
  }
}
```

Calling `session.update` after audio is enqueued is an error. Unknown session
fields are currently ignored. Tool definitions are not supported.

### `input_audio_buffer.append`

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64-encoded 2560-byte PCM16 frame>"
}
```

Each event must contain exactly one 80 ms input frame. The server uses a bounded
queue and rejects input once `max_audio_queue_frames` is exceeded. It also
rejects the frame that would exceed the prompt- and speaker-prefill-aware
`max_input_frames` reported by `session.updated`.

### `input_audio_buffer.commit`

```json
{"type": "input_audio_buffer.commit"}
```

The commit is an ordered drain marker. It does not stop the session and may be
used only when the input queue has capacity.

### `session.close`

```json
{"type": "session.close"}
```

The server drains the ordered stop marker, releases both SGLang KV sessions and
the NeMo audio session, then closes the WebSocket.

## Current limitations

- One active conversation per server process.
- PCM16 only; no Opus.
- Fixed 16 kHz server input and 22.05 kHz output.
- No server-side voice activity detection or turn detection.
- No user or assistant transcription events.
- No function calling or `conversation.item.create` support.
- No authentication or TLS in the example server.

Unsupported event types produce an error rather than being silently accepted.
