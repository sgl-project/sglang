# SGLang Diffusion Realtime Video API (WebSocket)

The realtime diffusion endpoint provides chunked video generation over WebSocket.
It is designed for interactive generation loops where the client can continuously
update prompts or stream input frames while receiving generated video chunks.

## Endpoint

- **WebSocket URL**: `ws://<host>:<port>/v1/realtime_video/generate`
- **Serialization**: MessagePack (`msgpack`)

The protocol is binary WebSocket + MessagePack, not JSON over HTTP.

## Start the Server

Use a realtime-capable model (for example, Krea realtime video):

```bash
sglang serve \
    --model-path  /home/admin/krea-realtime-video-diffusers \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload false \
    --image-encoder-cpu-offload false \
    --vae-cpu-offload false \
    --pin-cpu-memory false \
    --dit-layerwise-offload false \
    --num-gpus 1 \
    --port 30000
```

## Message Protocol

### 1) Initial generation request (send once after connect)

Send a MessagePack object that matches `RealtimeVideoGenerationsRequest`.

Common fields:

- `prompt` (`str`, required): initial prompt
- `mode` (`"t2v" | "v2v"`, optional): force mode selection
  - `"v2v"`: force v2v mode
  - `"t2v"`: force t2v mode (video actions are rejected)
- `first_frame` (`bytes | str`, optional): first conditioning frame
- `size` (`str`, optional): e.g. `"832x480"`
- `seed` (`int`, optional)
- `fps` (`int`, optional)
- `num_inference_steps` (`int`, optional)
- `guidance_scale` (`float`, optional)
- `guidance_scale_2` (`float`, optional)
- `negative_prompt` (`str`, optional)
- `enable_teacache` (`bool`, optional)
- `output_path` (`str`, optional)
- `output_quality` (`str`, optional)
- `output_compression` (`int`, optional)

### 2) Realtime action messages (send anytime during streaming)

Send a MessagePack object that matches `RealtimeAction`.

#### Prompt action

```json
{
  "type": "prompt",
  "action_content": "new prompt text"
}
```

#### Video action

Use either `video_frame` (single frame) or `video_frames` (list of frames):

```json
{
  "type": "video",
  "video_frames": ["<bytes>", "<bytes>", "..."]
}
```

Each frame should be encoded image bytes (for example JPEG/PNG bytes).

### 3) Server response messages

The server sends MessagePack objects:

- `{"type": "frame", "content": <bytes>}`: one generated chunk (mp4 bytes)
- `{"type": "error", "content": "<message>"}`: request/action/loop error

## Minimal Python Client

```python
import asyncio
import msgpack
import websockets


async def main():
    ws_url = "ws://127.0.0.1:30000/v1/realtime_video/generate"
    async with websockets.connect(ws_url) as ws:
        # 1) Send initial generation request
        init_req = {
            "prompt": "A person talking to camera",
            "size": "832x480",
            "seed": 1024,
            "num_inference_steps": 4,
            "fps": 12,
        }
        await ws.send(msgpack.packb(init_req))

        # 2) Receive chunks in background
        async def recv_loop():
            async for message in ws:
                data = msgpack.unpackb(message)
                if data.get("type") == "frame":
                    chunk_bytes = data["content"]
                    print("got frame chunk:", len(chunk_bytes), "bytes")
                else:
                    print("server message:", data)

        recv_task = asyncio.create_task(recv_loop())

        # 3) Send prompt updates during streaming
        await asyncio.sleep(5)
        await ws.send(
            msgpack.packb(
                {"type": "prompt", "action_content": "he smiles gently"}
            )
        )

        await asyncio.sleep(10)
        await ws.send(
            msgpack.packb(
                {"type": "prompt", "action_content": "he becomes very serious"}
            )
        )

        await asyncio.sleep(10)
        recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming Actions (Prompt/Video)

You can stream actions continuously while generation is running:

- `type="prompt"`: update text prompt during streaming (`action_content`)
- `type="video"`: stream live input frames
  - `video_frame`: one encoded frame bytes
  - `video_frames`: multiple encoded frame bytes in one action
  - requires v2v mode (`mode="v2v"`, or auto mode with `first_frame` set)

The server consumes prompt/video actions concurrently with generation. If no
new action is provided, generation continues with the latest effective state.

## Lifecycle Notes

- Keep one WebSocket connection alive for one interactive session.
- The server runs generation and action handling concurrently.
- Closing the WebSocket terminates the session and releases server-side realtime
  session state.
