# SGLang Omni WebUI

This is a small local WebUI for testing interleaved omni models through
SGLang's `/v1/omni/generate` API. It is intentionally a thin browser UI plus
HTTP proxy: model serving still happens in `sglang serve --model-type omni`.

The first target model is SenseNova U1. The UI supports fresh text-to-image
requests, explicit image editing, VLM checks, and advanced interleave sessions.
Requests use `stream: true`, so text appears incrementally and image generation
shows a placeholder until the final image event arrives.

## Start An Omni Server

Install SGLang with diffusion dependencies when image generation is needed:

```bash
pip install -e "python[all,diffusion]"
```

Start SenseNova U1 with the omni model type:

```bash
sglang serve \
  --model-type omni \
  --model-path sensenova/SenseNova-U1-8B-MoT \
  --host 0.0.0.0 \
  --port 30000
```

If the server runs on a remote GPU machine, forward the API port to your local
machine:

```bash
ssh -L 30000:127.0.0.1:30000 <remote-host>
```

## Start The WebUI

Run the WebUI locally:

```bash
python -m sglang.omni.webui.server \
  --host 127.0.0.1 \
  --port 7860 \
  --api-base http://127.0.0.1:30000 \
  --served-model sensenova/SenseNova-U1-8B-MoT
```

Open:

```text
http://127.0.0.1:7860
```

The UI displays the served model from `--served-model`, while requests use
`--request-model` for the API payload. It defaults to U1's quality image
configuration: `1024x1024`, 50 steps, text CFG `4.0`, and image CFG `1.0`.
Use the Draft or Medium presets only when you explicitly want a faster
interactive check. Custom width and height values should be multiples of 32.

## Recommended Flow

For unrelated new images, use `t2i` and start a fresh request. This matches the
official U1 usage pattern: a new topic should not inherit the previous SRT KV
session.

For image editing, use `edit` and provide an input image. The WebUI can reuse
the last generated image when `Use previous image as input` is enabled, or you
can upload a file through `Optional input image`.

1. Click `Preset: first image`.
2. Click `Send`.
3. Wait for the image to appear.
4. Click `Preset: edit previous`.
5. Click `Send` again.

Turn 2 sends the first generated image as an explicit input image. It does not
reuse the previous KV session.

`Keep interleave session across turns` is an advanced option for interleaved
text-image protocol experiments. It is useful when the model is expected to keep
one protocol-level conversation, but it should not be used for independent image
topics.

## API Proxy

The UI server proxies only the endpoints it needs:

| Local UI endpoint | Omni server endpoint |
| --- | --- |
| `GET /api/config` | local UI config |
| `GET /api/health` | `GET /health` |
| `POST /api/omni/generate` | `POST /v1/omni/generate` |
| `POST /api/omni/close` | `POST /v1/omni/sessions/{session_id}/close` |

`POST /api/omni/generate` passes through `text/event-stream` responses without
buffering when the browser sends `stream: true`.

This WebUI is for local development and demos. Do not expose it directly to an
untrusted network.
