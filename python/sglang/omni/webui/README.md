# SGLang Omni WebUI

This is a small local WebUI for testing interleaved omni models through
SGLang's `/v1/omni/generate` API. It is intentionally a thin browser UI plus
HTTP proxy: model serving still happens in `sglang serve --model-type omni`.

The first target model is SenseNova U1. The UI keeps an omni session across
turns and renders returned text/image segments in their original order, so it
is useful for checking multi-turn interleaved generation and image editing.

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
  --api-base http://127.0.0.1:30000
```

Open:

```text
http://127.0.0.1:7860
```

## Multi-Turn Flow

1. Click `Load turn 1`.
2. Click `Generate`.
3. Keep the returned session alive.
4. Click `Load turn 2`.
5. Click `Generate` again.

Turn 2 reuses the session returned by turn 1, so the model can refer to the
previous generated image and edit it.

## API Proxy

The UI server proxies only the endpoints it needs:

| Local UI endpoint | Omni server endpoint |
| --- | --- |
| `GET /api/health` | `GET /health` |
| `POST /api/omni/generate` | `POST /v1/omni/generate` |
| `POST /api/omni/close` | `POST /v1/omni/sessions/{session_id}/close` |

This WebUI is for local development and demos. Do not expose it directly to an
untrusted network.
