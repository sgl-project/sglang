# SGLang Omni WebUI

This is a small local WebUI for testing interleaved omni models through
SGLang's `/v1/omni/generate` API. It is intentionally a thin browser UI plus
HTTP proxy: model serving still happens in `sglang serve --model-type omni`.

The first target model is SenseNova U1. The UI supports a free-form chat loop
for repeated refinements, and also keeps a two-turn preset for quickly checking
that a later turn can edit a previously generated image.

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
`--request-model` for the API payload. It defaults to `1024x1024`, matching
U1's image sampling defaults. Custom width and height values should be multiples
of 32.

## Multi-Turn Flow

Use the `Message` composer for normal multi-turn refinement. Keep `Keep omni
session across turns` enabled, then click `Send` for each user turn. `New chat`
closes the current server session and clears the thread.

For the preset flow:

1. Click `Preset: first image`.
2. Click `Send`.
3. Keep the returned session alive.
4. Click `Preset: edit previous`.
5. Click `Send` again.

Turn 2 reuses the session returned by turn 1, so the model can refer to the
previous generated image and edit it.

## API Proxy

The UI server proxies only the endpoints it needs:

| Local UI endpoint | Omni server endpoint |
| --- | --- |
| `GET /api/config` | local UI config |
| `GET /api/health` | `GET /health` |
| `POST /api/omni/generate` | `POST /v1/omni/generate` |
| `POST /api/omni/close` | `POST /v1/omni/sessions/{session_id}/close` |

This WebUI is for local development and demos. Do not expose it directly to an
untrusted network.
