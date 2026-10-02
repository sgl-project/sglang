---
title: "Prompt enhancement"
description: "Connect an independently deployed SGLang LLM or VLM to diffusion image and video APIs."
---

Run an SRT server for prompt expansion and a diffusion server for generation.
The diffusion HTTP frontend calls SRT's `/v1/chat/completions` once before
constructing the generation request. The enhancer does not replace the
diffusion model's text encoder or run inside its GPU workers.

```text
Application -> diffusion HTTP server -> SRT enhancer server
                                    <- rewritten prompt
            <- image/video          <- diffusion pipeline
```

Prompt enhancement is opt-in per request. It changes conditioning and can
change intent or quality; it is not a lossless acceleration feature. Evaluate
the chosen enhancer and template on your workload before enabling it.

## Deploy two services

This example uses Linux with NVIDIA CUDA and two GPUs, one for each server.
Install [SGLang](/docs/get-started/install) for SRT and
[SGLang Diffusion](/docs/sglang-diffusion/installation) for diffusion. Separate
environments and hosts are supported; the diffusion host needs HTTP access to SRT.

Start a text-only enhancer:

```bash
CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --port 31000
```

Create `enhancer.json` on the diffusion host:

```json
{
  "base_url": "http://127.0.0.1:31000/v1",
  "model": "Qwen/Qwen3-4B-Instruct-2507"
}
```

Start diffusion in another terminal:

```bash
CUDA_VISIBLE_DEVICES=1 sglang serve \
  --model-path Tongyi-MAI/Z-Image-Turbo \
  --prompt-enhancer-config enhancer.json \
  --port 30000
```

Keep the SRT port separate from diffusion's HTTP and internal ports; diffusion
reserves `--port + 1` for its broker. For separate hosts, bind SRT to its service
interface with `--host` and update
`base_url`. Restrict network access and configure SRT authentication. Do not
expose an unauthenticated enhancer publicly. A shared enhancer can serve
multiple diffusion servers; size its concurrency and memory independently.

Send a generation request:

```bash
curl --fail-with-body http://127.0.0.1:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A red teapot on a windowsill, morning light",
    "enhance_prompt": true,
    "size": "512x512",
    "seed": 42,
    "response_format": "b64_json"
  }'
```

Omit `enhance_prompt` or set it to `false` to use the original prompt without
calling SRT. With the OpenAI Python SDK, pass `extra_body={"enhance_prompt": True}`
to `images.generate()`.

The same switch works with `/v1/images/edits` and `/v1/videos`, including
multipart requests (`-F 'enhance_prompt=true'`). Image responses expose the
rewritten text as `data[].revised_prompt`; video responses use `revised_prompt`.
Video creation waits for enhancement before returning the queued job. All `n`
outputs share one rewritten prompt.

Include enhancement in client-observed end-to-end latency. The response's
`inference_time_s` measures diffusion execution, not the preceding SRT call.

## Configure the enhancer

The configuration file is read at server startup. Its supported fields are:

| Field | Meaning |
| --- | --- |
| `base_url` | SRT OpenAI API base URL, including `/v1` |
| `model` | SRT model ID or its configured served model name |
| `system_prompt` | Optional replacement for the built-in, task-aware rewriting instruction |
| `include_images` | Forward reference images to a VLM; defaults to `false` |
| `timeout` | HTTP operation timeout in seconds; defaults to `60` |
| `api_key_env` | Name of an environment variable on the diffusion server containing the SRT API key |
| `generation_kwargs` | SRT chat-completion options, such as `max_tokens`, `temperature`, `chat_template_kwargs`, or `response_format` |

Default generation options are `max_tokens=512` and `temperature=0`.
`generation_kwargs` cannot replace the model, messages, single-output or
non-streaming settings, or enable tools. Choose a non-thinking model or configure
its supported chat template to suppress reasoning; only message `content` is
used as the diffusion prompt.

The enhancer receives a JSON text message containing `task` (`image`,
`image_edit`, or `video`) and `prompt`. It returns the replacement prompt as
text. Seed, negative prompt, references and diffusion sampling options are not
rewritten. The diffusion frontend reuses HTTP connections and makes no enhancer
calls during synthetic warmup.

Timeouts return HTTP 504. Upstream failures, empty responses and truncated
completions return HTTP 502 before diffusion is queued. There is no automatic
retry or fallback to the original prompt. An unconfigured server rejects
`enhance_prompt=true` with HTTP 400.

## Choose a model combination

There is no enhancer or diffusion model allowlist in this integration. Protocol
compatibility does not establish quality: use a template appropriate for the
target checkpoint and verify prompt adherence.

| Diffusion input | Enhancer requirements |
| --- | --- |
| Natural-language image/video prompts, such as Z-Image, FLUX, Qwen-Image, Wan, LTX or MiniMax-H3 | An SRT-served instruction model with chat completions; the default template is a starting point |
| Image editing or image-to-video | Text-only instruction rewriting, or an SRT VLM with `include_images=true` for image-aware rewriting |
| Structured captions, such as Ideogram 4 or LingBot-Video-MoE | A model-specific `system_prompt` and preferably schema-constrained `response_format`; the default prose template is unsuitable |
| ERNIE-Image or LongCat with built-in rewriting | Disable the built-in rewriter for the request when using this generic enhancer, to avoid rewriting twice |

For a VLM, deploy an image-capable SRT model and set `include_images=true` in
its configuration. Uploaded images are sent as data URLs, so the SRT server
does not need access to the diffusion host's filesystem. HTTP image URLs must
be reachable from SRT. Text-only models should leave this option disabled.
Audio and video references are not forwarded to the enhancer; it cannot inspect
their content. Image edits forward the supplied images; video creation forwards
the primary image reference, not model-specific reference bundles.

For [Ideogram 4](/cookbook/diffusion/Ideogram/Ideogram4), instruct the enhancer
to produce the checkpoint's JSON caption schema, including
`high_level_description` and `compositional_deconstruction`. Set the matching
JSON schema through `generation_kwargs.response_format`. JSON text is passed
through unchanged, not converted to prose. Valid JSON alone does not guarantee
the correct caption schema or equivalent output to a hosted service's expander.

This integration applies to the three HTTP image/video endpoints above,
including their normal parallelism, caching and disaggregated diffusion paths.
It does not add enhancement to offline `sglang generate`, realtime streaming,
mesh or action endpoints. For those workflows, call SRT in your application
first and pass its returned text as the diffusion prompt. For reproducibility,
store the rewritten prompt and replay it with `enhance_prompt=false`; a fixed
diffusion seed does not fix the enhancer's output.

## ERNIE-Image's checkpoint-provided enhancer

ERNIE-Image also has a separate, existing PE integration. By default, diffusion
loads its native Ministral3 enhancer in-process:

```bash
sglang serve --model-path baidu/ERNIE-Image --port 30000
```

Use `use_pe=false` to skip it for a request. For a memory-constrained deployment,
`--layerwise-offload-components pe` streams the native PE decoder from CPU.

To serve the checkpoint-provided PE through SRT, use a local checkpoint's PE
directory with its tokenizer and retain the ERNIE-specific protocol:

```bash
CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path /models/ERNIE-Image/pe --port 31000

CUDA_VISIBLE_DEVICES=1 sglang serve \
  --model-path /models/ERNIE-Image \
  --pe-server-url http://127.0.0.1:31000 --port 30000
```

`--pe-server-url` uses the checkpoint's PE tokenizer/template and SRT's
`/generate`; it is not the generic chat-completion integration. Keep using it
when you need ERNIE's trained PE behavior. If you instead use
`--prompt-enhancer-config`, send both `enhance_prompt=true` and `use_pe=false`.
For LongCat, use `enable_prompt_rewrite=false` with the generic enhancer.
These request flags skip native rewriting, not loading its model components.

For Ascend setup, see [models with AR stages](/docs/sglang-diffusion/models_with_ar#ascend-npu-env).
