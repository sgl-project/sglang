# GLM-5.3-Flash Router Image

Build recipe for an `sgl-router` image with the GLM-5.3-Flash tokenizer and
chat template baked in, so the router starts instantly without an outbound
HuggingFace call.

## Files

- `Dockerfile` — multi-stage override: takes the upstream
  `sgl-router:combine-router-admission-http2-loadaware` base and drops the
  tokenizer assets in `/tokenizer/`.
- `tokenizer_config.json` — upstream `tokenizer_config.json` from
  [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash),
  with the `chat_template` field **inlined** from `chat_template.jinja`. The
  router only reads the template from this file (see
  `experimental/sgl-router/src/tokenizer/chat_template.rs::extract_chat_template`),
  not from a sibling `.jinja`.
- `chat_template.jinja` — kept for reference and diff inspection. Not loaded
  by the router directly.

## Why the inline

GLM-5.3-Flash ships its chat template as a separate `chat_template.jinja`
file. The router's `load_tokenizer_config` (`tokenizer/adapter.rs`) only
loads `tokenizer_config.json` as a sibling of `tokenizer.json`, then looks
for a `chat_template` field inside it. Without the merge, the router falls
back to its built-in encoder detection and never applies the GLM template.

## Build

```bash
# Step 1: build the base router image from the parent branch.
docker build -f ../sgl-router.Dockerfile \
  -t sgl-router:combine-router-admission-http2-loadaware \
  ../..

# Step 2: layer the tokenizer on top. tokenizer.json is ~20 MB and is NOT
# committed to this repo — fetch it from HF before this step.
curl -sL -o tokenizer.json \
  https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/tokenizer.json

docker build -f Dockerfile -t sgl-router:glm53-flash-combine .
```

## Run

```bash
docker run --rm -p 30000:30000 sgl-router:glm53-flash-combine \
  --model-id glm-5.3-flash \
  --tokenizer-path /tokenizer/tokenizer.json \
  --worker-urls http://worker-1:8000 http://worker-2:8000
```

The router resolves `tokenizer_config.json` as a sibling of
`--tokenizer-path`, finds the inlined `chat_template`, and uses it to render
prompts before tokenization. No additional flag is needed.

## Image size

~101 MB total (78 MB base router + 20 MB tokenizer.json + ~30 KB configs).
