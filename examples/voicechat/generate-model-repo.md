# Generate SGLang VoiceChat stages

The published checkpoint is a unified NeMo checkpoint. SGLang serves its
Duplex and EarTTS autoregressive stages from separate Hugging Face-style model
directories. The NeMo sidecar continues to read the original checkpoint for
perception and codec weights.

## Directory layout

Choose a working directory outside the SGLang checkout:

```text
voicechat-data/
├── checkpoint/       # unified Hugging Face checkpoint
└── converted/
    ├── duplex/       # SGLang Duplex stage
    └── eartts/       # SGLang EarTTS stage and speaker latents
```

Run the commands from the SGLang checkout. Choose an absolute artifact path on
a local POSIX filesystem with at least 100 GB free:

```bash
export VOICECHAT_DATA=/absolute/path/to/voicechat-data
mkdir -p "$VOICECHAT_DATA"
```

## Step 1: Download the checkpoint

Accept the model terms on Hugging Face if required, authenticate through the
Hugging Face CLI, and download the repository. Do not put access tokens in the
command line.

```bash
mkdir -p "$VOICECHAT_DATA/checkpoint"
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --local-dir "$VOICECHAT_DATA/checkpoint"
```

Confirm that both primary files exist:

```bash
test -f "$VOICECHAT_DATA/checkpoint/config.json"
test -f "$VOICECHAT_DATA/checkpoint/model.safetensors"
```

## Step 2: Convert Duplex

Duplex conversion is bounded by the largest source tensor. It writes many
safetensors shards so the 40+ GB unified checkpoint does not need to be loaded
into host memory at once.

```bash
mkdir -p "$VOICECHAT_DATA/converted/duplex"
docker run --rm \
  --shm-size=8g \
  -v "$VOICECHAT_DATA/checkpoint:/checkpoint:ro" \
  -v "$VOICECHAT_DATA/converted:/converted" \
  -w /sgl-workspace/sglang \
  --entrypoint python \
  sglang-voicechat \
  examples/voicechat/convert_duplex_stage.py \
    --checkpoint /checkpoint \
    --config /checkpoint/config.json \
    --output /converted/duplex
```

The converter reads the base-model identifier from the unified configuration.
Pass `--base-model <model-or-path>` if converting a custom checkpoint without
that field. Mount a pre-populated Hugging Face cache if the conversion host
does not have outbound network access.

## Step 3: Convert EarTTS

EarTTS conversion instantiates NVIDIA's trained character-aware subword
encoder and precomputes its full-vocabulary lookup table. Run it in NVIDIA's
VoiceChat environment with a GPU. Select the allocated GPU by UUID rather than
assuming a physical index.

```bash
mkdir -p "$VOICECHAT_DATA/converted/eartts"
docker run --rm \
  --gpus 'device=<gpu-uuid>' \
  --shm-size=8g \
  -v "$PWD:/workspace/sglang:ro" \
  -v "$VOICECHAT_DATA/checkpoint:/checkpoint:ro" \
  -v "$VOICECHAT_DATA/converted:/converted" \
  -w /workspace/sglang \
  --entrypoint python \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest \
  examples/voicechat/convert_eartts_stage.py \
    --config /checkpoint/config.json \
    --model /checkpoint/model.safetensors \
    --output /converted/eartts
```

Pass `--base-model <model-or-path>` for a custom checkpoint whose tokenizer or
base-model reference differs from the published configuration. Reduce
`--precompute-batch-size` if the conversion runs out of temporary GPU memory.

## Step 4: Verify the output

```bash
test -f "$VOICECHAT_DATA/converted/duplex/config.json"
test -f "$VOICECHAT_DATA/converted/duplex/model.safetensors.index.json"
test -f "$VOICECHAT_DATA/converted/eartts/config.json"
test -f "$VOICECHAT_DATA/converted/eartts/model.safetensors"
test -n "$(find "$VOICECHAT_DATA/converted/eartts/speaker_latents" \
  -maxdepth 1 -name '*.pt' -print -quit)"
```

Conversion output may be owned by the container user. Correct ownership or
permissions only for the explicit `voicechat-data/converted` directory; never
apply recursive permission changes to the repository or a broad parent path.

Next: [Deploy and run](deploy.md)
