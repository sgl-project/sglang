# Pi0.5 ModelOpt FP8 Linear quantization

The tool calibrates the native Pi0.5 fused projections with NVIDIA ModelOpt and
exports a checkpoint consumed by SGLang's native FP8 Linear backend. It does not
export ONNX or require TensorRT.

The default Linear coverage matches the pure-FP8 path and default performance
options in https://jetson-ai-lab.com/tutorials/openpi_on_thor/: PaliGemma and
action-expert QKV, output, gate/up and down projections, SigLIP encoder QKV/output and MLP fc1/fc2, the multimodal projector,
and action input/output heads. Each fused projection has one E4M3 weight scale
and one static input scale. Conv2d, embeddings, norms/AdaRMS dense and time MLP
remain unquantized, matching the exclusions in
https://jetson-ai-lab.com/tutorials/openpi_on_thor/. There is no executed
language-model output head in this policy.

FP8 GEMMs compute with BF16 activations and biases. Action heads retain FP32
outputs for the denoising integration, but their weights and inputs are now FP8;
this does not retain their former FP32 GEMM precision. Tensor/tuple interfaces
and feature dimensions are preserved for native SigLIP and action callers.
Attention and KV cache precision remain unchanged. This covers the Linear
quantization described at https://jetson-ai-lab.com/tutorials/openpi_on_thor/,
not its optional attention MatMul or NVFP4 quantization.
The first version requires a single SM89+ NVIDIA CUDA GPU and resident BF16
inference; TP/SP and component offload are rejected. CUDA graphs remain available
after FP8 weight loading is complete. Previously exported two-component
checkpoints remain supported without enabling additional components.

## Explicit synthetic calibration

For the initial pipeline check, use the dummy-input recipe from
https://jetson-ai-lab.com/tutorials/openpi_on_thor/: standard-normal images and
action noise, uniformly random full-vocabulary token IDs, and all-true image/token masks. This intentionally differs
from normalized real observations. The option is explicit; dataset failures never
trigger an automatic fallback.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.multimodal_gen.tools.quantize_pi05_modelopt_fp8 \
  --model-path /path/to/pi05_bf16 \
  --dummy-calibration \
  --seed 123 \
  --output-dir /path/to/pi05_fp8_dummy
```

The default matches the fallback at
https://jetson-ai-lab.com/tutorials/openpi_on_thor/: one calibration observation
running all denoising steps. Four separate synthetic observations with seed
`seed + 1` validate BF16, fake-quant and the reloaded native checkpoint. Adjust with
`--dummy-num-samples` and `--dummy-validation-samples`; `--validation-data` overrides
the synthetic validation set. Metadata labels synthetic calibration and validation.
Use this checkpoint to check execution, not robot policy quality.

## Prepare calibration data

Install `nvidia-modelopt`. Prepare representative **real**, preprocessed
observations using the checkpoint's tokenizer, image transforms and robot
normalization. In Pi0.5, tokens include the task and discretized state; raw task
text tokens alone do not reproduce that conditioning. Use the same preprocessing
as the existing BF16 serving path. The tool deliberately does not interpret raw
robot datasets or silently substitute synthetic observations.

Save a list of tensor dictionaries with `torch.save`. Given a preprocessed
`VLAObservationBatch` named `observation`, one sample looks like:

```python
sample = {
    "images": {key: value.cpu() for key, value in observation.images.items()},
    "image_masks": {key: value.cpu() for key, value in observation.image_masks.items()},
    "tokens": observation.tokens.cpu(),
    "token_masks": observation.token_masks.cpu(),
    "noise": observation.noise.cpu(),
}
torch.save(samples, "calibration.pt")
```

Every sample is batch size 1. Images must be `[1, 3, H, W]` floats in `[-1, 1]`
with keys matching the checkpoint cameras; masks are `[1]` bool. Tokens are
`[1, L]` int64, token masks are bool. Supply fixed FP32 noise of shape
`[1, action_horizon, action_dim]` (including the internal padded action dimensions).
If the preprocessor does not provide noise, generate and store it using a seeded
`torch.Generator`. State is already represented in the effective tokens.

Start with 128–512 varied observations covering scenes, instructions and robot
states, and keep a separate held-out validation set. All configured denoising
steps run during calibration. Missing/invalid observations and failed forward
passes stop export; there is no dummy-input fallback.

## Quantize and validate

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.multimodal_gen.tools.quantize_pi05_modelopt_fp8 \
  --model-path /path/to/pi05_bf16 \
  --calibration-data calibration.pt \
  --validation-data validation.pt \
  --output-dir /path/to/pi05_fp8
```

The default components are `paligemma action_expert vision projector action_heads`.
Use `--components paligemma action_expert` to reproduce the earlier restricted
coverage, or select any subset to isolate sensitivity.
The output directory must not exist. The default denoising step count comes from
the checkpoint; `--num-steps` overrides it and is recorded in the exported config.

The output contains `model.safetensors`, `config.json`, copied tokenizer files
and OpenPI normalization assets when present. Calibration metadata records the
source model, data path, sample count, seed and denoising schedule.

With `--validation-data`, the tool reloads the native FP8 checkpoint and writes
`validation_report.json`, comparing BF16 vs native FP8 and ModelOpt fake-quant vs
native FP8 using identical noise. Metrics cover normalized internal action tensors,
including padded dimensions; task-specific unnormalization, valid action dimensions,
horizon slicing and closed-loop policy quality require separate evaluation.
No cosine or action-error threshold alone proves robot task success.

## Serve

```bash
sglang serve /path/to/pi05_fp8 \
  --pipeline-class-name Pi05Pipeline
```

The pipeline override ensures that arbitrary local output names dispatch to Pi0.5.
The model loader reads the checkpoint's `quantization_config` automatically; do not
use the LLM `--modelopt-quant` text-calibration loader for this checkpoint.
Benchmark against resident BF16 with identical graph settings, prompts, batch size,
action horizon and denoising steps. Measure prefix, denoise, end-to-end latency and
peak VRAM separately. Short action-expert GEMMs may not benefit from FP8.
