# MinerU Diffusion External Plugin (Example)

This example shows how to package an external multimodal diffusion model adapter
for SGLang via `SGLANG_EXTERNAL_*` environment variables.

The plugin targets the architecture name:

- `MinerUDiffusionForConditionalGeneration`

and demonstrates:

- external model class registration via `EntryClass`,
- external multimodal processor registration,
- external architecture declaration for multimodal runtime,
- DLLM launch configuration for diffusion decoding.

## Directory layout

```text
examples/runtime/mineru_diffusion_external_plugin/
├── README.md
├── dllm_config.yaml
└── mineru_sglang_plugin/
    ├── __init__.py
    ├── configuration_mineru_diffusion.py
    ├── processing_mineru_diffusion.py
    ├── mm_processor_mineru.py
    └── modeling_mineru_diffusion.py
```

## How to use

Install this package in your runtime image/environment, then launch:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=mineru_sglang_plugin
export SGLANG_EXTERNAL_MM_MODEL_ARCH=MinerUDiffusionForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=mineru_sglang_plugin

python -m sglang.launch_server \
  --model-path /path/to/model \
  --enable-multimodal \
  --disable-fast-image-processor \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config /path/to/dllm_config.yaml
```

> Note: this example is provided as an external-plugin reference. It is not
> wired into SGLang's built-in model registry by default.
