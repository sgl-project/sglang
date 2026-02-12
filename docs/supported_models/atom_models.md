# ATOM Models

## Introduction

ATOM is a high-performance model implementation backend specifically optimized for AMD Instinct GPUs. This doc guides users to run ATOM-accelerated models in SGLang, delivering optimal performance on AMD Instinct GPU devices (e.g., MI300 series).

## Requirements

Users need to install ATOM

## Supported Models

ATOM currently provides optimized implementations for the following model architectures on AMD Instinct GPUs:

- **Llama Models**: Llama 3.x
- **Qwen Models**: Qwen serial models
- **DeepSeek**: deepseek-r1, deepseek-v3.2

## Install ATOM

```bash
# Clone the ATOM repository
git clone https://github.com/ROCm/atom.git
cd atom
pip install -e .
```

## Run Model

Launch server with multiple AMD GPUs:

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8 \
    --model-impl atom
```

## Performance Benefits

ATOM provides significant performance improvements on AMD Instinct GPUs:

- **Optimized Kernels**: Custom-tuned kernels for AMD GPUs
- **Customized Fusions**: ATOM model provides customized fusion patterns

## Support

For ATOM-specific issues:
- Refer to the [ATOM GitHub Repository](https://github.com/ROCm/atom)
- ROCm documentation: [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- SGLang issues: [SGLang GitHub Issues](https://github.com/sgl-project/sglang/issues)

For general SGLang support, please refer to the main [SGLang documentation](https://sgl-project.github.io/docs/).
