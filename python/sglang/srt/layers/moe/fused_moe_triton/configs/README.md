This directory contains tuned configurations for different settings of the fused_moe kernel.
For different settings of
- E (number of experts)
- N (intermediate size)
- device_name (torch.cuda.get_device_name())
- dtype: The data type used by the fused MoE kernel for computation. Supported types include fp8_w8a8, int8_w8a8, int8_w8a16, int4_w4a16, etc. This determines the precision and quantization scheme for both weights and activations.
- block_shape: The block quantization shape introduced starting from DeepSeek V3/R1 models. This parameter defines the granularity for block-wise quantization, typically specified as `[block_n, block_k]` where `block_n` and `block_k` represent the block dimensions. For example, DeepSeek V3 commonly uses `[128, 128]` block shapes for efficient block-wise FP8 quantization.

the JSON file contains a mapping from M (batch size) to the chosen configuration.

The example configurations provided are for the Mixtral model for TP2 on H100
and TP4 on A100. Mixtral has intermediate size N = 14336, i.e. for TP2 we have
N = 7168 and for TP4 we have N = 3584.

See `benchmark/kernels/fused_moe_triton/README.md` on how to generate these config files.
