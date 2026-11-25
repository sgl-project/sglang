"""MoE-LoRA kernels: the legacy fused shrink/expand kernel, LoRA-aware block
alignment and virtual-expert path, and the engine's dispatch, activation, B,
finalize stages and CuTeDSL grouped GEMMs."""
