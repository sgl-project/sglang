"""Kernels for the MoE LoRA path, staged for the move to sglang.kernels.

The Triton modules here take tensors and a launch config; ``cutedsl`` is the
in-tree CuTeDSL grouped GEMM with its own compile pipeline. Nothing here
imports the plan, the runner, or a provider, so this package stays free of
pydantic and of the execution-plan types.
"""
