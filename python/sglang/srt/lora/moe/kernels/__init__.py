"""Triton kernels for the MoE LoRA path.

Nothing here imports the plan, the runner, or a provider. These modules take
tensors and a launch config, so they stay free of pydantic and of the
execution-plan types.
"""
