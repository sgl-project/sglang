"""sgl_flashinfer_trtllm MoE LoRA support (package).

Holds the LoRA-specific bits of the ``sgl_flashinfer_trtllm`` MoE backend so
that the existing files ``layers/moe/moe_runner/flashinfer_trtllm.py`` and
``lora/layers.py`` need only tiny injection points (re-exports or
``if backend == sgl_flashinfer_trtllm: ...`` branches) for the new LoRA path.

Submodules:
  * :mod:`.lora_dispatch` — the ``fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora``
    dispatch function moved out of ``flashinfer_trtllm.py``.
  * :mod:`.lora_layer` — the ``FusedMoEWithLoRA`` init/dispatch helpers moved
    out of ``lora/layers.py``.
  * :mod:`.specialized_expand` — the rank-specialized LoRA-B expand kernel
    moved out of ``lora/triton_ops/virtual_experts.py``.
"""
