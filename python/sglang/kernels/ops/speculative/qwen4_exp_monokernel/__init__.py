"""Qwen4-Exp (Qwen3.8-Flash-Next) FlyDSL monokernel: the whole decoder stack of a speculative target verify
(R <= 8 requests x L <= 4 draft tokens) or MTP draft step in persistent launches, reading sglang's AITER-layout
weights in place. gfx950 only; the runtime integration is ``sglang.srt.models.qwen4_exp_monokernel``.
"""
