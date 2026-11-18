"""Inkling2 routed experts on FlashInfer's CuTe DSL swap-AB MoE.

Coverage: SM100 / SM103, NVFP4 (packed `uint8`) or MXFP8 (`float8_e4m3fn`) expert weights, SiTU,
at most `max_num_tokens` tokens per call. Activations are quantized per token (4over6 NVFP4) or
MXFP8 inside the call. The output is this rank's partial sum over its local experts; the caller
reduces it across EP ranks.

Weights are the model's tensors, used in place and shared with `Inkling2MegaMoE`
(`weight_layout="up_gate_block16"`):
- w13 rows in FlashInfer's CuTe DSL swap-AB order (`up_gate_block16_rows_`), w2 rows plain.
- Block scales in CUTLASS 128x4 atoms over the same rows (`swizzle_blockscale`, the same bytes as
  FlashInfer `block_scale_interleave`).
"""

import torch
from flashinfer import CuteDslMoEWrapper, SfLayout
from flashinfer import autotune as flashinfer_autotune
from flashinfer import mxfp8_quantize, nvfp4_quantize
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

# NVFP4 activation global scale: 4over6 per-token scales with an E4M3 max of 256.
_NVFP4_ACT_GLOBAL_SCALE = 1.0 / (6.0 * 256.0)
_AUTOTUNE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def up_gate_block16_rows_(t: torch.Tensor) -> None:
    """Reorder the rows (dim 1) of a `[E, 2 * I, ...]` w13 weight or block-scale tensor in place,
    from per-row gate/up (gate j at 2j, up j at 2j + 1) to FlashInfer's CuTe DSL swap-AB order:
    per 32 rows, up 16b .. 16b + 15 then gate 16b .. 16b + 15."""
    assert t.dim() == 3 and t.shape[1] % 32 == 0
    r = torch.arange(t.shape[1], device=t.device)
    index = 2 * (16 * (r // 32) + r % 16) + 1 - (r % 32) // 16
    rows = t.view(torch.uint8)
    for expert in range(rows.shape[0]):
        rows[expert] = rows[expert, index]


class Inkling2FlashInferMoE:
    def __init__(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        num_experts: int,
        top_k: int,
        local_expert_offset: int,
        max_num_tokens: int,
        situ_beta: float,
        situ_linear_beta: float,
        w13_alpha: torch.Tensor | None = None,
        w2_alpha: torch.Tensor | None = None,
    ):
        """`w13` `[E, 2 * I, K]`, `w2` `[E, H, I]` (packed NVFP4 halves K), scales as above.
        `w13_alpha` / `w2_alpha` `[E]` float32: NVFP4 per-expert global weight scales (required for
        NVFP4, ignored for MXFP8)."""
        self.nvfp4 = w13.dtype == torch.uint8
        assert self.nvfp4 or w13.dtype == torch.float8_e4m3fn
        num_local, rows, _ = w13.shape
        hidden, intermediate = w2.shape[1], rows // 2
        vec = 16 if self.nvfp4 else 32

        def mma_scales(scale, m, k):
            return convert_sf_to_mma_layout(
                scale.view(torch.uint8).view(num_local * m, k // vec),
                m=m,
                k=k,
                num_groups=num_local,
                sf_vec_size=vec,
            )

        self.weights = dict(
            w1_weight=w13,
            w1_weight_sf=mma_scales(w13_scale, rows, hidden),
            w2_weight=w2,
            w2_weight_sf=mma_scales(w2_scale, hidden, intermediate),
        )
        if self.nvfp4:
            assert w13_alpha is not None and w2_alpha is not None
            self.recipe = NVFP44Over6Config(
                e4m3_max=256, err_mode="MAE", err_use_fast_math=False
            )
            self.gscale = torch.full(
                (1,), _NVFP4_ACT_GLOBAL_SCALE, device=w13.device, dtype=torch.float32
            )
            self.weights.update(
                w1_alpha=w13_alpha.float().contiguous(),
                w2_alpha=w2_alpha.float().contiguous(),
                fc2_input_scale=self.gscale,
                nvfp4_4over6=self.recipe,
            )
        else:
            ones = torch.ones(num_local, device=w13.device, dtype=torch.float32)
            self.weights.update(w1_alpha=ones, w2_alpha=ones, fc2_input_scale=None)
        self.layout = SfLayout.layout_linear
        self.num_experts, self.top_k, self.hidden = num_experts, top_k, hidden
        self.max_num_tokens = max_num_tokens
        self.wrapper = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden,
            intermediate_size=intermediate,
            use_cuda_graph=True,
            max_num_tokens=max_num_tokens,
            num_local_experts=num_local,
            local_expert_offset=local_expert_offset,
            quant_mode="w4a4" if self.nvfp4 else "mxfp8",
            swap_ab=True,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def __call__(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """`x` `[T, H]` bf16, `topk_weights` / `topk_ids` `[T, top_k]` (global expert ids)."""
        if self.nvfp4:
            values, scales, per_token = nvfp4_quantize(
                x,
                self.gscale,
                sfLayout=self.layout,
                per_token_activation=True,
                backend="cute-dsl",
                nvfp4_4over6=self.recipe,
            )
            scales = scales.unsqueeze(-1)
        else:
            values, scales = mxfp8_quantize(
                x, is_sf_swizzled_layout=False, alignment=128, backend="cuda"
            )
            scales = scales.view(torch.uint8).reshape(x.shape[0], x.shape[1] // 32)
            per_token = None
        return self.wrapper.run(
            x=values,
            x_sf=scales,
            token_selected_experts=topk_ids.int(),
            token_final_scales=topk_weights.float(),
            per_token_scale=per_token,
            **self.weights,
        )

    def autotune(self) -> None:
        """Pick tactics for the power-of-two token counts up to `max_num_tokens`, on random routes.
        Run once before CUDA graph capture."""
        tokens, device = self.max_num_tokens, self.weights["w1_weight"].device
        x = torch.randn(tokens, self.hidden, device=device, dtype=torch.bfloat16)
        ids = torch.stack(
            [
                torch.randperm(self.num_experts, device=device)[: self.top_k]
                for _ in range(tokens)
            ]
        )
        weights = torch.rand(tokens, self.top_k, device=device, dtype=torch.float32)
        with flashinfer_autotune(True):
            for n in _AUTOTUNE_TOKENS:
                if n <= tokens:
                    self(x[:n], weights[:n], ids[:n])
        torch.cuda.synchronize()
