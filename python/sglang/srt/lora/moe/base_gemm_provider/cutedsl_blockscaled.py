"""CuTeDSL MXFP8, MXFP4 and NVFP4 block-scaled GEMMs with BF16 outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_blockscaled,
)
from sglang.kernels.ops.lora.moe.dispatch_masked import dispatch_fill_masked_blockscaled
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    RowBuffers,
    prepare_buffer,
)
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_common import CuteDslTileMixin
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
)
from sglang.srt.lora.moe.quant_info import (
    MoeLoraBlockScaledQuantInfo,
    MoeLoraMxfp4QuantInfo,
    MoeLoraMxfp8QuantInfo,
    MoeLoraNvFp4QuantInfo,
)

if TYPE_CHECKING:
    from sglang.kernels.ops.lora.moe.quant_blockscaled import BlockScaledRows
    from sglang.srt.lora.workspace import LoraWorkspace


class _CuteDslBlockScaledMixin(CuteDslTileMixin):
    WIDE_EXPECTED_M_THRESHOLD = 32
    SF_VEC_SIZE = 32
    derived_base_weights = "packed block-scale atoms"
    supports_epilogue_down_lora = True
    # TMEM fits one base accumulator stage plus the LoRA one up to this tile width.
    EPILOGUE_LORA_SEPARATE_MAX_WIDTH = 128

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """Allocate gate/up rows for the fill and down rows for activation or down()."""
        device = hidden_states.device
        rows_q, sf_rows, rows_rs = self._quant_buffers(
            workspace, f"{prefix}:rows", lead, hidden_states.size(1), device
        )
        act_q, act_sf, act_rs = self._quant_buffers(
            workspace, f"{prefix}:act", lead, self.quant_info.intermediate_size, device
        )
        return rows_q, dict(
            sf_rows=sf_rows,
            rows_rs=rows_rs,
            act_q=act_q,
            act_sf=act_sf,
            act_rs=act_rs,
        )

    def _input_spec(self, rows, buffers):
        return self._rows_spec(
            "gemm1", rows, buffers["sf_rows"], rows.shape[-2], buffers["rows_rs"]
        )

    def _fill_masked(
        self, hidden_states, topk_ids, top_k, masked_m, pair_to_row, rows, buffers
    ) -> None:
        dispatch_fill_masked_blockscaled(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=self._input_spec(rows, buffers),
        )

    def _fill_contiguous(
        self, hidden_states, topk_ids, pair_to_row, seg_offsets, rows, buffers
    ) -> None:
        dispatch_fill_rows_contiguous_blockscaled(
            hidden_states,
            topk_ids,
            pair_to_row,
            rows_out=self._input_spec(rows, buffers),
            token_scales=buffers.get("token_rs"),
        )

    def _prepare_dummy(self, stage: str, config, lora_k: int = 0):
        weight = self._stage_weight(stage)
        device = weight.device
        _, n, k_store = weight.shape
        rows, scales, groups = self._dummy_layout(device)
        return self._prepare_gemm(
            torch.zeros((*rows, k_store), dtype=weight.dtype, device=device),
            weight,
            torch.zeros(
                self._scale_atom_bytes(
                    256, k_store * self._pack, self.SF_VEC_SIZE, *rows[:-1]
                ),
                dtype=torch.uint8,
                device=device,
            ),
            self._stage_scale(stage),
            torch.empty((*rows, n), dtype=torch.bfloat16, device=device),
            *self._dummy_groups(groups),
            config=config,
            sf_vec_size=self.SF_VEC_SIZE,
            alpha=self._alpha[stage],
            alpha_rows_per_half=self._alpha_rows_per_half[stage],
            **self._lora_dummy_kwargs(lora_k, scales[0], n, 256, device),
            tok_scale=(
                torch.zeros(scales, dtype=torch.float32, device=device)
                if self._per_row
                else None
            ),
        )

    def _lora_dummy_kwargs(
        self, lora_k: int, planes_per_slot: int, n: int, rows: int, device
    ) -> dict:
        """The LoRA build specializes on padded K. Weight rank is dynamic."""
        if not lora_k:
            return {}
        experts = self.quant_info.num_local_experts
        return dict(
            lora_w=torch.zeros(
                (experts, n, lora_k), dtype=torch.bfloat16, device=device
            ),
            lora_planes=torch.zeros(
                (planes_per_slot, rows, lora_k), dtype=torch.bfloat16, device=device
            ),
            lora_slots=torch.zeros((1,), dtype=torch.int32, device=device),
        )

    def _bind_weights(self, quant_info: MoeLoraBlockScaledQuantInfo) -> None:
        from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
        from sglang.kernels.ops.lora.moe.quant_blockscaled import (
            BlockScaledRows,
            quantize_rows,
        )

        self._scale_atom_bytes = scale_atom_bytes
        self._rows_cls = BlockScaledRows
        self._quantize_rows = quantize_rows
        self._pack = quant_info.elements_per_byte
        # FP4 pairs may load as int8; the kernels take packed uint8 storage.
        store_dtype = torch.uint8 if self._pack == 2 else quant_info.w13_weight.dtype
        self._w13 = quant_info.w13_weight.view(store_dtype)
        self._w2 = quant_info.w2_weight.view(store_dtype)
        self._sf_w13, self._sf_w2 = quant_info.packed_scales
        self._alpha = {"gemm1": None, "gemm2": None}
        self._alpha_rows_per_half = {"gemm1": 1, "gemm2": 1}
        self._act_scale = {"gemm1": None, "gemm2": None}
        # Per-token NVFP4: row scale = row_amax / denominator, applied in the epilogue.
        self._per_row = False
        self._row_denominator = 0.0
        self._nvfp4_4over6 = None  # NVFP4 per-token: the resolved activation recipe
        # Unread stand-ins for operands a build does not use (alpha, row scales, LoRA).
        device = self._w13.device
        self._alpha_ones = torch.ones(
            (quant_info.num_local_experts, 1), dtype=torch.float32, device=device
        )
        self._tok_scale_dummy = torch.zeros((1, 16), dtype=torch.float32, device=device)
        self._lora_dummies = (
            torch.zeros((1, 16, 16), dtype=torch.bfloat16, device=device),
            torch.zeros((1, 16, 16), dtype=torch.bfloat16, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )

    def _stage_constexprs(self, stage: str) -> tuple:
        return (
            self.SF_VEC_SIZE,
            self._alpha[stage] is not None,
            self._alpha_rows_per_half[stage],
            self._per_row,
        )

    def _quant_buffers(self, workspace, prefix, rows_shape, k, device):
        """``rows_shape`` is ``(slabs, rows)`` (expert slabs) or ``(rows,)`` (flat)."""
        *slabs, rows = rows_shape
        data = prepare_buffer(
            workspace,
            f"{prefix}:q",
            (*slabs, rows, k // self._pack),
            dtype=self._w13.dtype,
            device=device,
        )
        scales = prepare_buffer(
            workspace,
            f"{prefix}:sf",
            (self._scale_atom_bytes(rows, k, self.SF_VEC_SIZE, *slabs),),
            dtype=torch.uint8,
            device=device,
        )
        row_scales = None
        if self._per_row:
            # Zeroed once: rows no fill writes (padding, unrouted) keep a finite scale.
            row_scales = prepare_buffer(
                workspace,
                f"{prefix}:rs",
                (math.prod(slabs) * rows,),
                dtype=torch.float32,
                device=device,
                zero_on_first_allocation=True,
            )
        return data, scales, row_scales

    def _rows_spec(self, stage, data, scales, rows_per_slab, row_scales=None):
        recipe = self._nvfp4_4over6
        return self._rows_cls(
            data=data.reshape(-1, data.shape[-1]),
            scales=scales,
            sf_vec_size=self.SF_VEC_SIZE,
            rows_per_slab=rows_per_slab,
            global_scale=self._act_scale[stage],
            row_scales=row_scales,
            row_denominator=self._row_denominator,
            use_4over6=recipe is not None,
            err_mode="MAE" if recipe is None else recipe.err_mode,
            err_use_fast_math=recipe is not None and recipe.err_use_fast_math,
        )

    def _act_quant_target(
        self, row_state, *, fused_config: Mapping[str, int] | None = None
    ) -> BlockScaledRows | None:
        # Fused activation can only quantize whole scale groups within its tile.
        if fused_config is not None and (
            self._per_row or int(fused_config["BLOCK_SIZE_W"]) % self.SF_VEC_SIZE
        ):
            return None
        return self._down_rows_spec(row_state)

    def _down_rows_spec(self, row_state) -> BlockScaledRows:
        return self._rows_spec(
            "gemm2",
            row_state.act_q,
            row_state.act_sf,
            self.act_out_shape(row_state)[-2],
            row_state.act_rs,
        )

    def _launch(
        self,
        stage: str,
        a_q: torch.Tensor,
        sf_tokens: torch.Tensor,
        c: torch.Tensor,
        row_state,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
        lora: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        tok_scale: torch.Tensor | None = None,
    ) -> None:
        call = self._compiled[row_state.token_width][
            f"{stage}:lora" if lora is not None else stage
        ]
        lora_w, lora_p, lora_slots = lora if lora is not None else self._lora_dummies
        alpha = self._alpha[stage]
        dyn = self._as_dynamic_cute_tensor
        a_rows = self._rows_arg(a_q)
        tok_scale = (
            tok_scale.view(a_rows.shape[0], -1)
            if self._per_row
            else self._tok_scale_dummy
        )
        call.compiled_fn(
            dyn(a_rows, leading_dim=2),
            call.b_arg,
            dyn(sf_tokens, leading_dim=0),
            dyn(call.sf_weights, leading_dim=0),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._alpha_ones if alpha is None else alpha, leading_dim=1),
            dyn(tok_scale, leading_dim=1),
            dyn(lora_w, leading_dim=2),
            dyn(lora_p, leading_dim=2),
            dyn(lora_slots, leading_dim=0),
            dyn(self._group_rows_arg(row_state), leading_dim=0),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_q.device),
        )

    def enable_epilogue_down_lora(self, rank: int) -> None:
        """TMA needs rank divisible by 8. Pad the LoRA MMA K to a multiple of 16."""
        if rank % 8:
            raise ValueError(
                f"the epilogue down-LoRA takes ranks that are multiples of 8, got {rank}"
            )
        # The kernel adds a separate accumulator exactly when alpha is applied.
        if self._alpha["gemm2"] is not None:
            wide = sorted(
                width
                for width in self._tile_configs
                if width > self.EPILOGUE_LORA_SEPARATE_MAX_WIDTH
            )
            if wide:
                raise ValueError(
                    "the NVFP4 epilogue down-LoRA accumulates into a second TMEM "
                    "accumulator that fits beside token tiles up to "
                    f"{self.EPILOGUE_LORA_SEPARATE_MAX_WIDTH} wide; this provider "
                    f"compiles {wide}-wide tiles"
                )
        lora_k = -(-rank // 16) * 16
        for token_width, config in self._tile_configs.items():
            self._compile_stage(token_width, "gemm2", config, lora_k=lora_k)
        torch.cuda.synchronize(self._w13.device)

    def epilogue_down_lora_planes(
        self, row_state, workspace, *, max_loras: int, rank: int
    ) -> torch.Tensor:
        """Zero BF16 [max_loras, rows, round_up(rank, 16)] planes each forward
        because down-A leaves some rows unwritten."""
        rows = self._rows_arg(row_state.act_q)
        shape = (max_loras, rows.shape[0] * rows.shape[1], -(-rank // 16) * 16)
        planes = prepare_buffer(
            workspace,
            "down_epilogue:planes",
            shape,
            dtype=torch.bfloat16,
            device=rows.device,
        )
        planes.zero_()
        return planes

    def gateup(self, row_state, out: torch.Tensor) -> None:
        self._launch(
            "gemm1",
            self._input_rows(row_state),
            row_state.sf_rows,
            out,
            row_state,
            row_state.gemm1_schedule,
            row_state.gemm1_tiles,
            tok_scale=row_state.rows_rs,
        )

    def down(
        self,
        row_state,
        act_out: torch.Tensor,
        out: torch.Tensor,
        lora: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Epilogue LoRA takes [slots * E, hidden, rank] B weights, bridge planes,
        and a count-prefixed int32 slot list."""
        if not row_state.act_quant_ready:
            self._quantize_rows(
                act_out.reshape(-1, act_out.shape[-1]),
                self._down_rows_spec(row_state),
                row_state.pair_to_row,
            )
        if lora is not None:
            # The kernel's (plane, rows, rank) view: one plane per slot for
            # flat rows, per (slot, expert) slab for masked rows.
            weights, planes, slots = lora
            rows_per_plane = self._rows_arg(row_state.act_q).shape[1]
            lora = weights, planes.view(-1, rows_per_plane, planes.shape[-1]), slots
        self._launch(
            "gemm2",
            row_state.act_q,
            row_state.act_sf,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
            lora=lora,
            tok_scale=row_state.act_rs,
        )


def _check_per_token_alpha(stage: str, alpha: torch.Tensor, k: int) -> None:
    """Bound FP32 ``C * alpha`` in the per-token epilogue ``(C * alpha) * s``.
    ``C * s`` can overflow and ``alpha * s`` can underflow. For E2M1 times E4M3
    operands, nonzero |C| is in [2^-20, K * (6 * 448)^2]."""
    bound = torch.finfo(torch.float32).max / (k * (6.0 * 448.0) ** 2)
    magnitude = alpha.abs()
    largest = float(magnitude.max())
    if not math.isfinite(largest) or largest > bound:
        raise ValueError(
            f"per-token NVFP4 {stage} weight global scales reach {largest:.3g}; "
            f"the epilogue's C * alpha stays finite only up to {bound:.3g} (K={k})"
        )
    nonzero = magnitude[magnitude > 0]
    tiny = torch.finfo(torch.float32).tiny
    if nonzero.numel() and float(nonzero.min()) < tiny:
        raise ValueError(
            f"per-token NVFP4 {stage} weight global scales include "
            f"{float(nonzero.min()):.3g}, a subnormal fp32; below {tiny:.3g} the "
            "epilogue's C * alpha can round to zero for a nonzero accumulator "
            "(zero scales are allowed)"
        )


class _CuteDslNvFp4Mixin(_CuteDslBlockScaledMixin):
    SF_VEC_SIZE = 16
    derived_base_weights = "packed block-scale atoms and global-scale products"

    def _bind_weights(self, quant_info: MoeLoraNvFp4QuantInfo) -> None:
        super()._bind_weights(quant_info)
        w13_global, w2_global = quant_info.alpha
        if quant_info.act_scale_mode == "static":
            self._act_scale = {
                "gemm1": quant_info.w13_input_scale,
                "gemm2": quant_info.w2_input_scale,
            }
        else:
            self._per_row = True
            self._row_denominator = quant_info.activation_denominator
            self._nvfp4_4over6 = quant_info.recipe
            _check_per_token_alpha("gemm1", w13_global, quant_info.hidden_size)
            _check_per_token_alpha("gemm2", w2_global, quant_info.intermediate_size)
        self._alpha["gemm1"] = w13_global
        self._alpha_rows_per_half["gemm1"] = (
            quant_info.intermediate_size if w13_global.shape[1] == 2 else 1 << 30
        )
        self._alpha["gemm2"] = w2_global
        self._alpha_rows_per_half["gemm2"] = 1 << 30


class CuteDslBlockScaledMaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    sf_rows: torch.Tensor  # uint8 scale atoms
    act_q: torch.Tensor  # [E_local, m_max, inter / pack]
    act_sf: torch.Tensor
    act_quant_ready: bool = False
    # Per-token NVFP4: the stored fp32 row scales of each quantized buffer.
    rows_rs: torch.Tensor | None = None
    act_rs: torch.Tensor | None = None
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class _CuteDslBlockScaledMaskedProvider(
    _CuteDslBlockScaledMixin, MaskedRowDomainProvider
):
    _ROW_MODE = "masked"
    _row_state_cls = CuteDslBlockScaledMaskedRowState

    def __init__(self, quant_info: MoeLoraBlockScaledQuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_blockscaled

        self._prepare_gemm = prepare_masked_blockscaled
        # Token tiles cover half or all of a 128-row scale atom.
        self._init_tiles(quant_info, drop_narrow_tile=True)

    @staticmethod
    def _group_rows_arg(row_state: MaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    @staticmethod
    def _dummy_groups(masked_m: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (masked_m,)


class CuteDslBlockScaledContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    sf_rows: torch.Tensor
    act_q: torch.Tensor  # [m_pad_ceiling, inter / pack]
    act_sf: torch.Tensor
    act_quant_ready: bool = False
    # Per-token NVFP4: the stored fp32 row scales of each quantized buffer.
    rows_rs: torch.Tensor | None = None
    act_rs: torch.Tensor | None = None
    token_rs: torch.Tensor | None = None
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class _CuteDslBlockScaledContiguousProvider(
    _CuteDslBlockScaledMixin, ContiguousRowDomainProvider
):
    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslBlockScaledContiguousRowState

    def __init__(self, quant_info: MoeLoraBlockScaledQuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_contiguous_blockscaled,
        )

        self._prepare_gemm = prepare_contiguous_blockscaled
        # Token tiles cover half or all of a 128-row scale atom.
        self._init_tiles(quant_info, drop_narrow_tile=True)

    @staticmethod
    def _group_rows_arg(row_state: ContiguousRowState) -> torch.Tensor:
        return row_state.seg_counts

    def _dummy_groups(self, seg_offsets: torch.Tensor) -> tuple[torch.Tensor, ...]:
        experts = self.quant_info.num_local_experts
        return seg_offsets, seg_offsets.new_zeros(experts)

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """The fill prepass writes one FP32 token_rs scale per token before fanout."""
        rows, buffers = super()._input_buffers(workspace, prefix, lead, hidden_states)
        if self._per_row:
            buffers["token_rs"] = prepare_buffer(
                workspace,
                f"{prefix}:token_rs",
                (max(int(hidden_states.size(0)), 1),),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        return rows, buffers


def _contract(key: str, quant_info_cls) -> MoeBaseProviderContract:
    return MoeBaseProviderContract(key=key, quant_info_cls=quant_info_cls)


class CuteDslMxfp8MaskedProvider(_CuteDslBlockScaledMaskedProvider):
    contract = _contract("cutedsl_mxfp8_masked", MoeLoraMxfp8QuantInfo)
    _DTYPE_TAG = "mxfp8"


class CuteDslMxfp8ContiguousProvider(_CuteDslBlockScaledContiguousProvider):
    contract = _contract("cutedsl_mxfp8_contiguous", MoeLoraMxfp8QuantInfo)
    _DTYPE_TAG = "mxfp8"


class CuteDslMxfp4MaskedProvider(_CuteDslBlockScaledMaskedProvider):
    contract = _contract("cutedsl_mxfp4_masked", MoeLoraMxfp4QuantInfo)
    _DTYPE_TAG = "mxfp4"


class CuteDslMxfp4ContiguousProvider(_CuteDslBlockScaledContiguousProvider):
    contract = _contract("cutedsl_mxfp4_contiguous", MoeLoraMxfp4QuantInfo)
    _DTYPE_TAG = "mxfp4"


class CuteDslNvFp4MaskedProvider(_CuteDslNvFp4Mixin, _CuteDslBlockScaledMaskedProvider):
    contract = _contract("cutedsl_nvfp4_masked", MoeLoraNvFp4QuantInfo)
    _DTYPE_TAG = "nvfp4"


class CuteDslNvFp4ContiguousProvider(
    _CuteDslNvFp4Mixin, _CuteDslBlockScaledContiguousProvider
):
    contract = _contract("cutedsl_nvfp4_contiguous", MoeLoraNvFp4QuantInfo)
    _DTYPE_TAG = "nvfp4"
