"""CuTeDSL GEMMs on block-scaled checkpoints: MXFP8 (E4M3 x E8M0 per 32),
MXFP4 (E2M1 pairs x E8M0 per 32) and NVFP4 (E2M1 pairs x E4M3 per 16 under
per-expert global scales).

Activations are quantized per row into the weights' format with the scales
packed in the tcgen05 atom order; GEMM outputs remain BF16 for LoRA.
"""

from __future__ import annotations

import math
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
    MoeLoraMxfp4QuantInfo,
    MoeLoraMxfp8QuantInfo,
    MoeLoraMxQuantInfo,
    MoeLoraNvFp4QuantInfo,
)

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace


def _contract(key: str, quant_info_cls) -> MoeBaseProviderContract:
    return MoeBaseProviderContract(
        key=key,
        quant_info_cls=quant_info_cls,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )


class _CuteDslBlockScaledMixin(CuteDslTileMixin):
    # Token tiles cover half or all of a 128-row scale atom.
    TOKEN_WIDTHS = (
        CuteDslTileMixin.WIDE_TOKEN_WIDTH,
        CuteDslTileMixin.XWIDE_TOKEN_WIDTH,
    )
    WIDE_EXPECTED_M_THRESHOLD = 32
    SF_VEC_SIZE = 32
    derived_base_weights = "packed block-scale atoms"

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """The row domains' hook for block-scaled rows: two buffer sets from
        _quant_buffers(), the gate/up input under ``<prefix>:rows`` and the
        down input under ``<prefix>:act``. ``rows`` is ``rows_q``. ``buffers``:

          rows_q, sf_rows, rows_rs  the quantized input rows, their packed
                                    scale atoms and, per token, their stored
                                    FP32 row scales (None when static); the
                                    dispatch fill writes all of them
          act_q, act_sf, act_rs     the same three for the down GEMM's input;
                                    the activation writes them when
                                    _act_quant_target() gives it a target, and
                                    down() quantizes into them when the
                                    activation left ``act_quant_ready`` false
        """
        device = hidden_states.device
        rows_q, sf_rows, rows_rs = self._quant_buffers(
            workspace, f"{prefix}:rows", lead, hidden_states.size(1), device
        )
        act_q, act_sf, act_rs = self._quant_buffers(
            workspace, f"{prefix}:act", lead, self.quant_info.intermediate_size, device
        )
        return rows_q, dict(
            rows_q=rows_q,
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
            groups,
            config=config,
            sf_vec_size=self.SF_VEC_SIZE,
            alpha=self._alpha[stage],
            alpha_rows_per_half=self._alpha_rows_per_half[stage],
            **self._lora_dummy_kwargs(lora_k, scales[0], n, 256, device),
            lora_acc=self.EPILOGUE_LORA_ACC,
            tok_scale=(
                torch.zeros(scales, dtype=torch.float32, device=device)
                if self._per_row
                else None
            ),
        )

    def _bind_weights(self, quant_info: MoeLoraMxQuantInfo) -> None:
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
        # NVFP4 fills these with per-expert global scales and activation scales.
        self._alpha = {"gemm1": None, "gemm2": None}
        self._alpha_rows_per_half = {"gemm1": 1, "gemm2": 1}
        self._act_scale = {"gemm1": None, "gemm2": None}
        # NVFP4 per-token: stored fp32 row scales (row_amax / denominator)
        # beside every quantized row buffer, applied in the GEMM epilogue.
        self._per_row = False
        self._row_denominator = 0.0
        self._recipe = None  # NVFP4 per-token: the resolved activation recipe
        self._tok_scale_dummy = None
        # The epilogue down-LoRA: compiled on request for one rank.
        self._epilogue_lora_rank = 0
        self._epilogue_lora_k = 0
        self._lora_dummies = None

    def _stage_weight(self, stage: str) -> torch.Tensor:
        if stage == "gemm1":
            return self._w13
        if stage == "gemm2":
            return self._w2
        raise ValueError(f"unknown CuTeDSL base stage {stage!r}")

    def _stage_scale(self, stage: str) -> torch.Tensor:
        return self._sf_w13 if stage == "gemm1" else self._sf_w2

    def _stage_constexprs(self, stage: str) -> tuple:
        return (
            self.SF_VEC_SIZE,
            self._alpha[stage] is not None,
            self._alpha_rows_per_half[stage],
            self._per_row,
        )

    def _quant_buffers(self, workspace, prefix, rows_shape, k, device):
        """Quantized rows, their scale atoms and, per token, their stored row
        scales (None in static mode); ``rows_shape`` is ``(slabs, rows)`` for
        expert slabs or ``(rows,)`` for flat rows."""
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
            # Zeroed once so rows the fills never write (padding, unrouted
            # pairs) carry a finite scale; live rows are rewritten per forward.
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
        """The quantized rows of one GEMM input, as the kernels address them."""
        if self._per_row and row_scales is None:
            raise ValueError(f"per-token {stage} rows need their row-scale buffer")
        recipe = self._recipe
        return self._rows_cls(
            data=data.reshape(-1, data.shape[-1]),
            scales=scales,
            sf_vec_size=self.SF_VEC_SIZE,
            rows_per_slab=rows_per_slab,
            global_scale=self._act_scale[stage],
            row_scales=row_scales if self._per_row else None,
            row_denominator=self._row_denominator,
            candidate_select=recipe is not None,
            err_mode="MAE" if recipe is None else recipe.err_mode,
            err_use_fast_math=recipe is not None and recipe.err_use_fast_math,
        )

    def _act_quant_target(self, row_state, fused, config):
        """The unfused activation always writes the down input, per-token rows
        in its second pass. The fused activation works one column tile at a
        time: it writes only when the tile holds whole scale groups, and never
        a per-token scale, which needs the whole row first."""
        if fused and (self._per_row or int(config["BLOCK_SIZE_W"]) % self.SF_VEC_SIZE):
            return None
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
        """``lora`` = (weights [slots * E, N, rank], planes, slots int32 [1])
        selects the stage's epilogue-LoRA build; the plain build takes unread
        dummies in those slots. ``tok_scale`` is the input rows' stored
        per-token scale buffer (per-token mode only)."""
        call = self._compiled[row_state.token_width][
            f"{stage}:lora" if lora is not None else stage
        ]
        lora_w, lora_p, lora_slots = (
            lora if lora is not None else self._lora_dummy_operands(a_q.device)
        )
        # Runtime wrapping must match the compiled argument's MLIR type.
        dyn = self._as_dynamic_cute_tensor
        a_rows = self._rows_arg(a_q)
        call.compiled_fn(
            dyn(a_rows, leading_dim=2),
            call.b_arg,
            dyn(sf_tokens, leading_dim=0),
            dyn(call.sf_weights, leading_dim=0),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._alpha_arg(stage), leading_dim=1),
            dyn(self._tok_scale_arg(tok_scale, a_rows), leading_dim=1),
            dyn(lora_w, leading_dim=2),
            dyn(lora_p, leading_dim=2),
            dyn(lora_slots, leading_dim=0),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_q.device),
        )

    def _lora_dummy_operands(self, device: torch.device) -> tuple:
        if self._lora_dummies is None:
            self._lora_dummies = (
                torch.zeros((1, 16, 16), dtype=torch.bfloat16, device=device),
                torch.zeros((1, 16, 16), dtype=torch.bfloat16, device=device),
                torch.zeros((1,), dtype=torch.int32, device=device),
            )
        return self._lora_dummies

    def _tok_scale_arg(self, tok_scale: torch.Tensor | None, a_rows) -> torch.Tensor:
        """The per-token scale operand, (planes, rows) over the stage's input
        rows ``a_rows``, or the unread dummy of a static build."""
        if not self._per_row:
            if self._tok_scale_dummy is None:
                self._tok_scale_dummy = torch.zeros(
                    (1, 16), dtype=torch.float32, device=a_rows.device
                )
            return self._tok_scale_dummy
        if tok_scale is None:
            raise ValueError("per-token rows launch with their row-scale buffer")
        return tok_scale.view(a_rows.shape[0], -1)

    # The epilogue down-LoRA: the down GEMM adds ``down_lora_b @ bridge`` per
    # (slot, expert) tile; the raw bridge arrives in the provider's own row
    # order on one plane per slot. The MX families' LoRA MMAs share the base
    # accumulator, which nothing scales afterwards. NVFP4 scales the base
    # accumulator in the epilogue (alpha, then the per-token scale), so its
    # LoRA MMAs fill a second TMEM accumulator that the epilogue adds last:
    # C * alpha + D when static, (C * alpha) * s + D per token.
    supports_epilogue_down_lora = True
    EPILOGUE_LORA_ACC = "shared"
    # One base accumulator stage plus the LoRA accumulator fit TMEM beside
    # token tiles up to this width (the kernel's stage rule).
    EPILOGUE_LORA_SEPARATE_MAX_WIDTH = 128

    def enable_epilogue_down_lora(self, rank: int) -> None:
        """Compile the down stage with the epilogue LoRA for ``rank`` (a
        multiple of 8: the weight rows are TMA tiles; the MMA K pads to 16)."""
        if rank % 8:
            raise ValueError(
                f"the epilogue down-LoRA takes ranks that are multiples of 8, got {rank}"
            )
        if self.EPILOGUE_LORA_ACC == "separate":
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
        if self._epilogue_lora_rank == rank:
            return
        lora_k = -(-rank // 16) * 16
        for token_width, config in self._tile_configs.items():
            self._compile_stage(token_width, "gemm2", config, lora_k=lora_k)
        torch.cuda.synchronize(self._w13.device)
        self._epilogue_lora_rank, self._epilogue_lora_k = rank, lora_k

    def epilogue_down_lora_rank(self) -> int:
        return self._epilogue_lora_rank

    def epilogue_down_lora_planes(
        self, row_state, workspace, *, max_loras: int, rank: int
    ) -> torch.Tensor:
        """Zeroed BF16 bridge planes [max_loras, rows, rank] over the provider's
        down-input rows (every row of every slot, so unrouted and padded
        rows add nothing)."""
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

    def _epilogue_lora_operands(self, row_state, lora: tuple) -> tuple:
        """Reshape the planes to the kernel's (plane, rows, rank) view: one
        plane per slot for flat rows, per (slot, expert) slab for masked rows."""
        weights, planes, slots = lora
        rows_per_plane = self._rows_arg(row_state.act_q).shape[1]
        if weights.shape[0] != planes.shape[0] * self.quant_info.num_local_experts:
            raise ValueError(
                "the epilogue down-LoRA takes one weight plane per (slot, expert): "
                f"got weights {tuple(weights.shape)} for {planes.shape[0]} slots"
            )
        return weights, planes.view(-1, rows_per_plane, planes.shape[-1]), slots

    def _alpha_arg(self, stage: str) -> torch.Tensor:
        alpha = self._alpha[stage]
        if alpha is None:
            alpha = self._alpha[stage] = torch.ones(
                (self.quant_info.num_local_experts, 1),
                dtype=torch.float32,
                device=self._w13.device,
            )
        return alpha

    def _lora_dummy_kwargs(
        self, lora_k: int, planes_per_slot: int, n: int, rows: int, device
    ) -> dict:
        """Zero epilogue-LoRA operands of one slot for a compile-time build;
        the weights' rank extent is dynamic, so the padded K is the build."""
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

    def gateup(self, row_state, out: torch.Tensor) -> None:
        self._launch(
            "gemm1",
            row_state.rows_q,
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
        """``lora`` = (down_lora_b [slots * E, hidden, rank], bridge planes from
        epilogue_down_lora_planes, slots int32 [1]) adds the down-LoRA delta
        inside the GEMM."""
        if not row_state.act_quant_ready:
            # The activation that just ran left BF16 rows only (a fused
            # activation whose column tile splits a scale group, or that would
            # need a per-token scale): quantize them into the down input here.
            self._quantize_rows(
                act_out.reshape(-1, act_out.shape[-1]),
                self._rows_spec(
                    "gemm2",
                    row_state.act_q,
                    row_state.act_sf,
                    act_out.shape[-2],
                    row_state.act_rs,
                ),
                row_state.pair_to_row,
            )
        if lora is not None:
            if self._epilogue_lora_rank != lora[0].shape[-1]:
                raise ValueError(
                    f"the epilogue down-LoRA was compiled for rank "
                    f"{self._epilogue_lora_rank}, not {lora[0].shape[-1]}"
                )
            lora = self._epilogue_lora_operands(row_state, lora)
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
    """The per-token epilogue evaluates ``(C * alpha) * s``, alpha first: the
    stored row scale ``s`` spans every BF16 row, so ``C * s`` overflows on rows
    at the BF16 maximum and ``alpha * s`` can underflow to zero. The domain of
    alpha (a weight global scale per expert) is zero or a normal fp32 at most
    ``fp32_max / (K * (6 * 448)^2)``: above that ``C * alpha`` overflows for an
    accumulator the MMA can produce (``|C| <= K * (6 * 448)^2``); below the
    normal range it can round to zero for a nonzero accumulator (``|C| >=
    2^-20``, a product of an E2M1 code and an E4M3 scale per operand). Inside
    the domain ``C * alpha`` is nonzero for every nonzero accumulator, and the
    epilogue's two roundings, ``fl(C * alpha)`` then ``fl(. * s)``, each add
    2^-24 of their result when it is a normal fp32 and at most 2^-150 absolute
    when it is subnormal."""
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
    EPILOGUE_LORA_ACC = "separate"
    derived_base_weights = "packed block-scale atoms and global-scale products"

    def _bind_weights(self, quant_info: MoeLoraNvFp4QuantInfo) -> None:
        super()._bind_weights(quant_info)
        # The epilogue applies alpha per expert: the weight global scales times
        # the activation scale, folded in when static and applied per output
        # column from the stored row scales when per-token.
        w13_global, w2_global = quant_info.alpha
        if quant_info.act_scale_mode == "static":
            self._act_scale = {
                "gemm1": quant_info.w13_input_scale,
                "gemm2": quant_info.w2_input_scale,
            }
        else:
            self._per_row = True
            self._row_denominator = quant_info.activation_denominator
            self._recipe = quant_info.recipe
            _check_per_token_alpha("gemm1", w13_global, quant_info.hidden_size)
            _check_per_token_alpha("gemm2", w2_global, quant_info.intermediate_size)
        self._alpha["gemm1"] = w13_global
        self._alpha_rows_per_half["gemm1"] = (
            quant_info.intermediate_size if w13_global.shape[1] == 2 else 1 << 30
        )
        self._alpha["gemm2"] = w2_global
        self._alpha_rows_per_half["gemm2"] = 1 << 30


class CuteDslBlockScaledMaskedRowState(MaskedRowState, kw_only=True):
    """MaskedRowState plus the buffers of
    _CuteDslBlockScaledMixin._input_buffers() and the schedule fields
    (CuteDslTileMixin._schedules()). The activation wrappers assign
    ``act_quant_ready`` after every activation."""

    token_width: int
    rows_q: torch.Tensor  # [E_local, m_max, hidden / pack]; also hidden_permuted
    sf_rows: torch.Tensor  # uint8 scale atoms
    # The down GEMM's input. The materialized activation fills these, and so
    # does the static fused-B activation when its column tile holds whole scale
    # groups; otherwise the activation leaves act_quant_ready false and down()
    # quantizes.
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

    def __init__(self, quant_info: MoeLoraMxQuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_blockscaled
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_gemm = prepare_masked_blockscaled
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)


class CuteDslBlockScaledContiguousRowState(ContiguousRowState, kw_only=True):
    """ContiguousRowState plus the buffers of
    _CuteDslBlockScaledMixin._input_buffers() (``token_rs`` from the
    contiguous provider's own) and the schedule fields
    (CuteDslTileMixin._schedules()). The activation wrappers assign
    ``act_quant_ready`` after every activation."""

    token_width: int
    rows_q: torch.Tensor  # [m_pad_ceiling, hidden / pack]; also hidden_compact
    sf_rows: torch.Tensor
    act_q: torch.Tensor  # [m_pad_ceiling, inter / pack]
    act_sf: torch.Tensor
    act_quant_ready: bool = False
    # Per-token NVFP4: the stored fp32 row scales of each quantized buffer.
    rows_rs: torch.Tensor | None = None
    act_rs: torch.Tensor | None = None
    # Per-token NVFP4: the fill's pre-pass stores one scale per token here
    # before fanning it out to the token's rows.
    token_rs: torch.Tensor | None = None
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class _CuteDslBlockScaledContiguousProvider(
    _CuteDslBlockScaledMixin, ContiguousRowDomainProvider
):
    # Segments start on scale-atom boundaries.
    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslBlockScaledContiguousRowState

    def __init__(self, quant_info: MoeLoraMxQuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_contiguous_blockscaled,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_gemm = prepare_contiguous_blockscaled
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """Per token, adds ``token_rs``: one FP32 scale per token, which the
        fill's pre-pass writes before fanning it out to the token's rows."""
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
