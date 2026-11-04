"""CuTeDSL GEMMs on block-scaled checkpoints: MXFP8 (E4M3 x E8M0 per 32),
MXFP4 (E2M1 pairs x E8M0 per 32) and NVFP4 (E2M1 pairs x E4M3 per 16 under
per-expert global scales).

Activations are quantized per row into the weights' format with the scales
packed in the tcgen05 atom order; GEMM outputs remain BF16 for LoRA.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.lora.moe.activation_delta import act_delta
from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_blockscaled,
)
from sglang.kernels.ops.lora.moe.dispatch_masked import dispatch_fill_masked_blockscaled
from sglang.kernels.ops.lora.moe.fused_act import fused_b_act
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    expected_rows_per_expert,
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
    masked_m_max,
)
from sglang.srt.lora.moe.quant_info import (
    MoeLoraMxfp4QuantInfo,
    MoeLoraMxfp8QuantInfo,
    MoeLoraMxQuantInfo,
    MoeLoraNvFp4QuantInfo,
)


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

    def _bind_weights(self, quant_info: MoeLoraMxQuantInfo) -> None:
        from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
        from sglang.kernels.ops.lora.moe.quant_blockscaled import (
            BlockScaledRows,
            pack_scale_atoms,
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
        self._sf_w13 = pack_scale_atoms(quant_info.w13_scale)
        self._sf_w2 = pack_scale_atoms(quant_info.w2_scale)
        # NVFP4 fills these with per-expert global scales and activation scales.
        self._alpha = {"gemm1": None, "gemm2": None}
        self._alpha_rows_per_half = {"gemm1": 1, "gemm2": 1}
        self._act_scale = {"gemm1": None, "gemm2": None}

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
        )

    def _quant_buffers(self, workspace, prefix, rows_shape, k, device):
        """Quantized rows plus their scale atoms; ``rows_shape`` is
        ``(slabs, rows)`` for expert slabs or ``(rows,)`` for flat rows."""
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
        return data, scales

    def _rows_spec(self, stage, data, scales, rows_per_slab):
        """The quantized rows of one GEMM input, as the kernels address them."""
        return self._rows_cls(
            data=data.reshape(-1, data.shape[-1]),
            scales=scales,
            sf_vec_size=self.SF_VEC_SIZE,
            rows_per_slab=rows_per_slab,
            global_scale=self._act_scale[stage],
        )

    def _act_quant_for(self, row_state, config, rows_per_slab):
        """The down input the fused activation can write, if its column tile
        holds whole scale groups; otherwise down() quantizes separately."""
        if int(config["BLOCK_SIZE_W"]) % self.SF_VEC_SIZE:
            return None
        return self._rows_spec(
            "gemm2", row_state.act_q, row_state.act_sf, rows_per_slab
        )

    def _quantize(self, stage, rows, data, scales, pair_to_row):
        self._quantize_rows(
            rows.reshape(-1, rows.shape[-1]),
            self._rows_spec(stage, data, scales, rows.shape[-2]),
            pair_to_row,
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
    ) -> None:
        call = self._compiled[row_state.token_width][stage]
        # Runtime wrapping must match the compiled argument's MLIR type.
        dyn = self._as_dynamic_cute_tensor
        call.compiled_fn(
            dyn(self._rows_arg(a_q), leading_dim=2),
            call.b_arg,
            dyn(sf_tokens, leading_dim=0),
            dyn(call.sf_weights, leading_dim=0),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._alpha_arg(stage), leading_dim=1),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_q.device),
        )

    def _alpha_arg(self, stage: str) -> torch.Tensor:
        alpha = self._alpha[stage]
        if alpha is None:
            alpha = self._alpha[stage] = torch.ones(
                (self.quant_info.num_local_experts, 1),
                dtype=torch.float32,
                device=self._w13.device,
            )
        return alpha

    def _stage_kwargs(self, stage: str) -> dict:
        return dict(
            sf_vec_size=self.SF_VEC_SIZE,
            alpha=self._alpha[stage],
            alpha_rows_per_half=self._alpha_rows_per_half[stage],
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
        )

    def down(self, row_state, act_out: torch.Tensor, out: torch.Tensor) -> None:
        if not row_state.act_quant_ready:
            # Fused-B activation bypasses act_with_delta: quantize separately.
            self._quantize(
                "gemm2",
                act_out,
                row_state.act_q,
                row_state.act_sf,
                row_state.pair_to_row,
            )
        self._launch(
            "gemm2",
            row_state.act_q,
            row_state.act_sf,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
        )


class _CuteDslNvFp4Mixin(_CuteDslBlockScaledMixin):
    SF_VEC_SIZE = 16
    derived_base_weights = "packed block-scale atoms and global-scale products"

    def _bind_weights(self, quant_info: MoeLoraNvFp4QuantInfo) -> None:
        super()._bind_weights(quant_info)
        # The epilogue applies activation x weight global scales per expert;
        # gate and up rows of W13 may carry separate weight scales.
        w13_global = quant_info.w13_global_scale.reshape(
            quant_info.num_local_experts, -1
        )
        self._alpha["gemm1"] = (w13_global * quant_info.w13_input_scale).contiguous()
        self._alpha_rows_per_half["gemm1"] = (
            quant_info.intermediate_size if w13_global.shape[1] == 2 else 1 << 30
        )
        self._alpha["gemm2"] = (
            quant_info.w2_global_scale.reshape(-1, 1) * quant_info.w2_input_scale
        ).contiguous()
        self._alpha_rows_per_half["gemm2"] = 1 << 30
        self._act_scale = {
            "gemm1": quant_info.w13_input_scale,
            "gemm2": quant_info.w2_input_scale,
        }


class CuteDslBlockScaledMaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    rows_q: torch.Tensor  # [E_local, m_max, hidden / pack]; also hidden_permuted
    sf_rows: torch.Tensor  # uint8 scale atoms
    # Materialized activation fills these; fused-B activation quantizes in down.
    act_q: torch.Tensor  # [E_local, m_max, inter / pack]
    act_sf: torch.Tensor
    act_quant_ready: bool = False
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class _CuteDslBlockScaledMaskedProvider(
    _CuteDslBlockScaledMixin, MaskedRowDomainProvider
):
    _ROW_MODE = "masked"

    def __init__(self, quant_info: MoeLoraMxQuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_blockscaled
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_masked_gemm = prepare_masked_blockscaled
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        experts, n, k_store = weight.shape
        dummy_a = torch.zeros(
            (experts, 256, k_store), dtype=weight.dtype, device=device
        )
        dummy_sf = torch.zeros(
            self._scale_atom_bytes(
                256, k_store * self._pack, self.SF_VEC_SIZE, experts
            ),
            dtype=torch.uint8,
            device=device,
        )
        dummy_c = torch.empty((experts, 256, n), dtype=torch.bfloat16, device=device)
        dummy_masked = torch.zeros(experts, dtype=torch.int32, device=device)
        return self._prepare_masked_gemm(
            dummy_a,
            weight,
            dummy_sf,
            weight_sf,
            dummy_c,
            dummy_masked,
            config=config,
            **self._stage_kwargs(stage),
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _group_arg(row_state: CuteDslBlockScaledMaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslBlockScaledMaskedRowState:
        experts = self.quant_info.num_local_experts
        k = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        expected_m = expected_rows_per_expert(topk_ids.numel(), experts)
        token_width = self._token_width_for(hidden_states.size(0), expected_m)
        m_max = masked_m_max(hidden_states.size(0), token_width)
        device = hidden_states.device

        def _buf(name, shape, dtype):
            return prepare_buffer(workspace, name, shape, dtype=dtype, device=device)

        masked_m = _buf("masked:masked_m", (experts,), torch.int32)
        pair_to_row = _buf("masked:pair_to_row", (topk_ids.numel(),), torch.int32)
        rows_q, sf_rows = self._quant_buffers(
            workspace, "masked:rows", (experts, m_max), k, device
        )
        act_q, act_sf = self._quant_buffers(
            workspace, "masked:act", (experts, m_max), inter, device
        )
        geometry = dict(
            token_width=token_width,
            n_gemm1=self.gate_up_slices * inter,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        capacities = self._schedule_capacities(
            num_experts=experts, m_max=m_max, **geometry
        )
        dispatch_fill_masked_blockscaled(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=self._rows_spec("gemm1", rows_q, sf_rows, m_max),
        )
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace, f"cutedsl_masked:tw{token_width}", capacities, device
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            masked_m, m_max=m_max, **geometry, **schedule_outputs
        )
        return CuteDslBlockScaledMaskedRowState(
            hidden_permuted=rows_q,
            masked_m=masked_m,
            expected_m=expected_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            input_buffer_reuse=workspace is not None,
            token_width=token_width,
            rows_q=rows_q,
            sf_rows=sf_rows,
            act_q=act_q,
            act_sf=act_sf,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )

    def act_with_delta(
        self,
        row_state: CuteDslBlockScaledMaskedRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
    ) -> None:
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            act_quant=self._rows_spec(
                "gemm2", row_state.act_q, row_state.act_sf, row_state.m_max
            ),
        )
        row_state.act_quant_ready = True

    def fused_act(
        self,
        row_state: CuteDslBlockScaledMaskedRowState,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_rows: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing,
        config,
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
    ) -> None:
        act_quant = self._act_quant_for(row_state, config, row_state.m_max)
        fused_b_act(
            activation=activation,
            base_gateup=base_gateup,
            act_rows=act_rows,
            act_pairs=act_pairs,
            pair_to_row=row_state.pair_to_row,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            act_quant=act_quant,
        )
        row_state.act_quant_ready = act_quant is not None


class CuteDslBlockScaledContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    rows_q: torch.Tensor  # [m_pad_ceiling, hidden / pack]; also hidden_compact
    sf_rows: torch.Tensor
    act_q: torch.Tensor  # [m_pad_ceiling, inter / pack]
    act_sf: torch.Tensor
    act_quant_ready: bool = False
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

        self._prepare_contiguous_gemm = prepare_contiguous_blockscaled
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules_contiguous = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        num_experts, n, k_store = weight.shape
        dummy_a = torch.zeros((256, k_store), dtype=weight.dtype, device=device)
        dummy_sf = torch.zeros(
            self._scale_atom_bytes(256, k_store * self._pack, self.SF_VEC_SIZE),
            dtype=torch.uint8,
            device=device,
        )
        dummy_c = torch.empty((256, n), dtype=torch.bfloat16, device=device)
        dummy_seg = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        return self._prepare_contiguous_gemm(
            dummy_a,
            weight,
            dummy_sf,
            weight_sf,
            dummy_c,
            dummy_seg,
            config=config,
            **self._stage_kwargs(stage),
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _group_arg(row_state: CuteDslBlockScaledContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslBlockScaledContiguousRowState:
        base = super().prepare(
            hidden_states, topk_ids, top_k, workspace, fill_rows=False
        )
        num_experts = self.quant_info.num_local_experts
        hidden = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        device = hidden_states.device
        prefix = f"contiguous:a{self._m_alignment}"
        rows_q, sf_rows = self._quant_buffers(
            workspace, f"{prefix}:rows", (base.m_pad_ceiling,), hidden, device
        )
        act_q, act_sf = self._quant_buffers(
            workspace, f"{prefix}:act", (base.m_pad_ceiling,), inter, device
        )
        dispatch_fill_rows_contiguous_blockscaled(
            hidden_states,
            topk_ids,
            base.pair_to_row,
            rows_out=self._rows_spec("gemm1", rows_q, sf_rows, base.m_pad_ceiling),
        )
        # Distinct top-k experts bound each expert to one row per token.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        expected_m = expected_rows_per_expert(topk_ids.numel(), num_experts)
        token_width = self._token_width_for(max_expert_rows, expected_m)
        geometry = dict(
            m_pad_ceiling=base.m_pad_ceiling,
            max_expert_rows=max_expert_rows,
            m_alignment=self._m_alignment,
            token_width=token_width,
            n_gemm1=self.gate_up_slices * inter,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace,
                f"cutedsl_contiguous:a{self._m_alignment}:tw{token_width}",
                self._schedule_capacities(num_experts=num_experts, **geometry),
                device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules_contiguous(
            base.seg_counts, **geometry, **schedule_outputs
        )
        return CuteDslBlockScaledContiguousRowState(
            hidden_compact=rows_q,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            pair_to_row=base.pair_to_row,
            m_pad_ceiling=base.m_pad_ceiling,
            input_buffer_reuse=base.input_buffer_reuse,
            token_width=token_width,
            rows_q=rows_q,
            sf_rows=sf_rows,
            act_q=act_q,
            act_sf=act_sf,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )

    def act_with_delta(
        self,
        row_state: CuteDslBlockScaledContiguousRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        act_quant=None,
    ) -> None:
        act_delta(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            act_quant=self._rows_spec(
                "gemm2", row_state.act_q, row_state.act_sf, row_state.m_pad_ceiling
            ),
        )
        row_state.act_quant_ready = True

    def fused_act(
        self,
        row_state: CuteDslBlockScaledContiguousRowState,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_rows: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing,
        config,
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
    ) -> None:
        act_quant = self._act_quant_for(row_state, config, row_state.m_pad_ceiling)
        fused_b_act(
            activation=activation,
            base_gateup=base_gateup,
            act_rows=act_rows,
            act_pairs=act_pairs,
            pair_to_row=row_state.pair_to_row,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            act_quant=act_quant,
        )
        row_state.act_quant_ready = act_quant is not None


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
