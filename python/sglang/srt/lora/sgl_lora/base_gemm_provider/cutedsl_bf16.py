"""BF16 MoE provider backed by the SM100 CuTeDSL masked grouped GEMM.

The study's winning family (plan section 45): swap_ab, direct schedule,
canonical ``[E, N, K]`` weights, 1-CTA MMA. Everything except S2/S4 is
the shared :class:`MaskedRowDomainProvider` — same S1 preprocess (whose
``hidden_permuted``/``masked_m`` are exactly this kernel's A contract), same
S3 activation join, same S5 finalize as the DeepGEMM provider — so a
numerical difference between the two providers can only come from the GEMMs
themselves.

Compile-once discipline: both stage configs are compiled AT CONSTRUCTION
(LoRA-attach time) against the resident weights, and each is warmed with a
zero-tile launch — compile and module load must never happen inside CUDA-graph
capture (the study's keep-alive requirement). The compiled functions keep only
layout STRUCTURE static, so one function serves every per-forward ``m_max``;
A/C/masked_m/schedule are re-wrapped per call.

Dual-ownership rule (the section 45 flagged risk): the tile schedules are
rebuilt every forward, in ``prepare``, from the SAME ``masked_m`` the GEMMs
read. No caller can supply a schedule through another path.

Reachable from serving via ``SGLANG_LORA_MOE_BASE_PROVIDER=cutedsl`` (SM90+;
the device kernel is arch-dispatched in `cutedsl_masked.api`); the default
provider stays DeepGEMM until the gate rulings flip it per device.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.sgl_lora.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowWorkspace,
)
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

# PROCESS-GLOBAL compile cache. cute.compile never caches (by design -- it is
# the documented escape hatch for callers who bring their own cache), so every
# provider instance used to pay its own 6 compiles: measured 0.47 s each,
# 2.82 s per attach, and a 60-layer model paid ~2.8 MINUTES of pure compile at
# every server start (plan section 61.1). One compiled function serves every
# layer because the resident weight is a RUNTIME argument, not a Constexpr --
# only the per-layer wrapper around it differs. This is the same module-level
# dict pattern the other CuTeDSL sites in sglang use (e.g. the TGV GEMM's
# _TGV_CUTE_EXT_COMPILE_CACHE).
#
# The cache holds compiled functions ONLY, never a `b_arg`: retaining a
# wrapper would keep every layer's weight tensor alive for the process.
_COMPILE_CACHE: dict[tuple, object] = {}


class CuteDslStageCall(msgspec.Struct, frozen=True):
    """One stage's shared compiled function plus THIS layer's weight wrapper."""

    compiled_fn: object
    b_arg: object


class CuteDslMaskedWorkspace(MaskedRowWorkspace, kw_only=True):
    """Masked row domain plus this forward's packed tile schedules."""

    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16Provider(MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    # Study winner tiles (section 45): (128, 8)/pc128 for decode-scale rows,
    # (128, 64)/pc152 for large rows-per-expert. The tile choice is a
    # PERFORMANCE regime decision keyed on expected rows per expert — the
    # first boundary run of this provider mistakenly keyed it on the packing
    # ceiling alone and ran prefill on the decode tile at 0.58x.
    NARROW_TOKEN_WIDTH = 8
    NARROW_PERSISTENT_CLUSTERS = 128
    WIDE_TOKEN_WIDTH = 64
    WIDE_PERSISTENT_CLUSTERS = 128
    XWIDE_TOKEN_WIDTH = 128
    XWIDE_PERSISTENT_CLUSTERS = 128
    # The packed direct schedule is representable only at a trivial cluster
    # with 1-CTA MMA (see schedule_builder.build_dual_stage_schedules); these
    # feed BOTH the compiled config and the builder's guard so the two cannot
    # drift apart.
    CLUSTER_SHAPE_MN = (1, 1)
    USE_2CTA_INSTRS = False
    # Per-stage sweeps on GB300 (plan sections 53/55/58): wide wins from
    # m ~= 16. The wide->xwide transition is NOT monotonic (cluster
    # quantization: xwide wins m in [96, 128] and 256, wide wins back ~4% at
    # m=192); threshold 96 dominates the located grid -- it fixes the
    # m in [96, 128) cells (up to 1.26x on gemm2) and selects identically to
    # the old 128 threshold everywhere else. The residual known-loss cell
    # (xwide ~4% behind wide at m=192, both stages) is a Step-7
    # quantization-aware-selection candidate.
    WIDE_EXPECTED_M_THRESHOLD = 16
    XWIDE_EXPECTED_M_THRESHOLD = 96
    OUTPUT_WIDTH = 128

    def __init__(
        self,
        quant_info: SglLoraBf16QuantInfo,
        *,
        force_token_width: int | None = None,
    ):
        """``force_token_width`` is a LAB hook: the provider bench pins each
        compiled tile in turn to site the narrow/wide/xwide crossovers from a
        committed producer (gate-2 review). Production callers (the selector)
        never pass it — the regime policy in `_token_width_for` decides.
        """
        super().__init__(quant_info)
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
            MaskedGroupedGemmConfig,
            as_dynamic_cute_tensor,
            prepare,
        )

        if force_token_width not in (
            None,
            self.NARROW_TOKEN_WIDTH,
            self.WIDE_TOKEN_WIDTH,
            self.XWIDE_TOKEN_WIDTH,
        ):
            raise ValueError(f"no compiled tile has token width {force_token_width}")
        self._force_token_width = force_token_width
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
            build_dual_stage_schedules,
        )

        # The direct-schedule packing holds expert indices in 10 bits. Fail at
        # ATTACH like every other admission decision — the builder's own guard
        # would only fire on the first forward. Remediation paths if a >1024-
        # experts-per-rank geometry ever exists: repack the int32 fields, or
        # compile the kept static scheduler (no packing ABI), or fall back to
        # DeepGEMM (plan section 62).
        if quant_info.num_local_experts > MAX_EXPERTS:
            raise ValueError(
                f"{quant_info.num_local_experts} local experts exceed the "
                f"direct schedule's {MAX_EXPERTS}-expert packing; use the "
                "DeepGEMM provider for this geometry"
            )
        self._build_schedules = build_dual_stage_schedules
        # Token widths where each compiled config remains packable: the
        # narrow tile covers m_max up to 8 * 1024 rows/expert, the wide one to
        # 64 * 1024. The ceiling check in prepare() uses the SAME constant the
        # builder packs with, so guard and packing cannot drift.
        # Each compiled width packs up to width * MAX_TOKEN_CLUSTERS rows per
        # expert; the selector escalates through compiled widths against this.
        self._max_token_clusters = MAX_TOKEN_CLUSTERS

        device = quant_info.w13_weight.device
        experts = quant_info.num_local_experts
        # BF16 WGMMA accepts every N in 8..256 step 8, so the narrow N=8
        # tile IS constructible on SM90 -- but this port has not implemented
        # or validated it (the {64,128,256} gate mirrors the upstream example
        # policy pending that work; plan section 62). Hopper therefore
        # compiles only the wide/xwide pair and decode runs on the 64-token
        # tile -- the recorded headline H200 tuning experiment, worth up to
        # 8x fewer token columns of tensor-core work at decode. The selection
        # floor in `_token_width_for` follows the compiled set.
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, self.XWIDE_PERSISTENT_CLUSTERS),
        )
        if torch.cuda.get_device_capability(device) < (10, 0):
            tile_set = tile_set[1:]
            if force_token_width == self.NARROW_TOKEN_WIDTH:
                raise ValueError(
                    "the narrow (N=8) tile is not implemented/validated by "
                    "this port on SM90; force one of the wide widths"
                )
        self._compiled = {}
        for token_width, persistent_clusters in tile_set:
            if force_token_width is not None and token_width != force_token_width:
                continue
            config = MaskedGroupedGemmConfig(
                mma_tiler_mn=(self.OUTPUT_WIDTH, token_width),
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
                occupancy=1,
                mma_inst_tile_k=4,
                persistent_clusters=persistent_clusters,
                swap_ab=True,
                direct_schedule=True,
            )
            per_stage = {}
            for stage, weight in (
                ("gemm1", quant_info.w13_weight),
                ("gemm2", quant_info.w2_weight),
            ):
                k = weight.shape[2]
                n = weight.shape[1]
                # Geometry is in the key even though the compiled layouts are
                # dynamic: a wrong share is a silently wrong kernel, whereas an
                # over-specific key only costs a compile, and every MoE layer of
                # a model has identical (E, N, K) anyway -- so the 360 -> 6
                # collapse is fully realized either way.
                key = (
                    device.type,
                    device.index,
                    config,
                    experts,
                    n,
                    k,
                    torch.bfloat16,
                )
                compiled_fn = _COMPILE_CACHE.get(key)
                if compiled_fn is None:
                    # Dummy A/C fix only the layout structure; m stays dynamic.
                    dummy_a = torch.zeros(
                        (experts, 256, k), dtype=torch.bfloat16, device=device
                    )
                    dummy_c = torch.empty(
                        (experts, 256, n), dtype=torch.bfloat16, device=device
                    )
                    dummy_masked = torch.zeros(
                        experts, dtype=torch.int32, device=device
                    )
                    prepared = prepare(
                        dummy_a,
                        weight,
                        dummy_c,
                        dummy_masked,
                        config=config,
                    )
                    # Zero-tile warmup: the first launch loads the CUDA module,
                    # and that must happen at attach, never during capture
                    # (study keep-alive requirement). Zero work is a proven-safe
                    # replay state for this kernel. Only the compiling layer
                    # pays it; a cache hit means the module is already loaded.
                    prepared.launch()
                    compiled_fn = prepared.compiled_fn
                    _COMPILE_CACHE[key] = compiled_fn
                per_stage[stage] = CuteDslStageCall(
                    compiled_fn=compiled_fn,
                    b_arg=as_dynamic_cute_tensor(weight, leading_dim=2),
                )
            self._compiled[token_width] = per_stage
        torch.cuda.synchronize(device)

    def _token_width_for(self, m_max: int, expected_m: int) -> int:
        """Performance width first, then escalate through COMPILED widths
        until one can pack ``m_max`` (each width packs up to
        width x MAX_TOKEN_CLUSTERS rows per expert). The first version raised as soon as the performance
        pick could not pack, even when a wider compiled tile could represent
        the same workload (review finding: E=1024/top_k=1/T=65536 yields
        expected_m=64 -> wide, but m_max=65792 needs xwide). Selection never
        leaves the compiled set; only past the WIDEST compiled tile does
        admission fall back to DeepGEMM.
        """
        if self._force_token_width is not None:
            if m_max > self._force_token_width * self._max_token_clusters:
                raise ValueError(
                    f"forced token width {self._force_token_width} cannot pack "
                    f"m_max={m_max}"
                )
            return self._force_token_width
        if expected_m >= self.XWIDE_EXPECTED_M_THRESHOLD:
            performance_width = self.XWIDE_TOKEN_WIDTH
        elif (
            expected_m >= self.WIDE_EXPECTED_M_THRESHOLD
            or self.NARROW_TOKEN_WIDTH not in self._compiled
        ):
            performance_width = self.WIDE_TOKEN_WIDTH
        else:
            performance_width = self.NARROW_TOKEN_WIDTH
        for width in sorted(self._compiled):
            if width >= performance_width and m_max <= width * self._max_token_clusters:
                return width
        widest = max(self._compiled)
        raise ValueError(
            f"m_max={m_max} exceeds the widest compiled tile's schedule "
            f"packing ({widest * self._max_token_clusters}); admission must "
            "fall back to DeepGEMM"
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
    ) -> CuteDslMaskedWorkspace:
        base = super().prepare(hidden_states, topk_ids, top_k)
        token_width = self._token_width_for(base.m_max, base.expected_m)
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            base.masked_m,
            m_max=base.m_max,
            token_width=token_width,
            n_gemm1=2 * self.quant_info.intermediate_size,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        return CuteDslMaskedWorkspace(
            hidden_permuted=base.hidden_permuted,
            masked_m=base.masked_m,
            expected_m=base.expected_m,
            src2dst=base.src2dst,
            m_max=base.m_max,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )

    def _launch(
        self,
        stage: str,
        a: torch.Tensor,
        c: torch.Tensor,
        ws: CuteDslMaskedWorkspace,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack

        call = self._compiled[self._token_width_for(ws.m_max, ws.expected_m)][stage]

        def dyn(tensor: torch.Tensor, leading_dim: int):
            return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=leading_dim
            )

        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        call.compiled_fn(
            dyn(a, 2),
            call.b_arg,
            dyn(c, 2),
            dyn(ws.masked_m, 0),
            dyn(schedule, 0),
            dyn(tiles, 0),
            stream,
        )

    def gateup(self, ws: CuteDslMaskedWorkspace, out: torch.Tensor) -> None:
        self._launch(
            "gemm1", ws.hidden_permuted, out, ws, ws.gemm1_schedule, ws.gemm1_tiles
        )

    def down(
        self, ws: CuteDslMaskedWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        self._launch("gemm2", act_out, out, ws, ws.gemm2_schedule, ws.gemm2_tiles)
