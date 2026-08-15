"""BF16 MoE provider backed by the SM90/SM100 CuTeDSL masked grouped GEMM.

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

Selected explicitly by the evidence-backed serving config on SM90/SM100; the
device kernel is architecture-dispatched in ``cutedsl_masked.api``.

:class:`CuteDslBf16ContiguousProvider` (bottom of this module) is the
route-major twin over the contiguous row domain, SM100-only, reachable via
``the config files=contiguous_cutedsl``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowWorkspace,
    contiguous_m_pad_ceiling,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowWorkspace,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

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

    token_width: int
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

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.api import (
            MaskedGroupedGemmConfig,
            as_dynamic_cute_tensor,
            prepare,
        )

        self._as_dynamic_cute_tensor = as_dynamic_cute_tensor
        self._prepare_masked_gemm = prepare

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
            build_dual_stage_schedules,
            dual_stage_schedule_capacities,
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
        self._schedule_capacities = dual_stage_schedule_capacities
        # Token widths where each compiled config remains packable: the
        # narrow tile covers m_max up to 8 * 1024 rows/expert, the wide one to
        # 64 * 1024. The ceiling check in prepare() uses the SAME constant the
        # builder packs with, so guard and packing cannot drift.
        # Each compiled width packs up to width * MAX_TOKEN_CLUSTERS rows per
        # expert; the selector escalates through compiled widths against this.
        self._max_token_clusters = MAX_TOKEN_CLUSTERS

        device = quant_info.w13_weight.device
        # BF16 WGMMA accepts every N in 8..256 step 8, so the narrow N=8
        # tile IS constructible on SM90 -- but this port has not implemented
        # or validated it (the {64,128,256} gate mirrors the upstream example
        # config pending that work; plan section 62). Hopper therefore
        # compiles only the wide/xwide pair and decode runs on the 64-token
        # tile -- the recorded headline H200 tuning experiment, worth up to
        # 8x fewer token columns of tensor-core work at decode. The selection
        # floor in `_token_width_for` follows the compiled set.
        # The xwide regime has enough rows per expert to fill the physical
        # machine.  On GB300, using all 152 SMs was 10% faster than retaining
        # the study-era 128-cluster cap at T=2048; decode and small-prefill
        # regimes still prefer their narrower 128-cluster tiles.
        device_properties = torch.cuda.get_device_properties(device)
        device_capability = torch.cuda.get_device_capability(device)
        # The all-SM xwide schedule is measured only on the campaign's
        # 152-SM GB300.  Keep Hopper and other geometries on the qualified
        # 128-cluster schedule until they have their own evidence.
        xwide_clusters = self.XWIDE_PERSISTENT_CLUSTERS
        if (
            device_capability >= (10, 0)
            and device_properties.multi_processor_count == 152
        ):
            xwide_clusters = 152
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, xwide_clusters),
        )
        if device_capability < (10, 0):
            tile_set = tile_set[1:]

        from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
            cutedsl_version,
            load_config_table,
        )

        # Swept per-device bucket table; None = the threshold heuristics,
        # byte-identical to a build without any config file.
        version = cutedsl_version()
        self._config_table = load_config_table(
            self.contract.key,
            num_local_experts=quant_info.num_local_experts,
            n_gemm1=self._gate_up_slices * quant_info.intermediate_size,
            n_gemm2=quant_info.hidden_size,
            k=quant_info.hidden_size,
            expected_versions={"cutedsl": version} if version else None,
        )
        if self._config_table is not None:
            for bucket_m, payload in self._config_table.buckets.items():
                if "token_width" not in payload:
                    raise ValueError(
                        f"cutedsl_bf16 config bucket {bucket_m} lacks token_width"
                    )
            # Measured table tiles win over the built-in default for their
            # width; new widths are just extra compile-cache entries, warmed
            # at attach like the rest.
            widths = dict(tile_set)
            widths.update(
                (tile.token_width, tile.persistent_clusters)
                for tile in self._config_table.tiles
            )
            tile_set = tuple(sorted(widths.items()))
        self._compiled: dict[int, dict[tuple[str, bool], CuteDslStageCall]] = {}
        self._tile_configs: dict[int, object] = {}
        for token_width, persistent_clusters in tile_set:
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
            self._tile_configs[token_width] = config
            self._compiled[token_width] = {}
            self._compile_stage(token_width, "gemm1", produce_pdl=False)
            self._compile_stage(token_width, "gemm2", produce_pdl=False)
        torch.cuda.synchronize(device)

    def _compile_stage(
        self,
        token_width: int,
        stage: str,
        *,
        produce_pdl: bool,
    ) -> None:
        """Compile and warm one exact stage/producer role at attach time."""

        call_key = (stage, produce_pdl)
        if call_key in self._compiled[token_width]:
            return
        if stage == "gemm1":
            weight = self.quant_info.w13_weight
        elif stage == "gemm2":
            weight = self.quant_info.w2_weight
        else:
            raise ValueError(f"unknown CuTeDSL base stage {stage!r}")
        config = msgspec.structs.replace(
            self._tile_configs[token_width],
            produce_pdl=produce_pdl,
        )
        device = weight.device
        experts = self.quant_info.num_local_experts
        k = weight.shape[2]
        n = weight.shape[1]
        # Geometry and producer role are both in the key via ``config``. A
        # signal-on binary can therefore never alias the exact PDL-off twin.
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
            dummy_masked = torch.zeros(experts, dtype=torch.int32, device=device)
            prepared = self._prepare_masked_gemm(
                dummy_a,
                weight,
                dummy_c,
                dummy_masked,
                config=config,
            )
            # Zero-tile warmup loads the module before graph capture. Signaling
            # with no dependent launch is legal and performs no extra work.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        self._compiled[token_width][call_key] = CuteDslStageCall(
            compiled_fn=compiled_fn,
            b_arg=self._as_dynamic_cute_tensor(weight, leading_dim=2),
        )

    def configure_base_pdl(self, *, gateup_to_middle: bool) -> None:
        """Compile only the requested producer-on stages before graph capture."""

        requested = (("gemm1", gateup_to_middle),)
        compiled = False
        for token_width in self._compiled:
            for stage, enabled in requested:
                if enabled and (stage, True) not in self._compiled[token_width]:
                    self._compile_stage(token_width, stage, produce_pdl=True)
                    compiled = True
        if compiled:
            torch.cuda.synchronize(self.quant_info.w13_weight.device)

    def base_pdl_state(self) -> dict[str, object]:
        """Observed plan-local producer variants for benchmark provenance."""

        return {
            "provider": self.contract.key,
            "producer_signal_supported": True,
            "gateup_signal_compiled": all(
                ("gemm1", True) in stages for stages in self._compiled.values()
            ),
            "down_signal_compiled": all(
                ("gemm2", True) in stages for stages in self._compiled.values()
            ),
        }

    def _token_width_for(self, m_max: int, expected_m: int) -> int:
        """Performance width first, then escalate through COMPILED widths
        until one can pack ``m_max`` (each width packs up to
        width x MAX_TOKEN_CLUSTERS rows per expert). The first version raised as soon as the performance
        pick could not pack, even when a wider compiled tile could represent
        the same workload (review finding: E=1024/top_k=1/T=65536 yields
        expected_m=64 -> wide, but m_max=65792 needs xwide). Selection never
        leaves the compiled set; only past the WIDEST compiled tile does
        admission fall back to DeepGEMM.

        A swept bucket table replaces only the performance pick; the
        packability escalation is a correctness constraint and always clamps
        the table's choice to compiled widths that can pack ``m_max``.
        """
        if self._config_table is not None:
            performance_width = self._config_table.pick(expected_m)["token_width"]
        elif expected_m >= self.XWIDE_EXPECTED_M_THRESHOLD:
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
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslMaskedWorkspace:
        base = super().prepare(hidden_states, topk_ids, top_k, workspace)
        token_width = self._token_width_for(base.m_max, base.expected_m)
        schedule_outputs = {}
        if workspace is not None:
            capacity1, capacity2 = self._schedule_capacities(
                num_experts=base.masked_m.numel(),
                m_max=base.m_max,
                token_width=token_width,
                n_gemm1=self.gate_up_slices * self.quant_info.intermediate_size,
                n_gemm2=self.quant_info.hidden_size,
                output_width=self.OUTPUT_WIDTH,
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
            )
            prefix = f"base:cutedsl:tw{token_width}"
            schedule_outputs = {
                "schedule1_out": workspace.tensor(
                    f"{prefix}:gemm1_schedule",
                    (capacity1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "tiles1_out": workspace.tensor(
                    f"{prefix}:gemm1_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "schedule2_out": workspace.tensor(
                    f"{prefix}:gemm2_schedule",
                    (capacity2,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "tiles2_out": workspace.tensor(
                    f"{prefix}:gemm2_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
            }
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            base.masked_m,
            m_max=base.m_max,
            token_width=token_width,
            n_gemm1=self.gate_up_slices * self.quant_info.intermediate_size,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
            **schedule_outputs,
        )
        return CuteDslMaskedWorkspace(
            hidden_permuted=base.hidden_permuted,
            masked_m=base.masked_m,
            expected_m=base.expected_m,
            src2dst=base.src2dst,
            m_max=base.m_max,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
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
        *,
        produce_pdl: bool,
    ) -> None:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack

        try:
            call = self._compiled[ws.token_width][(stage, produce_pdl)]
        except KeyError as exc:
            raise RuntimeError(
                f"CuTeDSL {stage} token width {ws.token_width}, producer "
                f"PDL={produce_pdl} was not compiled at plan attach"
            ) from exc

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

    def gateup(
        self,
        ws: CuteDslMaskedWorkspace,
        out: torch.Tensor,
        *,
        produce_pdl: bool = False,
    ) -> None:
        self._launch(
            "gemm1",
            ws.hidden_permuted,
            out,
            ws,
            ws.gemm1_schedule,
            ws.gemm1_tiles,
            produce_pdl=produce_pdl,
        )

    def down(
        self,
        ws: CuteDslMaskedWorkspace,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        # No consumer takes a PDL edge off GEMM2, so only the
        # producer-off stage is ever compiled (see configure_base_pdl).
        self._launch(
            "gemm2",
            act_out,
            out,
            ws,
            ws.gemm2_schedule,
            ws.gemm2_tiles,
            produce_pdl=False,
        )


class CuteDslContiguousWorkspace(ContiguousRowWorkspace, kw_only=True):
    """Contiguous row domain plus this forward's packed tile schedules.

    Dual-ownership carries over from the masked twin: both schedules derive
    from the SAME device ``seg_counts`` the dispatch wrote, in one launch —
    the S1 seg-layout launch itself, which packs them through a
    ``ContiguousSchedulePack`` instead of a standalone builder launch; the
    GEMMs read the matching ``seg_offsets`` for their segment-base fold.
    """

    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16ContiguousProvider(ContiguousRowDomainProvider):
    """Route-major twin of :class:`CuteDslBf16Provider` (SM100-only).

    Same resident ``[E, N, K]`` BF16 weights, same swap_ab + direct-schedule
    winner family, same compile-once/warm-at-attach discipline; only the row
    domain changes.  The routed activation and every GEMM output live in ONE
    flat aligned buffer (O(T * top_k) rows — the DeepGEMM-contiguous memory
    math), the packed schedules keep per-expert LOCAL token clusters built
    from ``seg_counts`` — emitted by the S1 seg-layout launch itself through
    a ``ContiguousSchedulePack``, not a standalone builder launch — and the
    kernel folds ``seg_offsets[e]`` into the flat tile index on device.
    Every compiled token width must divide the
    segment alignment so a partial tile's overrun stays inside its expert's
    own aligned segment (the contiguous twin of masked slab padding);
    ``_compile_stage`` validates that per width at attach.

    SM100-only: the SM90 WGMMA sibling kernel has no contiguous port.  The
    selector converts tuned-domain GB300 prefill choices here by default;
    the 'contiguous_cutedsl' env value additionally forces gb300
    out-of-domain prefill keys here (untuned-geometry A/B), guarded by the
    1024-expert packing cap below so every menu choice stays attachable.
    The base GEMMs are factor-layout-agnostic — resident
    weights are identical either way — so the shared-outer plan differs from
    the per-expert one only in the domain-level glue and LoRA kernels, all
    inherited from :class:`ContiguousRowDomainProvider`.  No producer-PDL
    twins are compiled — the eligible plan families exclude base-GEMM PDL
    edges by construction (the shared plan's ``route_pdl`` chain lives
    inside the routing launches, not the base GEMMs).
    """

    contract = MoeBaseProviderContract(
        key="cutedsl_contiguous",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    # The segment alignment of the flat row buffer.  Pinned at 128: it keeps
    # byte-parity with DeepGEMM's contiguous m-alignment (identical
    # capture-memory math) and every study-winner token width (8/64/128)
    # divides it.  A decode-oriented narrower alignment was tried and
    # measured slower end-to-end than the masked decode path, so no
    # per-choice override exists.
    M_ALIGNMENT = 128

    # One source of truth for the study-winner tile config: identical
    # constants and the identical width-selection method as the masked
    # provider, so a tuning change there cannot silently diverge here.
    NARROW_TOKEN_WIDTH = CuteDslBf16Provider.NARROW_TOKEN_WIDTH
    NARROW_PERSISTENT_CLUSTERS = CuteDslBf16Provider.NARROW_PERSISTENT_CLUSTERS
    WIDE_TOKEN_WIDTH = CuteDslBf16Provider.WIDE_TOKEN_WIDTH
    WIDE_PERSISTENT_CLUSTERS = CuteDslBf16Provider.WIDE_PERSISTENT_CLUSTERS
    XWIDE_TOKEN_WIDTH = CuteDslBf16Provider.XWIDE_TOKEN_WIDTH
    XWIDE_PERSISTENT_CLUSTERS = CuteDslBf16Provider.XWIDE_PERSISTENT_CLUSTERS
    CLUSTER_SHAPE_MN = CuteDslBf16Provider.CLUSTER_SHAPE_MN
    USE_2CTA_INSTRS = CuteDslBf16Provider.USE_2CTA_INSTRS
    WIDE_EXPECTED_M_THRESHOLD = CuteDslBf16Provider.WIDE_EXPECTED_M_THRESHOLD
    XWIDE_EXPECTED_M_THRESHOLD = CuteDslBf16Provider.XWIDE_EXPECTED_M_THRESHOLD
    OUTPUT_WIDTH = CuteDslBf16Provider.OUTPUT_WIDTH

    _token_width_for = CuteDslBf16Provider._token_width_for

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        device = quant_info.w13_weight.device
        if torch.cuda.get_device_capability(device) < (10, 0):
            raise NotImplementedError(
                "cutedsl_contiguous requires SM100+ (the SM90 kernel has no "
                "contiguous port); use 'cutedsl' or 'deepgemm_contiguous' on "
                f"sm{torch.cuda.get_device_capability(device)}"
            )
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.api import (
            MaskedGroupedGemmConfig,
            as_dynamic_cute_tensor,
            prepare_contiguous,
        )

        self._as_dynamic_cute_tensor = as_dynamic_cute_tensor
        self._prepare_contiguous_gemm = prepare_contiguous

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
            contiguous_dual_stage_schedule_capacities,
            contiguous_dual_stage_schedule_pack,
            validate_contiguous_tile_geometry,
        )

        self._validate_tile_geometry = validate_contiguous_tile_geometry

        if quant_info.num_local_experts > MAX_EXPERTS:
            raise ValueError(
                f"{quant_info.num_local_experts} local experts exceed the "
                f"direct schedule's {MAX_EXPERTS}-expert packing; use the "
                "DeepGEMM provider for this geometry"
            )
        self._schedule_pack = contiguous_dual_stage_schedule_pack
        self._schedule_capacities = contiguous_dual_stage_schedule_capacities
        self._max_token_clusters = MAX_TOKEN_CLUSTERS

        # Same physical-machine rule as the masked provider: the all-SM xwide
        # schedule is qualified only on the campaign's 152-SM GB300.
        device_properties = torch.cuda.get_device_properties(device)
        xwide_clusters = self.XWIDE_PERSISTENT_CLUSTERS
        if device_properties.multi_processor_count == 152:
            xwide_clusters = 152
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, xwide_clusters),
        )

        from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
            cutedsl_version,
            load_config_table,
        )

        version = cutedsl_version()
        self._config_table = load_config_table(
            self.contract.key,
            num_local_experts=quant_info.num_local_experts,
            n_gemm1=self._gate_up_slices * quant_info.intermediate_size,
            n_gemm2=quant_info.hidden_size,
            k=quant_info.hidden_size,
            expected_versions={"cutedsl": version} if version else None,
        )
        if self._config_table is not None:
            for bucket_m, payload in self._config_table.buckets.items():
                if "token_width" not in payload:
                    raise ValueError(
                        f"cutedsl_contiguous config bucket {bucket_m} lacks "
                        "token_width"
                    )
            # Measured table tiles win over the built-in default for their
            # width; new widths are just extra compile-cache entries, warmed
            # at attach like the rest (and containment-checked there).
            widths = dict(tile_set)
            widths.update(
                (tile.token_width, tile.persistent_clusters)
                for tile in self._config_table.tiles
            )
            tile_set = tuple(sorted(widths.items()))

        self._compiled: dict[int, dict[str, CuteDslStageCall]] = {}
        self._tile_configs: dict[int, object] = {}
        for token_width, persistent_clusters in tile_set:
            # Segment-containment gate: a width that does not divide the
            # alignment could read/store across expert segments (the exact
            # hazard the masked slab padding absorbs), so it is rejected at
            # attach — including table-supplied widths.
            self._validate_tile_geometry(token_width, self._m_alignment)
            config = MaskedGroupedGemmConfig(
                mma_tiler_mn=(self.OUTPUT_WIDTH, token_width),
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
                occupancy=1,
                mma_inst_tile_k=4,
                persistent_clusters=persistent_clusters,
                swap_ab=True,
                direct_schedule=True,
                contiguous_segments=True,
            )
            self._tile_configs[token_width] = config
            self._compiled[token_width] = {}
            self._compile_stage(token_width, "gemm1")
            self._compile_stage(token_width, "gemm2")
        torch.cuda.synchronize(device)

    def _stage_weight(self, stage: str) -> torch.Tensor:
        if stage == "gemm1":
            return self.quant_info.w13_weight
        if stage == "gemm2":
            return self.quant_info.w2_weight
        raise ValueError(f"unknown CuTeDSL base stage {stage!r}")

    def _compile_stage(self, token_width: int, stage: str) -> None:
        """Compile and warm one exact stage geometry at attach time."""

        if stage in self._compiled[token_width]:
            return
        weight = self._stage_weight(stage)
        config = self._tile_configs[token_width]
        device = weight.device
        num_experts = self.quant_info.num_local_experts
        k = weight.shape[2]
        n = weight.shape[1]
        # ``contiguous_segments`` lives in the config, so an entry here can
        # never alias the masked provider's compile for the same geometry.
        key = (
            device.type,
            device.index,
            config,
            num_experts,
            n,
            k,
            torch.bfloat16,
        )
        compiled_fn = _COMPILE_CACHE.get(key)
        if compiled_fn is None:
            # Flat dummies fix only the layout structure (rows stay dynamic);
            # 256 rows keep the alignment-multiple contract.
            dummy_a = torch.zeros((256, k), dtype=torch.bfloat16, device=device)
            dummy_c = torch.empty((256, n), dtype=torch.bfloat16, device=device)
            dummy_seg = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
            prepared = self._prepare_contiguous_gemm(
                dummy_a,
                weight,
                dummy_c,
                dummy_seg,
                config=config,
                m_alignment=self._m_alignment,
            )
            # Zero-tile warmup loads the module before any graph capture.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        self._compiled[token_width][stage] = CuteDslStageCall(
            compiled_fn=compiled_fn,
            b_arg=self._as_dynamic_cute_tensor(weight, leading_dim=2),
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslContiguousWorkspace:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        # The pack geometry precedes the dispatch because the S1 seg-layout
        # launch emits the schedules itself; every input below is host-static
        # (the same values the domain's prepare re-derives), so pack and
        # dispatch cannot disagree on the row ceiling.
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        # A single expert receives at most one pair per token (a token's
        # top-k experts are distinct), the same host-static bound the masked
        # slab relies on; it drives packability escalation and the schedule
        # ABI check without any device readback.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        expected_m = (num_pairs - 1) // num_experts + 1 if num_pairs else 1
        token_width = self._token_width_for(max_expert_rows, expected_m)
        n_gemm1 = self.gate_up_slices * self.quant_info.intermediate_size
        n_gemm2 = self.quant_info.hidden_size
        schedule_outputs = {}
        if workspace is not None:
            capacity1, capacity2 = self._schedule_capacities(
                num_experts=num_experts,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max_expert_rows,
                m_alignment=self._m_alignment,
                token_width=token_width,
                n_gemm1=n_gemm1,
                n_gemm2=n_gemm2,
                output_width=self.OUTPUT_WIDTH,
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
            )
            # The alignment tag keeps a differently aligned sibling
            # instance's schedule buffers distinct: capacities derive from
            # alignment-dependent row ceilings, and instances may share one
            # layer workspace.
            prefix = f"base:cutedsl_contig:a{self._m_alignment}:tw{token_width}"
            schedule_outputs = {
                "schedule1_out": workspace.tensor(
                    f"{prefix}:gemm1_schedule",
                    (capacity1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "tiles1_out": workspace.tensor(
                    f"{prefix}:gemm1_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "schedule2_out": workspace.tensor(
                    f"{prefix}:gemm2_schedule",
                    (capacity2,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "tiles2_out": workspace.tensor(
                    f"{prefix}:gemm2_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
            }
        pack = self._schedule_pack(
            num_experts=num_experts,
            m_pad_ceiling=m_pad_ceiling,
            max_expert_rows=max_expert_rows,
            m_alignment=self._m_alignment,
            token_width=token_width,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            output_width=self.OUTPUT_WIDTH,
            device=hidden_states.device,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
            **schedule_outputs,
        )
        base = super().prepare(
            hidden_states, topk_ids, top_k, workspace, schedule_pack=pack
        )
        return CuteDslContiguousWorkspace(
            hidden_compact=base.hidden_compact,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            src2dst=base.src2dst,
            grouped_layout=base.grouped_layout,
            m_pad_ceiling=base.m_pad_ceiling,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
            gemm1_schedule=pack.schedule1,
            gemm1_tiles=pack.tiles1,
            gemm2_schedule=pack.schedule2,
            gemm2_tiles=pack.tiles2,
        )

    @staticmethod
    def _invoke(
        call: CuteDslStageCall,
        a: torch.Tensor,
        c: torch.Tensor,
        seg_offsets: torch.Tensor,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack

        def dyn(tensor: torch.Tensor, leading_dim: int):
            return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=leading_dim
            )

        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        # The unit L mode re-wraps per call exactly the way the compile path
        # wrapped its dummies; seg_offsets rides the masked_m argument slot.
        call.compiled_fn(
            dyn(a.unsqueeze(0), 2),
            call.b_arg,
            dyn(c.unsqueeze(0), 2),
            dyn(seg_offsets, 0),
            dyn(schedule, 0),
            dyn(tiles, 0),
            stream,
        )

    def _launch(
        self,
        stage: str,
        a: torch.Tensor,
        c: torch.Tensor,
        ws: CuteDslContiguousWorkspace,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        try:
            call = self._compiled[ws.token_width][stage]
        except KeyError as exc:
            raise RuntimeError(
                f"CuTeDSL contiguous {stage} token width {ws.token_width} "
                "was not compiled at plan attach"
            ) from exc
        self._invoke(call, a, c, ws.seg_offsets, schedule, tiles)

    def gateup(
        self,
        ws: CuteDslContiguousWorkspace,
        out: torch.Tensor,
        *,
        produce_pdl: bool = False,
    ) -> None:
        if produce_pdl:
            raise NotImplementedError(
                "cutedsl_contiguous compiles no base-GEMM producer twins: "
                "the eligible plan family excludes base-GEMM PDL edges"
            )
        self._launch(
            "gemm1", ws.hidden_compact, out, ws, ws.gemm1_schedule, ws.gemm1_tiles
        )

    def down(
        self,
        ws: CuteDslContiguousWorkspace,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._launch("gemm2", act_out, out, ws, ws.gemm2_schedule, ws.gemm2_tiles)
