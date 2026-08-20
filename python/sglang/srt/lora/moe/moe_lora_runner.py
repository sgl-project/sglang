"""MoE-LoRA runner for the MoE LoRA execution engine.

``MoeLoraRunner`` holds one MoE layer's LoRA execution state. The layer
wrapper holds a :class:`MoeLoraLayerEngine`; that engine creates the runner at
weight bind and keeps it for the layer's life. Construction admits every
resident base provider, keyed by name, plus the layer geometry they share;
``run`` selects the provider its plan names and executes the pipeline. No
stock ``MoeRunner`` is involved — the per-quant base stages live behind
:class:`MoeBaseProvider`, and this class owns the LoRA route views, the LoRA
kernels, and every pipeline buffer.

Every forward runs a typed ``MoeLoraExecutionPlan`` supplied by the caller —
:class:`MoeLoraLayerEngine` resolves one per phase from the plan tables at
weight bind, and the serial correctness pipeline ships there as the
``fallback.serial`` rows.  Whichever plan arrives, every consumed stage has
exactly one owner and every required route representation is built once:

    gate/up LoRA A  (grouped_lora_a: token-major hidden -> pair-major rank)
    gate/up LoRA B  (one_launch_sliced_lora_b -> standard [gate | up] delta)
    S1 prepare      (provider permute to its physical row domain)
    S2 gateup       (provider grouped GEMM)
    S3 act          (base + delta -> activation; writes provider rows and,
                     when required, a pair-major down-A source)
    down LoRA A     (grouped_lora_a, original pairs or provider-mapped rows)
    down LoRA B     (one_launch_sliced_lora_b -> unweighted LoRA delta [T, K, H])
    S4 down         (provider grouped GEMM)
    S5 finalize     (provider fixed-order top-k reduction; router coefficient
                     and routed scaling applied EXACTLY ONCE over
                     base + LoRA delta, at the provider-declared coefficient
                     precision)

Every batch runs this one LoRA-capable topology — base-only, mixed, and active
alike — so they share a single graph shape. Inactive assignments ride sentinel
routes and contribute exact zeros rather than being diverted to another path.

Base rows: serving gives the base model a REAL resident slot whose factors are
zero-filled and whose ``adapter_enabled`` entry is 0. Batch preparation
normalizes such assignments to the ``-1`` execution sentinel before any
layer runs — otherwise base rows build routed work against zero weights, which
is numerically harmless but inflates route padding, group counts, and every
LoRA GEMM's row count.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch

logger = logging.getLogger(__name__)

from sglang.srt.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MappedLoraAInput,
    MoeBaseProvider,
)
from sglang.srt.lora.moe.execution_plan import (
    ActFamily,
    BridgeLayout,
    DownOverlap,
    FinalizeFamily,
    GateUpOverlap,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MoeLoraExecutionPlan,
    Phase,
    SelectedPlan,
    Site,
    architecture_for_capability,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import (
    MoeLoraLaunchConfig,
    TileTable,
    resolve_tiles,
)
from sglang.srt.lora.moe.lora_a import run_lora_a
from sglang.srt.lora.moe.lora_b import run_lora_b
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.lora.moe.route_view import RouteView
from sglang.srt.lora.moe.routing import MoeLoraRoutes, build_routes
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace
from sglang.srt.runtime_context import get_lora

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass(slots=True)
class _LoraStageState:
    """One stage's LoRA side-branch results: A's rank, B's delta.

    A mutable box because ``run_parallel`` returns only the COMPUTE closure's
    value, and a side closure cannot rebind its enclosing local. ``None`` also
    means "this plan has no such stage", which callers check.
    """

    rank: torch.Tensor | None = None
    delta: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _DownAInput:
    """Standalone down-A source without exposing provider workspace details."""

    rows: torch.Tensor
    pair_to_row: torch.Tensor | None = None


class MoeLoraBatch(msgspec.Struct, kw_only=True):
    """The per-batch state the MoE-LoRA runner actually consumes.

    Narrow by design: the legacy ``LoRAInfo`` carries ~18 fields for the old
    kernels, and passing it wholesale would make it impossible to see what this
    runner depends on. ``token_lora_mapping`` holds active physical slot
    IDs, with every inactive assignment represented by the ``-1`` sentinel.
    """

    gate_up_lora_a: torch.Tensor  # [L_cap, E_f, slices*R_phys, H]
    gate_up_lora_b: torch.Tensor  # [L_cap, E_local, slices*I, R_phys]
    down_lora_a: torch.Tensor  # [L_cap, E_local, R_phys, I]
    down_lora_b: torch.Tensor  # [L_cap, E_f_down, H, R_phys]
    token_lora_mapping: torch.Tensor  # [T] int, adapter slot per token (-1 = base)
    adapter_enabled: torch.Tensor | None  # [L_cap], 0 marks an inactive slot
    use_cuda_graph: bool = False
    is_prefill: bool = False
    has_active_lora: bool = True

    @property
    def max_loras(self) -> int:
        return self.gate_up_lora_a.shape[0]


class MoeLoraRunner:
    """One MoE layer's MoE LoRA execution state and pipeline."""

    def __init__(
        self,
        *,
        providers: Mapping[str, MoeBaseProvider],
        top_k: int,
        routed_scaling_factor: float | None,
        activation: ActivationFn = ActivationFn.SILU,
        is_gated: bool = True,
        workspace: MoeLoraWorkspace | None = None,
        base_gemm_vendor: str = "cutedsl",
    ) -> None:
        if not providers:
            raise ValueError("a MoE LoRA runner needs at least one provider")
        self.providers = dict(providers)
        # Which vendor implements the base GEMMs, for the bind log.
        self.base_gemm_vendor = base_gemm_vendor
        # Every provider of a layer reads the same resident tensors, so the
        # geometry is a layer fact and lives here rather than per call.
        geometries = {
            (
                provider.hidden_size,
                provider.intermediate_size,
                provider.num_local_experts,
                provider.gate_up_slices,
                provider.contract.lora_delta_dtype,
            )
            for provider in self.providers.values()
        }
        if len(geometries) != 1:
            raise ValueError("providers of one layer must share resident geometry")
        (
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.gate_up_slices,
            self.lora_delta_dtype,
        ) = next(iter(geometries))
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        # The OTHER axis. Gating is a resident-shape fact -- the gate/up
        # buffer is one slice or two -- and is never inferred from the
        # activation, which is what made non-gated SiLU unservable.
        self.is_gated = is_gated
        self.workspace = workspace if workspace is not None else MoeLoraWorkspace()

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        row_orders: Sequence[str],
        vendor: str,
        workspace: MoeLoraWorkspace | None = None,
    ) -> MoeLoraRunner:
        """Admit the layer's resident state and bind one provider per row order.

        A layer needs at most two: the decode phase's and the prefill phase's.
        Both come from the same vendor, so the flag is read once per layer.
        """
        # No vendor fallback: _admit has already narrowed the device to SM90 or
        # SM100, and CuteDSL implements both. The fallback existed when the
        # floor was ">= SM90" and something had to serve what CuteDSL refused --
        # but DeepGEMM requires the very same two families, so below them there
        # was never anything to fall back to.
        cls._admit(base_layer)
        config = base_layer.moe_runner_config
        return cls(
            providers={
                rows: cls._build_provider(
                    base_layer, base_gemm_rows=rows, vendor=vendor
                )
                for rows in dict.fromkeys(row_orders)
            },
            base_gemm_vendor=vendor,
            # Layer-static routing scalars, read once rather than per forward.
            top_k=int(config.top_k),
            routed_scaling_factor=config.routed_scaling_factor,
            activation=ActivationFn.parse(config.activation),
            is_gated=bool(config.is_gated),
            workspace=workspace,
        )

    # ---- attach-time admission and validation ---------------------------

    @staticmethod
    def _admit(base_layer: FusedMoE) -> None:
        """Reject any resident state this engine does not actually consume.

        The base layer picks its runner backend, reformats resident weights,
        configures dispatch, and decides routed-scaling ownership BEFORE the
        LoRA layer attaches, so all of that is validated together here rather
        than assumed.
        """
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        if not isinstance(base_layer.quant_method, UnquantizedFusedMoEMethod):
            raise NotImplementedError(
                "MoE LoRA currently supports unquantized BF16 MoE only"
            )
        if (
            not isinstance(base_layer.dispatcher, StandardDispatcher)
            or base_layer.w13_weight.dtype != torch.bfloat16
            or base_layer.w2_weight.dtype != torch.bfloat16
        ):
            raise NotImplementedError(
                "MoE LoRA BF16 currently requires Standard dispatch and a "
                "resident BF16 provider"
            )

        major, minor = torch.cuda.get_device_capability()
        if major not in (9, 10):
            raise NotImplementedError(
                f"MoE LoRA BF16 supports SM90 and SM100 only; this device is "
                f"sm{major}{minor}. Its base GEMMs are WGMMA (SM90) and "
                "tcgen05 (SM100) kernels, and no other architecture implements "
                "either -- SM120 included, despite reporting a higher major."
            )
        if base_layer.dispatcher.skip_local_expert_mapping:
            raise NotImplementedError(
                "MoE LoRA BF16 requires EP-local expert IDs at the runner "
                "boundary, but this dispatcher keeps global IDs"
            )
        if base_layer.should_fuse_routed_scaling_factor_in_topk:
            raise NotImplementedError(
                "MoE LoRA BF16 applies routed scaling exactly once in its own "
                "finalize; this layer already folds it into the top-k weights"
            )

        config = base_layer.moe_runner_config
        supported_activation = config.activation in ActivationFn
        # Gating is validated as its own axis: the resident gate/up width must
        # agree with the layer's is_gated declaration.
        gateup_width = base_layer.w13_weight.shape[1]
        intermediate = base_layer.w2_weight.shape[2]
        if gateup_width != (2 if config.is_gated else 1) * intermediate:
            raise NotImplementedError(
                f"resident gate/up width {gateup_width} disagrees with "
                f"is_gated={config.is_gated} at intermediate {intermediate}"
            )
        if (
            not supported_activation
            or config.gemm1_alpha is not None
            or config.gemm1_clamp_limit is not None
            or config.swiglu_limit is not None
            or config.apply_router_weight_on_input
            or config.no_combine
            or config.num_fused_shared_experts
        ):
            raise NotImplementedError(
                "MoE LoRA BF16 supports SiLU or ReLU2 (gated or not) without "
                "fused shared experts, with route weighting owned by finalize"
            )

    @staticmethod
    def select_provider_cls(
        base_gemm_rows: str,
        vendor: str,
    ) -> type[MoeBaseProvider]:
        """Resolve (row order from the plan) x (vendor from serving config)."""
        if base_gemm_rows not in ("expert_major", "route_major"):
            raise ValueError(
                f"unknown MoE LoRA base-GEMM row order {base_gemm_rows!r}; "
                "expected 'expert_major' or 'route_major'"
            )
        if vendor == "cutedsl":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16ContiguousProvider,
                CuteDslBf16Provider,
            )

            return (
                CuteDslBf16Provider
                if base_gemm_rows == "expert_major"
                else CuteDslBf16ContiguousProvider
            )
        if vendor == "deepgemm":
            # Only these providers import deep_gemm, and deep_gemm_wrapper
            # binds its symbols only when the build is usable -- so an
            # unusable build must fail HERE, not at admission, or a CuteDSL
            # serve is refused for a dependency it never touches.
            from sglang.srt.layers import deep_gemm_wrapper

            if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
                raise NotImplementedError(
                    "--moe-lora-base-gemm deepgemm needs a usable JIT DeepGEMM "
                    "build; this one is disabled or absent"
                )
            from sglang.srt.lora.moe.base_gemm_provider.deep_gemm_bf16 import (
                DeepGemmBf16ContiguousProvider,
                DeepGemmBf16Provider,
            )

            return (
                DeepGemmBf16Provider
                if base_gemm_rows == "expert_major"
                else DeepGemmBf16ContiguousProvider
            )
        raise ValueError(
            f"unknown MoE LoRA base-GEMM vendor {vendor!r}; expected "
            "'cutedsl' or 'deepgemm'"
        )

    @classmethod
    def _build_provider(
        cls,
        base_layer: FusedMoE,
        *,
        base_gemm_rows: str,
        vendor: str,
    ) -> MoeBaseProvider:
        return cls.select_provider_cls(base_gemm_rows, vendor)(
            MoeLoraBf16QuantInfo(
                w13_weight=base_layer.w13_weight,
                w2_weight=base_layer.w2_weight,
                num_local_experts=int(base_layer.num_local_experts),
                intermediate_size=int(base_layer.w2_weight.shape[2]),
                hidden_size=int(base_layer.w2_weight.shape[1]),
            )
        )

    def validate_plan(self, plan: MoeLoraExecutionPlan, *, base_gemm_rows: str) -> None:
        """Reject unsupported provider/plan pairs before forward CUDA work.

        Takes the row order rather than the provider so the lookup stays with
        the ``providers`` dict that owns it, once, at bind time.
        """
        provider = self.providers[base_gemm_rows]
        plan.validate()
        if plan.act.activation is not self.activation:
            raise ValueError(
                f"plan activation {plan.act.activation.value} does not match "
                f"resident layer activation {self.activation.value}"
            )
        expected_slices = 2 if self.is_gated else 1
        if provider.gate_up_slices != expected_slices:
            raise ValueError(
                f"provider exposes {provider.gate_up_slices} gate/up slices "
                f"but is_gated={self.is_gated} needs {expected_slices}"
            )

        if plan.act.family is not ActFamily.MATERIALIZED:
            family, implementation = self._act_implementation(plan)
            if not provider.supports_fused_act(
                family,
                activation=self.activation.value,
                implementation=implementation,
            ):
                raise NotImplementedError(
                    f"{provider.contract.key} does not implement "
                    f"{family}/{implementation}"
                )
        if plan.down_b_scatter and not provider.supports_down_b_scatter():
            raise NotImplementedError(
                f"{provider.contract.key} does not implement the down-B "
                "scatter-into-base epilogue"
            )
        if plan.finalize.family is not FinalizeFamily.MATERIALIZED:
            family, implementation = self._finalize_implementation(plan)
            consumed_down_b = plan.finalize.consumed_down_b
            ownership_name = (
                "shared" if consumed_down_b.is_shared_outer else "per_expert"
            )
            if not provider.supports_fused_finalize(
                family,
                ownership_name,
                implementation=implementation,
            ):
                raise NotImplementedError(
                    f"{provider.contract.key} does not implement "
                    f"{family}/{ownership_name}/{implementation}"
                )

    @staticmethod
    def _act_implementation(
        plan: MoeLoraExecutionPlan,
    ) -> tuple[str, str]:
        return plan.act.family.value, "triton"

    @staticmethod
    def _finalize_implementation(
        plan: MoeLoraExecutionPlan,
    ) -> tuple[str, str]:
        return plan.finalize.family.value, "triton"

    # ---- forward --------------------------------------------------------

    def run_plan(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: MoeLoraBatch,
        *,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        base_gemm_rows: str,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        provider = self.providers[base_gemm_rows]
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        assert TopKOutputChecker.format_is_standard(topk_output)
        topk_ids = topk_output.topk_ids

        num_tokens = hidden_states.shape[0]
        if (
            batch.token_lora_mapping.ndim != 1
            or batch.token_lora_mapping.shape[0] != num_tokens
        ):
            raise RuntimeError(
                "MoE LoRA token/adapter assignment does not match the MoE "
                f"token domain: mapping has {batch.token_lora_mapping.shape[0]} rows "
                f"but the runner received {num_tokens}. Gather/remap "
                "assignments before MoE-DP execution."
            )

        self.workspace.begin_forward(graph_mode=batch.use_cuda_graph)
        routes = build_routes(
            plan,
            topk_ids=topk_ids,
            token_lora_mapping=batch.token_lora_mapping,
            num_local_experts=self.num_local_experts,
            max_loras=batch.max_loras,
            block_size=launch_config.routing_block_size,
            workspace=self.workspace,
        )

        gate_up, base_gemm_state, gateup_out = self._run_gate_up(
            plan,
            launch_config,
            provider,
            routes,
            hidden_states,
            topk_ids,
            batch,
            num_tokens,
        )
        act_out, down_a_input = self._run_act(
            plan,
            launch_config,
            provider,
            routes,
            base_gemm_state,
            gateup_out,
            gate_up,
            topk_ids,
            batch,
            num_tokens,
        )
        # Allocate before GEMM2 so a requested GEMM2 -> finalize PDL edge has
        # no allocator activity between its producer and dependent launch.
        output = self._allocate_output(
            num_tokens=num_tokens,
            dtype=hidden_states.dtype,
            device=act_out.device,
        )
        down_out, down_rank, down_delta = self._run_down(
            plan,
            launch_config,
            provider,
            routes,
            base_gemm_state,
            act_out,
            down_a_input,
            batch,
        )
        output = self._run_finalize(
            plan,
            launch_config,
            provider,
            routes,
            base_gemm_state,
            output,
            down_out,
            down_rank,
            down_delta,
            topk_output,
            batch,
            num_tokens,
        )
        return StandardCombineInput(hidden_states=output)

    @staticmethod
    def _route_for_a(spec: LoraASpec, routes: MoeLoraRoutes) -> RouteView:
        if spec.family is LoraAFamily.TOKEN_DEDUP_GROUPED:
            if routes.shared_token is None:
                raise ValueError("shared token route was not constructed")
            return routes.shared_token
        if spec.family is LoraAFamily.INDEXED:
            return routes.raw(spec.is_shared_outer)
        return routes.aligned(spec.is_shared_outer)

    @staticmethod
    def _route_for_b(spec: LoraBSpec, routes: MoeLoraRoutes) -> RouteView:
        if spec.family is LoraBFamily.INDEXED_PAIRS:
            return routes.raw(spec.is_shared_outer)
        return routes.aligned(spec.is_shared_outer)

    def _run_a(
        self,
        launch_config: MoeLoraLaunchConfig,
        spec: LoraASpec,
        input: torch.Tensor,
        weight: torch.Tensor,
        routes: MoeLoraRoutes,
        name: str,
        *,
        pair_to_row: torch.Tensor | None = None,
    ) -> torch.Tensor:
        route = self._route_for_a(spec, routes)
        if pair_to_row is not None:
            if pair_to_row.numel() != route.topk_ids.numel():
                raise ValueError(
                    "mapped down-A pair_to_row must have one entry per routed pair"
                )
        num_output_rows = (
            route.topk_ids.shape[0]
            if spec.output_layout is BridgeLayout.TOKEN_MAJOR
            else route.topk_ids.numel()
        )
        output = self.workspace.tensor(
            f"{name}:output",
            (num_output_rows, weight.shape[1]),
            dtype=self.lora_delta_dtype,
            device=input.device,
        )
        config = launch_config.for_a(spec.site)
        return run_lora_a(
            spec,
            input=input,
            weight=weight,
            output=output,
            routing=route,
            config=config,
            pair_to_row=pair_to_row,
        )

    def _run_b(
        self,
        launch_config: MoeLoraLaunchConfig,
        spec: LoraBSpec,
        bridge: torch.Tensor,
        weight: torch.Tensor,
        destination: torch.Tensor,
        routes: MoeLoraRoutes,
    ) -> torch.Tensor:
        route = self._route_for_b(spec, routes)
        config = launch_config.for_b(spec.site)
        if spec.family is LoraBFamily.ONE_LAUNCH_SLICED and "BLOCK_SIZE_M" in config:
            configured_block = int(config["BLOCK_SIZE_M"])
            if configured_block != route.block_size:
                raise ValueError(
                    f"{spec.family.value} LoRA-B consumes the aligned route's "
                    "exact BLOCK_SIZE_M: config declares "
                    f"{configured_block}, route uses {route.block_size}"
                )
        if spec.site is Site.GATE_UP:
            width = weight.shape[1] // self.gate_up_slices
            offsets = tuple(slice_id * width for slice_id in range(self.gate_up_slices))
        else:
            offsets = (0,)
        run_lora_b(
            spec,
            bridge=bridge,
            weight=weight,
            destination=destination,
            routing=route,
            destination_offsets=offsets,
            config=config,
            intermediate_top_k=(
                self.top_k if spec.input_layout is BridgeLayout.TOKEN_MAJOR else 1
            ),
        )

    def _run_gate_up(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> tuple[_LoraStageState, object, torch.Tensor]:
        state = _LoraStageState()

        def gate_up_a() -> None:
            state.rank = self._run_a(
                launch_config,
                plan.gate_up_a,
                hidden_states,
                batch.gate_up_lora_a.flatten(0, 1),
                routes,
                "gate_up_a",
            )

        def gate_up_b() -> None:
            delta = self.workspace.tensor(
                "gate_up_b:delta",
                (
                    num_tokens * self.top_k,
                    self.gate_up_slices * self.intermediate_size,
                ),
                dtype=self.lora_delta_dtype,
                device=state.rank.device,
            )
            self._run_b(
                launch_config,
                plan.gate_up_b,
                state.rank,
                batch.gate_up_lora_b.flatten(0, 1),
                delta,
                routes,
            )
            state.delta = delta

        def base() -> tuple[object, torch.Tensor]:
            base_gemm_state = provider.prepare(
                hidden_states,
                topk_ids,
                self.top_k,
                self.workspace,
            )
            gateup_out = self.workspace.tensor(
                "base:gateup",
                provider.gateup_out_shape(base_gemm_state),
                dtype=provider.contract.gate_up_output_dtype,
                device=hidden_states.device,
            )
            provider.gateup(base_gemm_state, gateup_out)
            provider.release_prepared_inputs(base_gemm_state)
            return base_gemm_state, gateup_out

        if plan.gate_up_overlap is GateUpOverlap.NONE:
            gate_up_a()
            if plan.gate_up_b is not None:
                gate_up_b()
            base_gemm_state, gateup = base()
        elif plan.gate_up_overlap is GateUpOverlap.GATE_UP_A:
            base_gemm_state, gateup = self.workspace.run_parallel(
                name=plan.gate_up_overlap.value,
                device=hidden_states.device,
                compute=base,
                side=gate_up_a,
            )
            if plan.gate_up_b is not None:
                gate_up_b()
        else:

            def gate_up_a_b() -> None:
                gate_up_a()
                gate_up_b()

            base_gemm_state, gateup = self.workspace.run_parallel(
                name=plan.gate_up_overlap.value,
                device=hidden_states.device,
                compute=base,
                side=gate_up_a_b,
            )
        return state, base_gemm_state, gateup

    def _run_act(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        base_gemm_state,
        gateup_out: torch.Tensor,
        gate_up: _LoraStageState,
        topk_ids: torch.Tensor,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> tuple[torch.Tensor, _DownAInput | None]:
        act_out = self.workspace.tensor(
            "act:masked",
            provider.act_out_shape(base_gemm_state),
            dtype=provider.contract.lora_activation_dtype,
            device=gateup_out.device,
        )
        exposes_pair_activation = True
        mapped_down_a: MappedLoraAInput | None = None
        if (
            plan.act.family is ActFamily.B_ACTIVATION
            and plan.down_a.family is LoraAFamily.GROUPED
        ):
            mapped_down_a = provider.mapped_down_lora_a_input(base_gemm_state, act_out)
            if mapped_down_a is not None:
                exposes_pair_activation = False
        act_pairs = (
            self.workspace.tensor(
                "act:pairs",
                (num_tokens, self.top_k, provider.intermediate_size),
                dtype=provider.contract.lora_activation_dtype,
                device=gateup_out.device,
            )
            if exposes_pair_activation
            else None
        )
        if plan.act.family is ActFamily.MATERIALIZED:
            provider.act_with_delta(
                base_gemm_state,
                gateup_out,
                gate_up.delta.view(
                    num_tokens,
                    self.top_k,
                    provider.gate_up_slices * provider.intermediate_size,
                ),
                topk_ids,
                act_out,
                act_pairs,
                activation=self.activation.value,
            )
            return act_out, _DownAInput(act_pairs)

        consumed_route = plan.act.consumed_gate_up_b
        route = routes.aligned(consumed_route.is_shared_outer)
        family, implementation = self._act_implementation(plan)
        provider.run_fused_act(
            base_gemm_state,
            family,
            implementation=implementation,
            activation=self.activation.value,
            base_gateup=gateup_out,
            act_masked=act_out,
            act_pairs=act_pairs,
            routing=route,
            config=launch_config.for_act(plan.act.family),
            bridge_gateup=gate_up.rank,
            b_gate_up=batch.gate_up_lora_b.flatten(0, 1),
            bridge_top_k=(
                self.top_k
                if plan.gate_up_a.output_layout is BridgeLayout.TOKEN_MAJOR
                else 1
            ),
        )
        if mapped_down_a is not None:
            down_a_input = _DownAInput(
                mapped_down_a.rows,
                mapped_down_a.pair_to_row,
            )
        elif act_pairs is not None:
            down_a_input = _DownAInput(act_pairs)
        else:
            down_a_input = None
        return act_out, down_a_input

    def _run_down(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        base_gemm_state,
        act_out: torch.Tensor,
        down_a_input: _DownAInput | None,
        batch: MoeLoraBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        state = _LoraStageState()

        def down_a() -> None:
            if down_a_input is None:
                raise RuntimeError("standalone down A requires pair activation")
            state.rank = self._run_a(
                launch_config,
                plan.down_a,
                down_a_input.rows.view(-1, self.intermediate_size),
                batch.down_lora_a.flatten(0, 1),
                routes,
                "down_a",
                pair_to_row=down_a_input.pair_to_row,
            )

        def down_b() -> None:
            delta = self.workspace.tensor(
                "down_b:delta",
                (state.rank.shape[0], self.hidden_size),
                dtype=self.lora_delta_dtype,
                device=state.rank.device,
            )
            self._run_b(
                launch_config,
                plan.down_b,
                state.rank,
                batch.down_lora_b.flatten(0, 1),
                delta,
                routes,
            )
            state.delta = delta

        def base() -> torch.Tensor:
            down_out = self.workspace.tensor(
                "base:down",
                provider.down_out_shape(base_gemm_state),
                dtype=torch.bfloat16,
                device=act_out.device,
            )
            provider.down(base_gemm_state, act_out, down_out)
            return down_out

        if plan.down_overlap is DownOverlap.NONE:
            if state.rank is None:
                down_a()
            if plan.down_b_scatter:
                # Experiment reordering (plan-validated to this serial
                # branch): the base down GEMM writes its rows FIRST, then the
                # same one-launch down-B tiling scatter-adds the unweighted
                # delta into them through src2dst, and the materialized
                # finalize runs in no-pair-delta mode.  The [T, K, H]
                # pair-major delta buffer is never allocated on this path.
                down_out = base()
                assert state.rank is not None
                # The same one-launch down-B tiling, the same site launch
                # config and aligned route, but targeting
                # down_out[src2dst[pair]] with a read-modify-write add instead
                # of storing a dense pair-major delta.
                provider.run_down_b_scatter(
                    base_gemm_state,
                    down_out=down_out,
                    bridge=state.rank,
                    b_down=batch.down_lora_b.flatten(0, 1),
                    routing=self._route_for_b(plan.down_b, routes),
                    config=launch_config.for_b(plan.down_b.site),
                )
            else:
                if plan.down_b is not None:
                    down_b()
                down_out = base()
        elif plan.down_overlap is DownOverlap.DOWN_A:
            down_out = self.workspace.run_parallel(
                name=plan.down_overlap.value,
                device=act_out.device,
                compute=base,
                side=down_a,
            )
            if plan.down_b is not None:
                down_b()
        elif plan.down_overlap is DownOverlap.DOWN_B:
            if state.rank is None:
                down_a()
            down_out = self.workspace.run_parallel(
                name=plan.down_overlap.value,
                device=act_out.device,
                compute=base,
                side=down_b,
            )
        else:

            def down_a_b() -> None:
                down_a()
                down_b()

            down_out = self.workspace.run_parallel(
                name=plan.down_overlap.value,
                device=act_out.device,
                compute=base,
                side=down_a_b,
            )

        return down_out, state.rank, state.delta

    def _allocate_output(
        self,
        *,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            return torch.empty(
                (num_tokens, self.hidden_size),
                dtype=dtype,
                device=device,
            )

    def _run_finalize(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        base_gemm_state,
        output: torch.Tensor,
        down_out: torch.Tensor,
        down_rank: torch.Tensor,
        down_delta: torch.Tensor | None,
        topk_output,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> torch.Tensor:
        if plan.finalize.family is FinalizeFamily.MATERIALIZED:
            if plan.down_b_scatter:
                # No-pair-delta mode: the unweighted delta was already
                # scatter-added into the base down rows.  NUMERICS: the delta
                # is rounded to BF16 JOINTLY with the base row before this
                # FP32 weighted top-k sum, whereas the shipped tail rounds
                # the delta to BF16 separately (pair-major) and keeps base
                # and delta as two BF16 operands of that sum — output
                # equality versus the shipped tail is therefore judged by
                # the established allclose discipline, not bitwise.
                provider.finalize(
                    base_gemm_state,
                    down_out,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                    self.routed_scaling_factor,
                    output,
                    lora_delta=None,
                )
                return output
            provider.finalize(
                base_gemm_state,
                down_out,
                topk_output.topk_ids,
                topk_output.topk_weights,
                self.routed_scaling_factor,
                output,
                lora_delta=down_delta.view(num_tokens, self.top_k, self.hidden_size),
            )
            return output

        consumed = plan.finalize.consumed_down_b
        route = routes.raw(consumed.is_shared_outer)
        b_down = batch.down_lora_b.flatten(0, 1)
        _, implementation = self._finalize_implementation(plan)
        provider.run_shared_rank_finalize(
            base_gemm_state,
            implementation=implementation,
            down_masked=down_out,
            bridge=down_rank,
            b_down=b_down,
            routing=route,
            topk_weights=topk_output.topk_weights,
            routed_scaling_factor=self.routed_scaling_factor,
            output=output,
            token_rank=self.workspace.tensor(
                "finalize:shared_token_rank",
                (num_tokens, down_rank.shape[1]),
                dtype=down_rank.dtype,
                device=down_rank.device,
            ),
            config=launch_config.shared_finalize,
        )
        return output


class MoeLoraLayerEngine:
    """Everything one MoE layer needs to run LoRA behind ``run_moe_core``.

    Construction reads layer-static facts; the first weight bind resolves
    one plan per phase from the plan tables, one tile table per plan from
    the tile tables, builds the runner, and validates every entry — all
    server-lifetime constants. The forward path is a phase lookup plus an
    M-bucket pick.
    """

    _config_logged = False

    def __init__(self, base_layer: FusedMoE, *, workspace: MoeLoraWorkspace) -> None:
        import torch as _torch

        weight_device = base_layer.w2_weight.device
        if weight_device.type != "cuda":
            raise NotImplementedError("MoE LoRA requires a CUDA layer")
        capability = _torch.cuda.get_device_capability(weight_device)
        config = base_layer.moe_runner_config
        self._base_layer = base_layer
        self.architecture = architecture_for_capability(*capability)
        self.activation = ActivationFn.parse(config.activation)
        self.hidden_size = int(base_layer.w2_weight.shape[1])
        self.num_local_experts = int(base_layer.num_local_experts)
        # Server-lifetime constant, so nothing vendor-shaped reaches a forward.
        self.base_gemm_vendor = get_lora().moe_lora_base_gemm
        self.workspace = workspace
        self._selected: dict[Phase, SelectedPlan] | None = None
        self._tiles: dict[Phase, TileTable] = {}
        self._runner: MoeLoraRunner | None = None
        self._is_shared_outer: bool | None = None
        self._physical_rank: int | None = None

    @property
    def is_bound(self) -> bool:
        return self._selected is not None

    def ensure_bound(self, *, is_shared_outer: bool, physical_rank: int) -> None:
        """Resolve plans, tiles, and the runner once; later binds only assert.

        Both inputs are server-lifetime constants (the resident layout flag
        and the pool-padded rank), so everything selection-shaped happens
        here and nothing remains for the forward path.
        """
        if physical_rank < 1:
            raise ValueError("the resident physical LoRA rank must be positive")
        if self.is_bound:
            if is_shared_outer != self._is_shared_outer:
                raise ValueError(
                    "resident MoE-LoRA factor layout changed after binding"
                )
            if physical_rank != self._physical_rank:
                raise ValueError(
                    "resident MoE-LoRA physical rank changed after binding"
                )
            return

        selected = resolve_plans(
            architecture=self.architecture,
            is_shared_outer=is_shared_outer,
            physical_rank=physical_rank,
            activation=self.activation,
            hidden_size=self.hidden_size,
            num_local_experts=self.num_local_experts,
        )
        runner = MoeLoraRunner.from_layer(
            self._base_layer,
            row_orders=tuple(sel.base_gemm_rows for sel in selected.values()),
            vendor=self.base_gemm_vendor,
            workspace=self.workspace,
        )
        tiles: dict[Phase, TileTable] = {}
        for phase, sel in selected.items():
            runner.validate_plan(sel.plan, base_gemm_rows=sel.base_gemm_rows)
            table = resolve_tiles(
                architecture_value=self.architecture.value,
                plan_key_name=sel.name,
                physical_rank=physical_rank,
            )
            table.validate_for_plan(sel.plan)
            tiles[phase] = table

        self._selected = selected
        self._tiles = tiles
        self._runner = runner
        self._is_shared_outer = is_shared_outer
        self._physical_rank = physical_rank
        if not MoeLoraLayerEngine._config_logged:
            MoeLoraLayerEngine._config_logged = True
            logger.info(
                "MoE LoRA plans bound (%s, hidden=%d, local_experts=%d, "
                "rank=%d): %s [base GEMM: %s]",
                self.architecture.value,
                self.hidden_size,
                self.num_local_experts,
                physical_rank,
                ", ".join(
                    f"{phase.value}={sel.key}@{sel.base_gemm_rows}"
                    for phase, sel in selected.items()
                ),
                runner.base_gemm_vendor,
            )

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: MoeLoraBatch,
    ) -> StandardCombineInput:
        if not self.is_bound:
            raise RuntimeError("MoE LoRA weights must be bound before running")
        phase = Phase.PREFILL if batch.is_prefill else Phase.DECODE
        sel = self._selected[phase]
        launch_config = self._tiles[phase].config_for(
            dispatch_output.hidden_states.shape[0]
        )
        assert self._runner is not None
        return self._runner.run_plan(
            dispatch_output,
            batch,
            plan=sel.plan,
            launch_config=launch_config,
            base_gemm_rows=sel.base_gemm_rows,
        )
