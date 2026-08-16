"""MoE-LoRA runner for the MoE LoRA execution engine.

``MoeLoraRunner`` is the single object the LoRA layer wrapper holds for one
MoE layer. Construction admits the resident provider contract and binds the
base provider; ``run`` executes the pipeline. No stock ``MoeRunner`` is
involved — the per-quant base stages live behind :class:`MoeBaseProvider`,
and this class owns the LoRA route views, the LoRA kernels, and every pipeline
buffer.

The default remains the serial correctness pipeline.  A typed
``MoeLoraExecutionPlan`` may instead force one of the retained fusion and
overlap candidates; every consumed stage then has exactly one owner and every
required route representation is built once:

    gate/up LoRA A  (grouped_lora_a: token-major hidden -> pair-major rank)
    gate/up LoRA B  (one_launch_sliced_lora_b -> canonical [gate | up] delta)
    S1 prepare      (provider permute to its physical row domain)
    S2 gateup       (provider grouped GEMM)
    S3 act          (base + delta -> activation; writes provider rows and,
                     when required, a canonical pair-major down-A source)
    down LoRA A     (grouped_lora_a, canonical pairs or provider-mapped rows)
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
canonicalizes such assignments to the ``-1`` execution sentinel before any
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

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MappedLoraAInput,
    MoeBaseProvider,
)
from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    BridgeLayout,
    EarlyOverlap,
    FinalizeFamily,
    LateOverlap,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MiddleFamily,
    MoeLoraExecutionPlan,
    Phase,
    SelectedPlan,
    Site,
    architecture_for_capability,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import (
    PROVISIONAL_LAUNCH_CONFIG,
    MoeLoraLaunchConfig,
    TileTable,
    resolve_tiles,
)
from sglang.srt.lora.moe.lora_a import run_lora_a
from sglang.srt.lora.moe.lora_b import run_lora_b
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.lora.moe.route_factory import (
    MoeLoraRoutes,
    build_routes,
)
from sglang.srt.lora.moe.routing import RouteView
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace, run_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass(slots=True)
class _GateLoraState:
    rank: torch.Tensor | None = None
    delta: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _DownAInput:
    """Standalone down-A source without exposing provider workspace details."""

    rows: torch.Tensor
    pair_to_row: torch.Tensor | None = None


_EARLY_PARALLEL_REGION = {
    EarlyOverlap.GATE_A: "early_gate_a",
    EarlyOverlap.GATE_A_B: "early_gate_a_b",
}
_LATE_PARALLEL_REGION = {
    LateOverlap.DOWN_A: "late_down_a",
    LateOverlap.DOWN_B: "late_down_b",
    LateOverlap.DOWN_A_B: "late_down_a_b",
}


class MoeLoraBatch(msgspec.Struct, kw_only=True):
    """The per-batch state the MoE-LoRA runner actually consumes.

    Narrow by design: the legacy ``LoRAInfo`` carries ~18 fields for the old
    kernels, and passing it wholesale would make it impossible to see what this
    runner depends on. ``token_slots`` holds canonical active physical slot
    IDs, with every inactive assignment represented by the ``-1`` sentinel.
    """

    gate_up_lora_a: torch.Tensor  # [L_cap, E_f, slices*R_phys, H]
    gate_up_lora_b: torch.Tensor  # [L_cap, E_local, slices*I, R_phys]
    down_lora_a: torch.Tensor  # [L_cap, E_local, R_phys, I]
    down_lora_b: torch.Tensor  # [L_cap, E_f_down, H, R_phys]
    token_slots: torch.Tensor  # [T] int, physical slot per token (-1 = base)
    adapter_enabled: torch.Tensor | None  # [L_cap], 0 marks an inactive slot
    use_cuda_graph: bool = False
    is_prefill: bool = False
    has_active_lora: bool = True

    @property
    def slot_capacity(self) -> int:
        return self.gate_up_lora_a.shape[0]


class MoeLoraRunner:
    """One MoE layer's MoE LoRA execution state and pipeline."""

    def __init__(
        self,
        *,
        providers: Mapping[str, MoeBaseProvider],
        top_k: int,
        routed_scaling_factor: float | None,
        activation: ActivationFamily = ActivationFamily.SWIGLU,
        workspace: MoeLoraWorkspace | None = None,
    ) -> None:
        if not providers:
            raise ValueError("a MoE LoRA runner needs at least one provider")
        self.providers = dict(providers)
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
        self.workspace = workspace if workspace is not None else MoeLoraWorkspace()

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        provider_names: Sequence[str],
        workspace: MoeLoraWorkspace | None = None,
    ) -> MoeLoraRunner:
        """Admit the layer's resident state and bind its unique providers."""
        cls._admit(base_layer)
        config = base_layer.moe_runner_config
        return cls(
            providers={
                name: cls._build_provider(base_layer, provider_name=name)
                for name in dict.fromkeys(provider_names)
            },
            # Layer-static routing scalars, read once rather than per forward.
            top_k=int(config.top_k),
            routed_scaling_factor=config.routed_scaling_factor,
            activation=(
                ActivationFamily.SWIGLU
                if config.activation == "silu"
                else ActivationFamily.RELU2
            ),
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
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.moe.utils import get_moe_runner_backend
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

        resident_backend = (
            base_layer.quant_method.runner.runner_backend
            if base_layer.quant_method.runner is not None
            else get_moe_runner_backend()
        )
        # Both shipped providers (DeepGEMM and CuTeDSL) consume the DeepGEMM
        # backend's canonical resident weight layout ([E, 2I, H] gate-first
        # BF16 with EP-local expert IDs), so admission is gated on that
        # backend AND on DeepGEMM being usable regardless of which provider
        # the env selects; a Triton-resident provider is separate later work.
        if not resident_backend.is_deep_gemm():
            raise NotImplementedError(
                "MoE LoRA BF16 currently requires --moe-runner-backend "
                "deep_gemm (canonical gate-first [E, 2I, H] BF16 weights and "
                f"EP-local expert IDs); this layer resolved to "
                f"{resident_backend}"
            )
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise NotImplementedError(
                "MoE LoRA BF16 requires a usable JIT DeepGEMM build: every "
                "base provider consumes the DeepGEMM-resident weight layout"
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
        supported_activation = config.activation in ("silu", "relu2")
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
        provider_name: str,
    ) -> type[MoeBaseProvider]:
        """Resolve the provider explicitly selected by the serving config."""
        from sglang.srt.lora.moe.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )

        if provider_name == "deepgemm":
            return DeepGemmBf16Provider
        if provider_name == "deepgemm_contiguous":
            from sglang.srt.lora.moe.base_gemm_provider.deep_gemm_bf16 import (
                DeepGemmBf16ContiguousProvider,
            )

            return DeepGemmBf16ContiguousProvider
        if provider_name == "cutedsl":
            if torch.cuda.get_device_capability() < (9, 0):
                raise NotImplementedError(
                    "the CuTeDSL MoE LoRA MoE provider requires SM90+ "
                    "(tcgen05 on SM100+, WGMMA on SM90); this device is "
                    f"sm{torch.cuda.get_device_capability()}"
                )
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16Provider,
            )

            return CuteDslBf16Provider
        if provider_name == "cutedsl_contiguous":
            if torch.cuda.get_device_capability() < (10, 0):
                raise NotImplementedError(
                    "the contiguous CuTeDSL MoE LoRA MoE provider requires "
                    "SM100+ (the SM90 kernel has no contiguous port); this "
                    f"device is sm{torch.cuda.get_device_capability()}"
                )
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16ContiguousProvider,
            )

            return CuteDslBf16ContiguousProvider
        raise ValueError(
            f"unknown MoE LoRA MoE provider {provider_name!r}; expected "
            "'deepgemm', 'deepgemm_contiguous', 'cutedsl', or "
            "'cutedsl_contiguous'"
        )

    @classmethod
    def _build_provider(
        cls,
        base_layer: FusedMoE,
        *,
        provider_name: str,
    ) -> MoeBaseProvider:
        return cls.select_provider_cls(provider_name)(
            MoeLoraBf16QuantInfo(
                w13_weight=base_layer.w13_weight,
                w2_weight=base_layer.w2_weight,
                num_local_experts=int(base_layer.num_local_experts),
                intermediate_size=int(base_layer.w2_weight.shape[2]),
                hidden_size=int(base_layer.w2_weight.shape[1]),
            )
        )

    def prepare_plan(
        self,
        plan: MoeLoraExecutionPlan,
        *,
        provider_name: str,
        is_shared_outer: bool,
    ) -> None:
        """Validate one menu entry against its provider, once, at bind time."""
        provider = self.providers[provider_name]
        plan.validate_ownership(is_shared_outer)
        self._validate_plan_provider(plan, provider)

    def _validate_plan_provider(
        self, plan: MoeLoraExecutionPlan, provider: MoeBaseProvider
    ) -> None:
        """Reject unsupported provider/plan pairs before forward CUDA work."""
        plan.validate()
        if plan.middle.activation is not self.activation:
            raise ValueError(
                f"plan activation {plan.middle.activation.value} does not match "
                f"resident layer activation {self.activation.value}"
            )
        expected_slices = 2 if self.activation is ActivationFamily.SWIGLU else 1
        if provider.gate_up_slices != expected_slices:
            raise ValueError(
                f"provider exposes {provider.gate_up_slices} gate/up "
                f"slices but {self.activation.value} needs {expected_slices}"
            )

        if plan.middle.family is not MiddleFamily.MATERIALIZED:
            family, implementation = self._middle_implementation(plan)
            if not provider.supports_fused_middle(
                family,
                activation=self._activation_name(),
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
            assert consumed_down_b is not None
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
    def _middle_implementation(
        plan: MoeLoraExecutionPlan,
    ) -> tuple[str, str]:
        return plan.middle.family.value, "triton"

    @staticmethod
    def _finalize_implementation(
        plan: MoeLoraExecutionPlan,
    ) -> tuple[str, str]:
        return plan.finalize.family.value, "triton"

    # ---- forward --------------------------------------------------------

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: MoeLoraBatch,
        *,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig = PROVISIONAL_LAUNCH_CONFIG,
        provider_name: str,
        output_dtype: torch.dtype | None = None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        provider = self.providers[provider_name]
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        assert TopKOutputChecker.format_is_standard(topk_output)
        topk_ids = topk_output.topk_ids

        output_dtype = hidden_states.dtype if output_dtype is None else output_dtype
        provider.validate_runtime_inputs(hidden_states, output_dtype=output_dtype)
        num_tokens = self._checked_token_count(hidden_states, batch)
        self.workspace.begin_forward(graph_mode=batch.use_cuda_graph)
        launch_config.validate_for_plan(plan)
        routes = build_routes(
            plan,
            topk_ids=topk_ids,
            token_slots=batch.token_slots,
            num_local_experts=self.num_local_experts,
            max_loras=batch.slot_capacity,
            block_size=launch_config.routing_block_size,
            gate_a_block_size=launch_config.gate_a_routing_block_size,
            workspace=self.workspace,
        )

        gate, ws, gateup_out = self._run_early(
            plan,
            launch_config,
            provider,
            routes,
            hidden_states,
            topk_ids,
            batch,
            num_tokens,
        )
        act_out, down_a_input = self._run_middle(
            plan,
            launch_config,
            provider,
            routes,
            ws,
            gateup_out,
            gate,
            topk_ids,
            batch,
            num_tokens,
        )
        # Allocate before GEMM2 so a requested GEMM2 -> finalize PDL edge has
        # no allocator activity between its producer and dependent launch.
        output = self._allocate_output(
            num_tokens=num_tokens,
            dtype=output_dtype,
            device=act_out.device,
        )
        down_out, down_rank, down_delta = self._run_late(
            plan,
            launch_config,
            provider,
            routes,
            ws,
            act_out,
            down_a_input,
            batch,
        )
        output = self._run_finalize(
            plan,
            launch_config,
            provider,
            routes,
            ws,
            output,
            down_out,
            down_rank,
            down_delta,
            topk_output,
            batch,
            num_tokens,
        )
        return StandardCombineInput(hidden_states=output)

    def _checked_token_count(
        self, hidden_states: torch.Tensor, batch: MoeLoraBatch
    ) -> int:
        num_tokens = hidden_states.shape[0]
        if batch.token_slots.ndim != 1 or batch.token_slots.shape[0] != num_tokens:
            raise RuntimeError(
                "MoE LoRA token/adapter assignment does not match the MoE "
                f"token domain: mapping has {batch.token_slots.shape[0]} rows "
                f"but the runner received {num_tokens}. Gather/remap "
                "assignments before MoE-DP execution."
            )
        if batch.token_slots.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "MoE LoRA token_slots must be int32 or int64, got "
                f"{batch.token_slots.dtype}"
            )
        if batch.token_slots.device != hidden_states.device:
            raise ValueError(
                "MoE LoRA token_slots and hidden states must share a device"
            )
        if batch.adapter_enabled is not None:
            if (
                batch.adapter_enabled.ndim != 1
                or batch.adapter_enabled.shape[0] != batch.slot_capacity
            ):
                raise ValueError(
                    "adapter_enabled must have one entry per resident LoRA slot"
                )
            if batch.adapter_enabled.device != hidden_states.device:
                raise ValueError(
                    "adapter_enabled and hidden states must share a device"
                )
        return num_tokens

    def _route_for_a(self, spec: LoraASpec, routes: MoeLoraRoutes) -> RouteView:
        if spec.family is LoraAFamily.TOKEN_DEDUP_GROUPED:
            if routes.shared_token is None:
                raise ValueError("shared token route was not constructed")
            return routes.shared_token
        if spec.family is LoraAFamily.INDEXED:
            return routes.raw(spec.is_shared_outer)
        if (
            spec.site is Site.GATE_UP
            and not spec.is_shared_outer
            and routes.gate_a_aligned_per_expert is not None
        ):
            return routes.gate_a_aligned_per_expert
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
        input_row_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        route = self._route_for_a(spec, routes)
        if input_row_map is not None:
            if not (spec.site is Site.DOWN and spec.family is LoraAFamily.GROUPED):
                raise ValueError(
                    "mapped provider rows are supported only by standalone "
                    "grouped down-A"
                )
            if (
                input_row_map.ndim != 1
                or input_row_map.dtype != torch.int32
                or input_row_map.device != input.device
                or not input_row_map.is_contiguous()
                or input_row_map.numel() != route.topk_ids.numel()
            ):
                raise ValueError(
                    "mapped down-A pair_to_row must be one contiguous int32 "
                    "entry per canonical routed pair on the input device"
                )
        num_tokens = (
            input.shape[0]
            if spec.site is Site.GATE_UP
            else (
                route.topk_ids.shape[0]
                if input_row_map is not None
                else input.shape[0] // self.top_k
            )
        )
        rows = (
            num_tokens
            if spec.output_layout is BridgeLayout.TOKEN_MAJOR
            else num_tokens * self.top_k
        )
        output = self.workspace.tensor(
            f"{name}:output",
            (rows, weight.shape[1]),
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
            input_row_map=input_row_map,
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

    def _run_gate_a(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        routes: MoeLoraRoutes,
        hidden_states: torch.Tensor,
        batch: MoeLoraBatch,
    ) -> torch.Tensor:
        return self._run_a(
            launch_config,
            plan.gate_a,
            hidden_states,
            batch.gate_up_lora_a.flatten(0, 1),
            routes,
            "gate_a",
        )

    def _run_gate_b(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        routes: MoeLoraRoutes,
        rank: torch.Tensor,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> torch.Tensor:
        if plan.gate_b is None:
            raise ValueError("the selected middle owns gate B")
        delta = self.workspace.tensor(
            "gate_b:delta",
            (
                num_tokens * self.top_k,
                self.gate_up_slices * self.intermediate_size,
            ),
            dtype=self.lora_delta_dtype,
            device=rank.device,
        )
        self._run_b(
            launch_config,
            plan.gate_b,
            rank,
            batch.gate_up_lora_b.flatten(0, 1),
            delta,
            routes,
        )
        return delta

    def _run_base_gateup(
        self,
        provider: MoeBaseProvider,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[object, torch.Tensor]:
        provider = provider
        ws = provider.prepare(
            hidden_states,
            topk_ids,
            self.top_k,
            self.workspace,
        )
        gateup_out = self.workspace.tensor(
            "base:gateup",
            provider.gateup_out_shape(ws),
            dtype=provider.contract.gate_up_output_dtype,
            device=hidden_states.device,
        )
        provider.gateup(ws, gateup_out)
        provider.release_prepared_inputs(ws)
        return ws, gateup_out

    def _run_early(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> tuple[_GateLoraState, object, torch.Tensor]:
        state = _GateLoraState()

        def gate_a() -> None:
            state.rank = self._run_gate_a(
                plan,
                launch_config,
                routes,
                hidden_states,
                batch,
            )

        def gate_b() -> None:
            if state.rank is None:
                raise RuntimeError("gate B ran before gate A")
            state.delta = self._run_gate_b(
                plan,
                launch_config,
                routes,
                state.rank,
                batch,
                num_tokens,
            )

        def base() -> tuple[object, torch.Tensor]:
            return self._run_base_gateup(
                provider,
                hidden_states,
                topk_ids,
            )

        if plan.early_overlap is EarlyOverlap.NONE:
            gate_a()
            if plan.gate_b is not None:
                gate_b()
            ws, gateup = base()
        elif plan.early_overlap is EarlyOverlap.GATE_A:
            ws, gateup = run_parallel(
                self.workspace,
                name=_EARLY_PARALLEL_REGION[EarlyOverlap.GATE_A],
                device=hidden_states.device,
                compute=base,
                side=gate_a,
            )
            if plan.gate_b is not None:
                gate_b()
        else:

            def gate_a_b() -> None:
                gate_a()
                gate_b()

            ws, gateup = run_parallel(
                self.workspace,
                name=_EARLY_PARALLEL_REGION[EarlyOverlap.GATE_A_B],
                device=hidden_states.device,
                compute=base,
                side=gate_a_b,
            )
        return state, ws, gateup

    def _activation_name(self) -> str:
        return "silu" if self.activation is ActivationFamily.SWIGLU else "relu2"

    def _run_middle(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        ws,
        gateup_out: torch.Tensor,
        gate: _GateLoraState,
        topk_ids: torch.Tensor,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> tuple[torch.Tensor, _DownAInput | None]:
        provider = provider
        act_out = self.workspace.tensor(
            "middle:act_masked",
            provider.act_out_shape(ws),
            dtype=provider.contract.lora_activation_dtype,
            device=gateup_out.device,
        )
        exposes_pair_activation = True
        mapped_down_a: MappedLoraAInput | None = None
        if (
            plan.middle.family is MiddleFamily.B_ACTIVATION
            and plan.down_a is not None
            and plan.down_a.family is LoraAFamily.GROUPED
        ):
            mapped_down_a = provider.mapped_down_lora_a_input(ws, act_out)
            if mapped_down_a is not None:
                exposes_pair_activation = False
        act_pairs = (
            self.workspace.tensor(
                "middle:act_pairs",
                (num_tokens, self.top_k, provider.intermediate_size),
                dtype=provider.contract.lora_activation_dtype,
                device=gateup_out.device,
            )
            if exposes_pair_activation
            else None
        )
        if plan.middle.family is MiddleFamily.MATERIALIZED:
            assert act_pairs is not None
            if gate.delta is None:
                raise RuntimeError("materialized middle requires gate/up delta")
            provider.act_with_delta(
                ws,
                gateup_out,
                gate.delta.view(
                    num_tokens,
                    self.top_k,
                    provider.gate_up_slices * provider.intermediate_size,
                ),
                topk_ids,
                act_out,
                act_pairs,
                activation=self._activation_name(),
            )
            return act_out, _DownAInput(act_pairs)

        consumed_route = plan.middle.consumed_gate_b
        assert consumed_route is not None
        route = routes.aligned(consumed_route.is_shared_outer)
        family, implementation = self._middle_implementation(plan)
        provider.run_fused_middle(
            ws,
            family,
            implementation=implementation,
            activation=self._activation_name(),
            base_gateup=gateup_out,
            act_masked=act_out,
            act_pairs=act_pairs,
            routing=route,
            config=launch_config.for_middle(plan.middle.family),
            bridge_gateup=gate.rank,
            b_gate_up=batch.gate_up_lora_b.flatten(0, 1),
            bridge_top_k=(
                self.top_k
                if plan.gate_a.output_layout is BridgeLayout.TOKEN_MAJOR
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

    def _run_down_a(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        routes: MoeLoraRoutes,
        down_a_input: _DownAInput,
        batch: MoeLoraBatch,
    ) -> torch.Tensor:
        if plan.down_a is None:
            raise ValueError("the execution plan has no standalone down A")
        return self._run_a(
            launch_config,
            plan.down_a,
            down_a_input.rows.view(-1, self.intermediate_size),
            batch.down_lora_a.flatten(0, 1),
            routes,
            "down_a",
            input_row_map=down_a_input.pair_to_row,
        )

    def _run_down_b(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        routes: MoeLoraRoutes,
        rank: torch.Tensor,
        batch: MoeLoraBatch,
    ) -> torch.Tensor:
        if plan.down_b is None:
            raise ValueError("the selected finalizer owns down B")
        delta = self.workspace.tensor(
            "down_b:delta",
            (rank.shape[0], self.hidden_size),
            dtype=self.lora_delta_dtype,
            device=rank.device,
        )
        self._run_b(
            launch_config,
            plan.down_b,
            rank,
            batch.down_lora_b.flatten(0, 1),
            delta,
            routes,
        )
        return delta

    def _run_down_b_scatter(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        ws,
        down_out: torch.Tensor,
        rank: torch.Tensor,
        batch: MoeLoraBatch,
    ) -> None:
        """Scatter-add the unweighted down-B delta into the base down rows.

        The down_b_scatter experiment: the SAME one-launch down-B tiling (and
        the same down-B site launch config and aligned route) runs AFTER the
        base down GEMM and targets ``down_out[src2dst[pair]]`` with a
        read-modify-write add instead of storing a dense pair-major delta, so
        the ``down_b:delta`` workspace buffer is never allocated.
        """
        spec = plan.down_b
        if spec is None:
            raise ValueError("down-B scatter requires the standalone down-B stage")
        provider.run_down_b_scatter(
            ws,
            down_out=down_out,
            bridge=rank,
            b_down=batch.down_lora_b.flatten(0, 1),
            routing=self._route_for_b(spec, routes),
            config=launch_config.for_b(spec.site),
        )

    def _run_base_down(
        self, provider: MoeBaseProvider, ws, act_out: torch.Tensor
    ) -> torch.Tensor:
        down_out = self.workspace.tensor(
            "base:down",
            provider.down_out_shape(ws),
            dtype=torch.bfloat16,
            device=act_out.device,
        )
        provider.down(ws, act_out, down_out)
        return down_out

    def _run_late(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        ws,
        act_out: torch.Tensor,
        down_a_input: _DownAInput | None,
        batch: MoeLoraBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        rank_holder: dict[str, torch.Tensor | None] = {"value": None}
        delta_holder: dict[str, torch.Tensor | None] = {"value": None}

        def down_a() -> None:
            if down_a_input is None:
                raise RuntimeError("standalone down A requires pair activation")
            rank_holder["value"] = self._run_down_a(
                plan,
                launch_config,
                routes,
                down_a_input,
                batch,
            )

        def down_b() -> None:
            rank = rank_holder["value"]
            if rank is None:
                raise RuntimeError("down B ran before down A")
            delta_holder["value"] = self._run_down_b(
                plan,
                launch_config,
                routes,
                rank,
                batch,
            )

        def base() -> torch.Tensor:
            return self._run_base_down(provider, ws, act_out)

        if plan.late_overlap is LateOverlap.NONE:
            if rank_holder["value"] is None:
                down_a()
            if plan.down_b_scatter:
                # Experiment reordering (plan-validated to this serial
                # branch): the base down GEMM writes its rows FIRST, then the
                # same one-launch down-B tiling scatter-adds the unweighted
                # delta into them through src2dst, and the materialized
                # finalize runs in no-pair-delta mode.  The [T, K, H]
                # pair-major delta buffer is never allocated on this path.
                down_out = base()
                rank = rank_holder["value"]
                assert rank is not None
                self._run_down_b_scatter(
                    plan, launch_config, provider, routes, ws, down_out, rank, batch
                )
            else:
                if plan.down_b is not None:
                    down_b()
                down_out = base()
        elif plan.late_overlap is LateOverlap.DOWN_A:
            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.DOWN_A],
                device=act_out.device,
                compute=base,
                side=down_a,
            )
            if plan.down_b is not None:
                down_b()
        elif plan.late_overlap is LateOverlap.DOWN_B:
            if rank_holder["value"] is None:
                down_a()
            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.DOWN_B],
                device=act_out.device,
                compute=base,
                side=down_b,
            )
        else:

            def down_a_b() -> None:
                down_a()
                down_b()

            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.DOWN_A_B],
                device=act_out.device,
                compute=base,
                side=down_a_b,
            )

        rank = rank_holder["value"]
        if rank is None:
            raise RuntimeError("execution plan did not produce down-A output")
        return down_out, rank, delta_holder["value"]

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
        ws,
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
                if down_delta is not None:
                    raise RuntimeError(
                        "the down-B scatter path materializes no LoRA delta"
                    )
                # No-pair-delta mode: the unweighted delta was already
                # scatter-added into the base down rows.  NUMERICS: the delta
                # is rounded to BF16 JOINTLY with the base row before this
                # FP32 weighted top-k sum, whereas the shipped tail rounds
                # the delta to BF16 separately (pair-major) and keeps base
                # and delta as two BF16 operands of that sum — output
                # equality versus the shipped tail is therefore judged by
                # the established allclose discipline, not bitwise.
                provider.finalize(
                    ws,
                    down_out,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                    self.routed_scaling_factor,
                    output,
                    lora_delta=None,
                )
                return output
            if down_delta is None:
                raise RuntimeError(
                    "materialized finalize requires a pair-major down delta"
                )
            provider.finalize(
                ws,
                down_out,
                topk_output.topk_ids,
                topk_output.topk_weights,
                self.routed_scaling_factor,
                output,
                lora_delta=down_delta.view(num_tokens, self.top_k, self.hidden_size),
            )
            return output

        consumed = plan.finalize.consumed_down_b
        assert consumed is not None
        route = routes.raw(consumed.is_shared_outer)
        b_down = batch.down_lora_b.flatten(0, 1)
        _, implementation = self._finalize_implementation(plan)
        provider.run_shared_rank_finalize(
            ws,
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
        self.activation = (
            ActivationFamily.SWIGLU
            if config.activation == "silu"
            else ActivationFamily.RELU2
        )
        self.hidden_size = int(base_layer.w2_weight.shape[1])
        self.num_local_experts = int(base_layer.num_local_experts)
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
            provider_names=tuple(sel.provider for sel in selected.values()),
            workspace=self.workspace,
        )
        tiles: dict[Phase, TileTable] = {}
        for phase, sel in selected.items():
            runner.prepare_plan(
                sel.plan,
                provider_name=sel.provider,
                is_shared_outer=is_shared_outer,
            )
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
                "MoE LoRA plans bound (%s, hidden=%d, local_experts=%d, rank=%d): %s",
                self.architecture.value,
                self.hidden_size,
                self.num_local_experts,
                physical_rank,
                ", ".join(
                    f"{phase.value}={sel.key}@{sel.provider}"
                    for phase, sel in selected.items()
                ),
            )

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: MoeLoraBatch,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> StandardCombineInput:
        if not self.is_bound:
            raise RuntimeError("MoE LoRA weights must be bound before running")
        phase = Phase.PREFILL if batch.is_prefill else Phase.DECODE
        sel = self._selected[phase]
        launch_config = self._tiles[phase].config_for(
            dispatch_output.hidden_states.shape[0]
        )
        assert self._runner is not None
        return self._runner.run(
            dispatch_output,
            batch,
            plan=sel.plan,
            launch_config=launch_config,
            provider_name=sel.provider,
            output_dtype=output_dtype,
        )
