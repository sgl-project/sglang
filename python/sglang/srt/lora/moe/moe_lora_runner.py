"""Execute a bound MoE LoRA plan; base-only tokens use adapter slot -1."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch

logger = logging.getLogger(__name__)

from sglang.srt.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.base_gemm_provider.base import (
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
from sglang.srt.lora.moe.kernels.lora_a import run_lora_a
from sglang.srt.lora.moe.kernels.lora_b import invoke_down_b_into_base, run_lora_b
from sglang.srt.lora.moe.launch_config import (
    MoeLoraLaunchConfig,
    TileTable,
    resolve_tiles,
)
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
    rank: torch.Tensor | None = None
    delta: torch.Tensor | None = None


class MoeLoraBatch(msgspec.Struct, kw_only=True):

    gate_up_lora_a: torch.Tensor  # [L_cap, E_f, slices*R_phys, H]
    gate_up_lora_b: torch.Tensor  # [L_cap, E_local, slices*I, R_phys]
    down_lora_a: torch.Tensor  # [L_cap, E_local, R_phys, I]
    down_lora_b: torch.Tensor  # [L_cap, E_f_down, H, R_phys]
    token_lora_mapping: torch.Tensor  # [T] int, adapter slot per token (-1 = base)
    seg_indptr: torch.Tensor  # [num_requests + 1] int, token boundaries per request
    use_cuda_graph: bool = False
    is_prefill: bool = False

    @property
    def max_loras(self) -> int:
        return self.gate_up_lora_a.shape[0]


class MoeLoraRunner:
    def __init__(
        self,
        *,
        providers: Mapping[str, MoeBaseProvider],
        top_k: int,
        routed_scaling_factor: float | None,
        activation: ActivationFn = ActivationFn.SILU,
        is_gated: bool = True,
        workspace: MoeLoraWorkspace | None = None,
    ) -> None:
        if not providers:
            raise ValueError("a MoE LoRA runner needs at least one provider")
        self.providers = dict(providers)
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
        self.is_gated = is_gated
        self.workspace = workspace if workspace is not None else MoeLoraWorkspace()
        self.plans: dict[Phase, SelectedPlan] = {}
        self.tiles: dict[Phase, TileTable] = {}

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        workspace: MoeLoraWorkspace | None = None,
        is_shared_outer: bool,
        physical_rank: int,
    ) -> MoeLoraRunner:
        cls._admit(base_layer)
        weight_device = base_layer.w2_weight.device
        if weight_device.type != "cuda":
            raise NotImplementedError("MoE LoRA requires a CUDA layer")
        architecture = architecture_for_capability(
            *torch.cuda.get_device_capability(weight_device)
        )
        config = base_layer.moe_runner_config
        activation = ActivationFn.parse(config.activation)
        plans = resolve_plans(
            architecture=architecture,
            is_shared_outer=is_shared_outer,
            physical_rank=physical_rank,
            activation=activation,
            hidden_size=int(base_layer.hidden_size),
            num_local_experts=int(base_layer.num_local_experts),
        )
        # The shared finalize holds a token's rank-space sum in one tile of
        # ``next_power_of_2(rank)`` columns, swept to 256.
        if physical_rank > 256 and any(
            sel.plan.finalize.family is not FinalizeFamily.MATERIALIZED
            for sel in plans.values()
        ):
            raise ValueError(
                f"the shared finalize is capped at rank 256, got {physical_rank}"
            )
        vendor = get_lora().moe_lora_base_gemm
        runner = cls(
            providers={
                rows: cls._build_provider(
                    base_layer, base_gemm_rows=rows, vendor=vendor
                )
                for rows in dict.fromkeys(sel.base_gemm_rows for sel in plans.values())
            },
            top_k=int(config.top_k),
            routed_scaling_factor=config.routed_scaling_factor,
            activation=activation,
            is_gated=bool(config.is_gated),
            workspace=workspace,
        )
        for phase, sel in plans.items():
            runner.validate_plan(sel.plan, base_gemm_rows=sel.base_gemm_rows)
            runner.tiles[phase] = resolve_tiles(
                architecture_value=architecture.value,
                plan_key_name=sel.name,
                physical_rank=physical_rank,
            )
        runner.plans = dict(plans)
        return runner

    @staticmethod
    def _admit(base_layer: FusedMoE) -> None:
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
        if getattr(base_layer, "with_bias", False):
            raise NotImplementedError(
                "MoE LoRA providers carry no expert biases; this layer has them"
            )

        config = base_layer.moe_runner_config
        # Membership by value: `str in Enum` raises TypeError before Python 3.12.
        supported_activation = config.activation in {fn.value for fn in ActivationFn}
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
        if base_gemm_rows not in ("expert_major", "route_major"):
            raise ValueError(
                f"unknown MoE LoRA base-GEMM row order {base_gemm_rows!r}; "
                "expected 'expert_major' or 'route_major'"
            )
        if vendor == "cutedsl":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16ContiguousProvider,
                CuteDslBf16MaskedProvider,
            )

            return (
                CuteDslBf16MaskedProvider
                if base_gemm_rows == "expert_major"
                else CuteDslBf16ContiguousProvider
            )
        if vendor == "triton":
            if base_gemm_rows != "route_major":
                raise ValueError(
                    "the triton base-GEMM provider is route-major only; "
                    f"a plan row asked for {base_gemm_rows!r}"
                )
            from sglang.srt.lora.moe.base_gemm_provider.triton_bf16 import (
                TritonBf16ContiguousProvider,
            )

            return TritonBf16ContiguousProvider
        raise ValueError(
            f"unknown MoE LoRA base-GEMM vendor {vendor!r}; expected "
            "'cutedsl' or 'triton'"
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

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: MoeLoraBatch,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        phase = Phase.PREFILL if batch.is_prefill else Phase.DECODE
        selected = self.plans[phase]
        launch_config = self.tiles[phase].config_for(
            dispatch_output.hidden_states.shape[0]
        )

        provider = self.providers[selected.base_gemm_rows]
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
            selected.plan,
            topk_ids=topk_ids,
            token_lora_mapping=batch.token_lora_mapping,
            seg_indptr=batch.seg_indptr,
            num_local_experts=self.num_local_experts,
            max_loras=batch.max_loras,
            block_size=launch_config.routing_block_size,
            workspace=self.workspace,
        )

        gate_up, base_gemm_state, gateup_out = self._run_gate_up(
            selected.plan,
            launch_config,
            provider,
            routes,
            hidden_states,
            topk_ids,
            batch,
            num_tokens,
        )
        act_out, down_a_input, down_a_gather, pair_to_row = self._run_act(
            selected.plan,
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
        output = self._allocate_output(
            num_tokens=num_tokens,
            dtype=hidden_states.dtype,
            device=act_out.device,
        )
        down_out, down_rank, down_delta = self._run_down(
            selected.plan,
            launch_config,
            provider,
            routes,
            base_gemm_state,
            act_out,
            down_a_input,
            down_a_gather,
            pair_to_row,
            batch,
        )
        output = self._run_finalize(
            selected.plan,
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
        if spec.family is LoraAFamily.TOKEN_GROUPED:
            if routes.shared_token is None:
                raise ValueError("shared token route was not constructed")
            return routes.shared_token
        if spec.family is LoraAFamily.PER_PAIR:
            return routes.raw(spec.is_shared_outer)
        return routes.aligned(spec.is_shared_outer)

    @staticmethod
    def _route_for_b(spec: LoraBSpec, routes: MoeLoraRoutes) -> RouteView:
        if spec.family is LoraBFamily.PER_PAIR:
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
    ) -> None:
        route = self._route_for_b(spec, routes)
        config = launch_config.for_b(spec.site)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        act_out = self.workspace.tensor(
            "act:masked",
            provider.act_out_shape(base_gemm_state),
            dtype=provider.contract.lora_activation_dtype,
            device=gateup_out.device,
        )
        provider_rows, pair_to_row = provider.mapped_down_lora_a_input(
            base_gemm_state, act_out
        )
        # Fused B+activation can feed grouped down-A through the provider's row map.
        exposes_pair_activation = not (
            plan.act.family is ActFamily.B_ACTIVATION
            and plan.down_a.family is LoraAFamily.GROUPED
        )
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
            # Pair-major input needs no gather.
            return act_out, act_pairs, None, pair_to_row

        provider.fused_act(
            base_gemm_state,
            activation=self.activation.value,
            base_gateup=gateup_out,
            act_rows=act_out,
            act_pairs=act_pairs,
            routing=routes.aligned(False),
            config=launch_config.for_act(plan.act.family),
            bridge_gateup=gate_up.rank,
            b_gate_up=batch.gate_up_lora_b.flatten(0, 1),
            bridge_top_k=(
                self.top_k
                if plan.gate_up_a.output_layout is BridgeLayout.TOKEN_MAJOR
                else 1
            ),
        )
        if exposes_pair_activation:
            return act_out, act_pairs, None, pair_to_row
        # Only provider-ordered input uses pair_to_row; pair-major input must not.
        return act_out, provider_rows, pair_to_row, pair_to_row

    def _run_down(
        self,
        plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        base_gemm_state,
        act_out: torch.Tensor,
        down_a_input: torch.Tensor,
        down_a_gather: torch.Tensor | None,
        pair_to_row: torch.Tensor,
        batch: MoeLoraBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        state = _LoraStageState()

        def down_a() -> None:
            state.rank = self._run_a(
                launch_config,
                plan.down_a,
                down_a_input.view(-1, self.intermediate_size),
                batch.down_lora_a.flatten(0, 1),
                routes,
                "down_a",
                pair_to_row=down_a_gather,
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

        def down_b_into_base(down_out: torch.Tensor) -> None:
            invoke_down_b_into_base(
                down_rows=down_out.view(-1, provider.hidden_size),
                pair_to_row=pair_to_row,
                bridge=state.rank,
                b_down=batch.down_lora_b.flatten(0, 1),
                # In-place B requires aligned rows even when the selected family is raw.
                routing=routes.aligned(plan.down_b.is_shared_outer),
                config=launch_config.for_b(plan.down_b.site),
            )

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
            down_a()
            if plan.down_b_into_base:
                down_out = base()
                down_b_into_base(down_out)
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
            if plan.down_b_into_base:
                down_b_into_base(down_out)
            elif plan.down_b is not None:
                down_b()
        elif plan.down_overlap is DownOverlap.DOWN_B:
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
            provider.finalize(
                base_gemm_state,
                down_out,
                topk_output.topk_ids,
                topk_output.topk_weights,
                self.routed_scaling_factor,
                output,
                lora_delta=(
                    None
                    if plan.down_b_into_base
                    else down_delta.view(num_tokens, self.top_k, self.hidden_size)
                ),
            )
            return output

        # The shared finalize applies the router weights in FP32 and rounds once.
        if topk_output.topk_weights.dtype != torch.float32:
            raise TypeError(
                "topk_weights must stay FP32 until the finalize applies them"
            )
        provider.shared_rank_finalize(
            base_gemm_state,
            down_rows=down_out,
            bridge=down_rank,
            b_down=batch.down_lora_b.flatten(0, 1),
            routing=routes.raw(plan.finalize.is_shared_outer),
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
