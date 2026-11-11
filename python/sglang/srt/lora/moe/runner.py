"""Execute a bound MoE LoRA plan; base-only tokens use adapter slot -1."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch
import triton

from sglang.kernels.ops.lora.common.lora_a import (
    grouped_lora_a,
    per_row_lora_a,
    shrink_column_tiles,
)
from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.kernels.ops.lora.moe.lora_b import invoke_down_b_into_base, run_lora_b
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
)
from sglang.srt.lora.moe.families import family_of_layer, resolve_vendor
from sglang.srt.lora.moe.plan import (
    ActFamily,
    AFamily,
    ASpec,
    BFamily,
    BridgeLayout,
    BSpec,
    DownOverlap,
    FinalizeFamily,
    GateUpOverlap,
    MoeLoraLaunchConfig,
    MoePlan,
    SelectedPlan,
    Site,
    resolve_plans,
)
from sglang.srt.lora.moe.quant_info import nvfp4_activation_scale_mode
from sglang.srt.lora.moe.routing import MoeLoraRoutes, build_moe_routes
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.srt.lora.workspace import LoraWorkspace

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
    use_cuda_graph: bool = False  # Graph-stable storage for either graph family.
    is_prefill: bool = False  # MoE plan phase; target verification uses decode.

    @property
    def max_loras(self) -> int:
        return self.gate_up_lora_a.shape[0]


logger = logging.getLogger(__name__)
# Every distinct plan selection a process makes is logged once (one line per
# rank for a homogeneous model), so a server log names the rows it serves.
_SELECTIONS_LOGGED: set[tuple] = set()


def _log_selection(key: tuple, plans: Mapping, providers: Mapping) -> None:
    """One INFO line per distinct resolved selection per process: the key
    (architecture, quant family, vendor, activation-scale mode, layout, rank)
    together with the selected row of every phase, its base-GEMM rows, the
    provider class serving those rows and the activation recipe, so two
    selections that share a key but differ in any of these each get a line."""
    rows = tuple(
        (phase.name.lower(), sel.name, sel.base_gemm_rows)
        for phase, sel in plans.items()
    )
    provider_names = tuple(
        (base_rows, type(provider).__name__)
        for base_rows, provider in providers.items()
    )
    recipe = next(
        (
            provider.quant_info.recipe
            for provider in providers.values()
            if getattr(provider.quant_info, "recipe", None) is not None
        ),
        None,
    )
    entry = (*key, rows, provider_names, repr(recipe))
    if entry in _SELECTIONS_LOGGED:
        return
    _SELECTIONS_LOGGED.add(entry)
    architecture, family, vendor, act_scale, is_shared_outer, physical_rank = key
    logger.info(
        "MoE LoRA plans selected for %s %s/%s act_scale=%s recipe=%s %s rank %d: "
        "%s; providers %s",
        architecture,
        family,
        vendor,
        act_scale,
        recipe,
        "shared-outer" if is_shared_outer else "per-expert",
        physical_rank,
        {phase: f"{name} ({base_rows})" for phase, name, base_rows in rows},
        dict(provider_names),
    )


def _epilogue_down_b(plan: MoePlan) -> bool:
    return plan.down_b is not None and plan.down_b.family is BFamily.EPILOGUE


class MoeLoraRunner:
    def __init__(
        self,
        *,
        providers: Mapping[str, MoeBaseProvider],
        top_k: int,
        routed_scaling_factor: float | None,
        activation: ActivationFn = ActivationFn.SILU,
        is_gated: bool = True,
        workspace: LoraWorkspace | None = None,
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
        self.workspace = workspace if workspace is not None else LoraWorkspace()
        self.plans: dict[Phase, SelectedPlan] = {}

    @property
    def derived_base_weights(self) -> str | None:
        """What the providers derive from the base weights, if anything."""
        reasons = dict.fromkeys(
            provider.derived_base_weights
            for provider in self.providers.values()
            if provider.derived_base_weights
        )
        return " and ".join(reasons) or None

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        workspace: LoraWorkspace | None = None,
        is_shared_outer: bool,
        physical_rank: int,
    ) -> MoeLoraRunner:
        family = cls._admit(base_layer)
        weight_device = base_layer.w2_weight.device
        if weight_device.type != "cuda":
            raise NotImplementedError("MoE LoRA requires a CUDA layer")
        major, _ = torch.cuda.get_device_capability(weight_device)
        architecture = architecture_for_capability(major)
        config = base_layer.moe_runner_config
        activation = ActivationFn.parse(config.activation)
        vendor = resolve_vendor(
            family, get_moe_runner_backend().value.removeprefix("lora_")
        )
        act_scale = cls._act_scale_mode(base_layer, family)
        plans = resolve_plans(
            architecture=architecture,
            is_shared_outer=is_shared_outer,
            physical_rank=physical_rank,
            activation=activation,
            hidden_size=int(base_layer.hidden_size),
            num_local_experts=int(base_layer.num_local_experts),
            quant_family=family,
            vendor=vendor,
            act_scale=act_scale,
        )
        # Shared finalizers hold the rank reduction in one tile, tested up to 256.
        if physical_rank > 256 and any(
            sel.plan.finalize.family is not FinalizeFamily.MATERIALIZED
            for sel in plans.values()
        ):
            raise ValueError(
                f"the shared finalizes are capped at rank 256, got {physical_rank}"
            )
        # One provider per row layout the plans use. One quant info per layer and
        # quant-info class: the layer's providers share it and what it derives.
        quant_infos: dict[type, object] = {}
        providers: dict[str, MoeBaseProvider] = {}
        for rows in dict.fromkeys(sel.base_gemm_rows for sel in plans.values()):
            provider_cls = select_provider_cls(rows, family, vendor)
            quant_info_cls = provider_cls.contract.quant_info_cls
            if quant_info_cls not in quant_infos:
                quant_infos[quant_info_cls] = quant_info_cls.from_layer(base_layer)
            providers[rows] = provider_cls(quant_infos[quant_info_cls])
        runner = cls(
            providers=providers,
            top_k=int(config.top_k),
            routed_scaling_factor=config.routed_scaling_factor,
            activation=activation,
            is_gated=bool(config.is_gated),
            workspace=workspace,
        )
        for sel in plans.values():
            runner.validate_plan(sel.plan, base_gemm_rows=sel.base_gemm_rows)
            if _epilogue_down_b(sel.plan):
                runner.providers[sel.base_gemm_rows].enable_epilogue_down_lora(
                    physical_rank
                )
        runner.plans = dict(plans)
        _log_selection(
            (architecture, family, vendor, act_scale, is_shared_outer, physical_rank),
            plans,
            runner.providers,
        )
        return runner

    @classmethod
    def _admit(cls, base_layer: FusedMoE) -> str:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher

        family = family_of_layer(base_layer)
        if not isinstance(base_layer.dispatcher, StandardDispatcher):
            raise NotImplementedError(
                "MoE LoRA currently requires the Standard dispatcher"
            )

        major, minor = torch.cuda.get_device_capability()
        if major not in (9, 10):
            raise NotImplementedError(
                f"MoE LoRA supports SM90 and SM100 only; this device is "
                f"sm{major}{minor}. Its base GEMMs are WGMMA (SM90) and "
                "tcgen05 (SM100) kernels, and no other architecture implements "
                "either -- SM120 included, despite reporting a higher major."
            )
        if base_layer.dispatcher.skip_local_expert_mapping:
            raise NotImplementedError(
                "MoE LoRA requires EP-local expert IDs at the runner "
                "boundary, but this dispatcher keeps global IDs"
            )
        if base_layer.should_fuse_routed_scaling_factor_in_topk:
            raise NotImplementedError(
                "MoE LoRA applies routed scaling exactly once in its own "
                "finalize; this layer already folds it into the top-k weights"
            )
        if getattr(base_layer, "with_bias", False):
            raise NotImplementedError(
                "MoE LoRA providers carry no expert biases; this layer has them"
            )

        config = base_layer.moe_runner_config
        # Membership by value: `str in Enum` raises TypeError before Python 3.12.
        supported_activation = config.activation in {fn.value for fn in ActivationFn}
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
                "MoE LoRA supports SiLU or ReLU2 (gated or not) without "
                "fused shared experts, with route weighting owned by finalize"
            )
        return family

    @staticmethod
    def _act_scale_mode(base_layer: FusedMoE, family: str) -> str:
        """The activation-scale mode the base layer resolved: NVFP4 layers
        follow their quant config's per-token flag; every other family is
        static."""
        if family != "nvfp4":
            return "static"
        return nvfp4_activation_scale_mode(base_layer)

    def validate_plan(self, plan: MoePlan, *, base_gemm_rows: str) -> None:
        provider = self.providers[base_gemm_rows]
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
        if _epilogue_down_b(plan) and not getattr(
            provider, "supports_epilogue_down_lora", False
        ):
            raise ValueError(
                f"the plan fuses down-B into the base GEMM epilogue, which "
                f"{provider.contract.key} does not implement"
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
        launch_config = selected.tiles.config_for(
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

        self.workspace.begin_forward(
            graph_mode=batch.use_cuda_graph, is_prefill_graph=batch.is_prefill
        )
        routes = build_moe_routes(
            selected.plan,
            topk_ids=topk_ids,
            token_lora_mapping=batch.token_lora_mapping,
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
        act_out, down_a_input, down_a_gather = self._run_act(
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

    def _run_a(
        self,
        launch_config: MoeLoraLaunchConfig,
        spec: ASpec,
        input: torch.Tensor,
        weight: torch.Tensor,
        routes: MoeLoraRoutes,
        name: str,
        *,
        pair_to_row: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``output`` replaces the pair-major bridge buffer: with
        ``output_rows`` the grouped kernel stores pair p at row output_rows[p]
        of its slot's plane (a 3-D output)."""
        if spec.family is AFamily.TOKEN_GROUPED:
            route = routes.shared_token
            if route is None:
                raise ValueError("shared token route was not constructed")
        elif spec.family in (AFamily.PER_ROW, AFamily.TOKEN_DENSE):
            route = routes.raw(spec.is_shared_outer)
        else:
            route = routes.aligned(spec.is_shared_outer)
        num_output_rows = (
            route.num_tokens
            if spec.output_layout is BridgeLayout.TOKEN_MAJOR
            else route.num_rows
        )
        shape: tuple[int, ...] = (num_output_rows, weight.shape[1])
        if spec.family is AFamily.TOKEN_DENSE:
            # One bridge plane per resident slot; per-row B indexes the planes.
            shape = (route.max_loras, num_output_rows, weight.shape[1])
        if output is None:
            output = self.workspace.tensor(
                f"{name}:output",
                shape,
                dtype=self.lora_delta_dtype,
                device=input.device,
            )
        elif spec.family is not AFamily.GROUPED:
            raise NotImplementedError(
                f"only the grouped A family writes a placed bridge, not {spec.family.value}"
            )
        config = launch_config.for_a(spec.site)
        pair_input = spec.site is Site.DOWN
        match spec.family.value:
            case "token_grouped":
                grouped_lora_a(input, weight, output, route, config=config)
            case "grouped":
                partial = counters = None
                split_k = int(config.get("SPLIT_K", 1))
                if split_k > 1:
                    # Serial split-K: the last arrival sums the FP32 planes and
                    # resets its counter for the next graph replay.
                    partial = self.workspace.tensor(
                        f"{name}:partial",
                        (split_k, num_output_rows, weight.shape[1]),
                        dtype=torch.float32,
                        device=input.device,
                    )
                    tiles = (
                        triton.cdiv(route.sorted_pair_ids.numel(), route.block_size)
                        * shrink_column_tiles(
                            int(config["BLOCK_SIZE_N"]), weight.shape[1], 1, False
                        )[1]
                    )
                    counters = self.workspace.tensor(
                        f"{name}:splitk_counters",
                        (tiles,),
                        dtype=torch.int32,
                        device=input.device,
                        zero_on_first_allocation=True,
                    )
                grouped_lora_a(
                    input,
                    weight,
                    output,
                    route,
                    config=config,
                    pair_input=pair_input,
                    pair_to_row=pair_to_row,
                    partial=partial,
                    counters=counters,
                    output_rows=output_rows,
                )
            case "per_row":
                per_row_lora_a(
                    input, weight, output, route, config=config, pair_input=pair_input
                )
            case "token_dense":
                torch.matmul(
                    input,
                    weight.transpose(1, 2),
                    out=output.view(weight.shape[0], input.shape[0], weight.shape[1]),
                )
            case family:
                raise NotImplementedError(f"no LoRA-A executor for {family!r}")
        return output

    def _run_b(
        self,
        launch_config: MoeLoraLaunchConfig,
        spec: BSpec,
        bridge: torch.Tensor,
        weight: torch.Tensor,
        destination: torch.Tensor,
        routes: MoeLoraRoutes,
        *,
        slot_planes: bool = False,
    ) -> None:
        route = (
            routes.raw(spec.is_shared_outer)
            if spec.family is BFamily.PER_ROW
            else routes.aligned(spec.is_shared_outer)
        )
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
            pair_bridge=spec.input_layout is BridgeLayout.PAIR_MAJOR,
            slot_planes=slot_planes,
        )

    def _run_gate_up(
        self,
        plan: MoePlan,
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
                slot_planes=plan.gate_up_a.family is AFamily.TOKEN_DENSE,
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
        plan: MoePlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        base_gemm_state,
        gateup_out: torch.Tensor,
        gate_up: _LoraStageState,
        topk_ids: torch.Tensor,
        batch: MoeLoraBatch,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        act_out = self.workspace.tensor(
            "act:masked",
            provider.act_out_shape(base_gemm_state),
            dtype=provider.contract.lora_activation_dtype,
            device=gateup_out.device,
        )
        provider_rows = act_out.view(-1, act_out.shape[-1])
        pair_to_row = base_gemm_state.pair_to_row
        # Fused B+activation can feed grouped down-A through the provider's row map.
        exposes_pair_activation = not (
            plan.act.family is ActFamily.B_ACTIVATION
            and plan.down_a.family is AFamily.GROUPED
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
            return act_out, act_pairs, None

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
            return act_out, act_pairs, None
        # Only provider-ordered input uses pair_to_row; pair-major input must not.
        return act_out, provider_rows, pair_to_row

    def _run_down(
        self,
        plan: MoePlan,
        launch_config: MoeLoraLaunchConfig,
        provider: MoeBaseProvider,
        routes: MoeLoraRoutes,
        base_gemm_state,
        act_out: torch.Tensor,
        down_a_input: torch.Tensor,
        down_a_gather: torch.Tensor | None,
        batch: MoeLoraBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        state = _LoraStageState()

        def down_a(**placed) -> None:
            state.rank = self._run_a(
                launch_config,
                plan.down_a,
                down_a_input.view(-1, self.intermediate_size),
                batch.down_lora_a.flatten(0, 1),
                routes,
                "down_a",
                pair_to_row=down_a_gather,
                **placed,
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
                pair_to_row=base_gemm_state.pair_to_row,
                bridge=state.rank,
                b_down=batch.down_lora_b.flatten(0, 1),
                # In-place B requires aligned rows even when the selected family is raw.
                routing=routes.aligned(plan.down_b.is_shared_outer),
                config=launch_config.for_b(plan.down_b.site),
            )

        def base() -> torch.Tensor:
            down_out = provider.down_output(base_gemm_state, self.workspace)
            provider.down(base_gemm_state, act_out, down_out)
            return down_out

        if _epilogue_down_b(plan):
            # The down A kernel places the raw bridge in the provider's row
            # order, one plane per slot; the base GEMM adds slot x expert
            # weights times those rows per tile.
            weights = batch.down_lora_b.flatten(0, 1)
            planes = provider.epilogue_down_lora_planes(
                base_gemm_state,
                self.workspace,
                max_loras=batch.max_loras,
                rank=weights.shape[-1],
            )
            if down_a_gather is None:
                raise ValueError("the epilogue down-B needs provider-ordered rows")
            down_a(output=planes, output_rows=down_a_gather)
            down_out = provider.down_output(base_gemm_state, self.workspace)
            provider.down(
                base_gemm_state,
                act_out,
                down_out,
                lora=(weights, planes, self._slots_used(batch)),
            )
            return down_out, state.rank, None

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

    def _slots_used(self, batch: MoeLoraBatch) -> torch.Tensor:
        """int32 [1]: the resident slots the batch addresses (max slot + 1),
        once per forward and mapping, replayed inside graphs."""
        mapping = batch.token_lora_mapping

        def build():
            slots = self.workspace.tensor(
                "lora:slots_used", (1,), dtype=torch.int32, device=mapping.device
            )
            slots.copy_(mapping.max().view(1))
            slots.add_(1)
            return slots

        return self.workspace.route(mapping, ("slots_used",), build)

    def _allocate_output(
        self,
        *,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric
        from sglang.srt.runtime_context import get_parallel

        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            return torch.empty(
                (num_tokens, self.hidden_size),
                dtype=dtype,
                device=device,
            )

    def _run_finalize(
        self,
        plan: MoePlan,
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
                    if plan.down_b_into_base or _epilogue_down_b(plan)
                    else down_delta.view(num_tokens, self.top_k, self.hidden_size)
                ),
            )
            return output

        provider.shared_outer_finalize(
            base_gemm_state,
            down_rows=down_out,
            bridge=down_rank,
            b_down=batch.down_lora_b.flatten(0, 1),
            routing=routes.raw(plan.finalize.is_shared_outer),
            topk_weights=topk_output.topk_weights,
            routed_scaling_factor=self.routed_scaling_factor,
            output=output,
            family=plan.finalize.family,
            launch_config=launch_config,
            workspace=self.workspace,
            token_route=routes.shared_token,
        )
        return output
