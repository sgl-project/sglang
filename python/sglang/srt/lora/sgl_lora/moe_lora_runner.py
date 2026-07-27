"""MoE-LoRA runner for the SGL LoRA execution engine.

``SglMoeLoraRunner`` is the single object the LoRA layer wrapper holds for one
MoE layer. Construction admits the resident provider contract and binds the
base provider; ``run`` executes the pipeline. No stock ``MoeRunner`` is
involved — the per-quant base stages live behind :class:`MoeBaseProvider`,
and this class owns the LoRA route views, the LoRA kernels, and every pipeline
buffer.

Serial pipeline (correctness baseline; overlap/fusion topologies are later
measured candidates, not defaults):

    gate/up LoRA A  (grouped_lora_a: token-major hidden -> pair-major rank)
    gate/up LoRA B  (stock_grouped_lora_b -> canonical [gate | up] delta)
    S1 prepare      (provider permute to its physical row domain)
    S2 gateup       (provider grouped GEMM)
    S3 act          (base + delta -> SwiGLU; writes provider rows AND the
                     canonical pair-major down-A source)
    down LoRA A     (grouped_lora_a, pair-major input)
    down LoRA B     (stock_grouped_lora_b -> unweighted pair delta [T, K, H])
    S4 down         (provider grouped GEMM)
    S5 finalize     (provider fixed-order top-k reduction; router coefficient
                     and routed scaling applied EXACTLY ONCE over
                     base + pair delta, at the provider-declared coefficient
                     precision)

Every batch runs this one LoRA-capable topology — base-only, mixed, and active
alike — so they share a single graph shape. Inactive assignments ride sentinel
routes and contribute exact zeros rather than being diverted to another path.

Base rows: serving gives the base model a REAL resident slot whose factors are
zero-filled and whose ``adapter_enabled`` entry is 0.  Such slots are masked to
the ``-1`` execution sentinel before routing — otherwise base rows build routed
work against zero weights, which is numerically harmless but inflates route
padding, group counts, and every LoRA GEMM's row count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import MoeBaseProvider
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a, stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    RouteView,
    build_virtual_expert_routing,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class Bf16MoeLaunchConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Explicit provisional tiles; this is not a serving selector."""

    routing_block_size: int
    lora_a: dict[str, int]
    lora_b: dict[str, int]


# Single source of truth for the provisional tiles. Serving, the benchmark lab,
# and the tests all import this so a change cannot land in one and not another.
# It is a correctness baseline; the measured policy arrives with the campaign's
# tuning evidence.
PROVISIONAL_LAUNCH_CONFIG = Bf16MoeLaunchConfig(
    routing_block_size=16,
    lora_a={
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    },
    lora_b={
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    },
)


class SglMoeLoraBatch(msgspec.Struct, kw_only=True):
    """The per-batch state the MoE-LoRA runner actually consumes.

    Narrow by design: the legacy ``LoRAInfo`` carries ~18 fields for the old
    kernels, and passing it wholesale would make it impossible to see what this
    runner depends on. ``token_slots`` still holds physical slot IDs; the
    runner masks disabled slots to the ``-1`` sentinel.
    """

    gate_up_lora_a: torch.Tensor  # [L_cap, E_f, slices*R_phys, H]
    gate_up_lora_b: torch.Tensor  # [L_cap, E_local, slices*I, R_phys]
    down_lora_a: torch.Tensor  # [L_cap, E_local, R_phys, I]
    down_lora_b: torch.Tensor  # [L_cap, E_f_down, H, R_phys]
    token_slots: torch.Tensor  # [T] int, physical slot per token (-1 = base)
    adapter_enabled: torch.Tensor | None  # [L_cap], 0 marks an inactive slot
    physical_rank: int
    shared_outer: bool

    @property
    def slot_capacity(self) -> int:
        return self.gate_up_lora_a.shape[0]


class SglMoeLoraRunner:
    """One MoE layer's SGL LoRA execution state and pipeline."""

    def __init__(
        self,
        *,
        provider: MoeBaseProvider,
        top_k: int,
        routed_scaling_factor: float | None,
        launch_config: Bf16MoeLaunchConfig = PROVISIONAL_LAUNCH_CONFIG,
    ) -> None:
        self.provider = provider
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.launch_config = launch_config

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        launch_config: Bf16MoeLaunchConfig = PROVISIONAL_LAUNCH_CONFIG,
    ) -> SglMoeLoraRunner:
        """Admit the layer's resident state and bind a base provider to it."""
        cls._admit(base_layer)
        config = base_layer.moe_runner_config
        return cls(
            provider=cls._build_provider(base_layer),
            # Layer-static routing scalars, read once rather than per forward.
            top_k=int(config.top_k),
            routed_scaling_factor=config.routed_scaling_factor,
            launch_config=launch_config,
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
                "sgl_lora currently supports unquantized BF16 MoE only"
            )
        if (
            not isinstance(base_layer.dispatcher, StandardDispatcher)
            or base_layer.w13_weight.dtype != torch.bfloat16
            or base_layer.w2_weight.dtype != torch.bfloat16
        ):
            raise NotImplementedError(
                "sgl_lora BF16 currently requires Standard dispatch and a "
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
                "sgl_lora BF16 currently requires --moe-runner-backend "
                "deep_gemm (canonical gate-first [E, 2I, H] BF16 weights and "
                f"EP-local expert IDs); this layer resolved to "
                f"{resident_backend}"
            )
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise NotImplementedError(
                "sgl_lora BF16 requires a usable JIT DeepGEMM build: every "
                "base provider consumes the DeepGEMM-resident weight layout"
            )
        if base_layer.dispatcher.skip_local_expert_mapping:
            raise NotImplementedError(
                "sgl_lora BF16 requires EP-local expert IDs at the runner "
                "boundary, but this dispatcher keeps global IDs"
            )
        if base_layer.should_fuse_routed_scaling_factor_in_topk:
            raise NotImplementedError(
                "sgl_lora BF16 applies routed scaling exactly once in its own "
                "finalize; this layer already folds it into the top-k weights"
            )

        config = base_layer.moe_runner_config
        if (
            config.activation != "silu"
            or not config.is_gated
            or config.gemm1_alpha is not None
            or config.gemm1_clamp_limit is not None
            or config.swiglu_limit is not None
            or config.apply_router_weight_on_input
            or config.no_combine
            or config.num_fused_shared_experts
        ):
            raise NotImplementedError(
                "sgl_lora BF16 currently supports canonical gated SiLU without "
                "fused shared experts, with route weighting owned by finalize"
            )

    @staticmethod
    def select_provider_cls() -> type[MoeBaseProvider]:
        """The env-selected provider class; shared with the guardrail matrix.

        The production-runner backend of the correctness matrix must construct
        its provider through THIS selection, not by naming a class, or an
        env-selected provider is never the one the matrix observes.
        """
        from sglang.srt.environ import envs
        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )

        # An enumerable selection rather than an if-chain: an unknown name or
        # an unsupported device fails HERE at attach, never as a wrong default
        # (the failure mode review round 4 caught in the backend admission).
        selected = envs.SGLANG_LORA_MOE_BASE_PROVIDER.get()
        if selected == "deepgemm":
            return DeepGemmBf16Provider
        if selected == "cutedsl":
            if torch.cuda.get_device_capability() < (9, 0):
                raise NotImplementedError(
                    "SGLANG_LORA_MOE_BASE_PROVIDER=cutedsl requires SM90+ "
                    "(tcgen05 on SM100+, WGMMA on SM90); this device is "
                    f"sm{torch.cuda.get_device_capability()}"
                )
            from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16Provider,
            )

            return CuteDslBf16Provider
        raise ValueError(
            f"unknown SGLANG_LORA_MOE_BASE_PROVIDER {selected!r}; expected "
            "'deepgemm' or 'cutedsl'"
        )

    @classmethod
    def _build_provider(cls, base_layer: FusedMoE) -> MoeBaseProvider:
        return cls.select_provider_cls()(
            SglLoraBf16QuantInfo(
                w13_weight=base_layer.w13_weight,
                w2_weight=base_layer.w2_weight,
                num_local_experts=int(base_layer.num_local_experts),
                intermediate_size=int(base_layer.w2_weight.shape[2]),
                hidden_size=int(base_layer.w2_weight.shape[1]),
            )
        )

    def validate_factors(
        self,
        *,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        shared_outer: bool,
    ) -> None:
        """Check the immutable LoRA-weight contract once, when weights bind.

        Dtype and expert domain cannot change between forwards, so validating
        per forward would only add launch overhead.
        """
        # Eleventh S3 review: the provider is the single source of truth
        # for the local expert count; the constructor used to take a
        # duplicate that was always this same value.
        expert_count = self.provider.num_local_experts
        outer_count = 1 if shared_outer else expert_count
        expected = (outer_count, expert_count, expert_count, outer_count)
        factors = (gate_up_lora_a, gate_up_lora_b, down_lora_a, down_lora_b)
        actual = tuple(weight.shape[1] for weight in factors)
        if actual != expected:
            raise ValueError(
                "SGL LoRA factor domains do not match the resident provider: "
                f"expected {expected}, got {actual}"
            )
        expected_dtype = self.provider.contract.lora_delta_dtype
        for name, weight in zip(
            ("gate_up_lora_a", "gate_up_lora_b", "down_lora_a", "down_lora_b"),
            factors,
        ):
            if weight.dtype != expected_dtype:
                raise TypeError(
                    f"sgl_lora requires {expected_dtype} {name}, got {weight.dtype}"
                )

    # ---- forward --------------------------------------------------------

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: SglMoeLoraBatch,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.srt.utils import dispose_tensor

        provider = self.provider
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        assert TopKOutputChecker.format_is_standard(topk_output)
        topk_ids = topk_output.topk_ids

        output_dtype = hidden_states.dtype if output_dtype is None else output_dtype
        provider.validate_runtime_inputs(hidden_states, output_dtype=output_dtype)
        num_tokens = self._checked_token_count(hidden_states, batch)
        # Shared-outer factors are keyed by ADAPTER alone, so gate/up-A and
        # down-B read the outer domain while gate/up-B and down-A stay
        # per-expert. Without shared-outer the two are the same route.
        per_expert, outer = self._route_views(batch, topk_ids)

        gate_up_delta = self._gate_up_lora(
            hidden_states, batch, num_tokens, a_route=outer, b_route=per_expert
        )

        # ---- base S1/S2 and the S3 LoRA injection join. ----
        ws = provider.prepare(hidden_states, topk_ids, self.top_k)
        gateup_out = torch.empty(
            provider.gateup_out_shape(ws),
            dtype=provider.contract.gate_up_output_dtype,
            device=hidden_states.device,
        )
        provider.gateup(ws, gateup_out)
        provider.release_prepared_inputs(ws)

        act_out, activation_lora_input = self._activate(
            ws, gateup_out, gate_up_delta, topk_ids, num_tokens
        )
        dispose_tensor(gateup_out)
        dispose_tensor(gate_up_delta)

        down_delta = self._down_lora(
            activation_lora_input,
            batch,
            num_tokens,
            a_route=per_expert,
            b_route=outer,
        )
        dispose_tensor(activation_lora_input)

        # ---- base S4/S5: weights + routed scaling once over base + delta. ----
        down_out = torch.empty(
            provider.down_out_shape(ws),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        provider.down(ws, act_out, down_out)
        dispose_tensor(act_out)

        output = self._finalize(
            ws, down_out, down_delta, topk_output, num_tokens, output_dtype
        )
        dispose_tensor(down_out)
        dispose_tensor(down_delta)
        return StandardCombineInput(hidden_states=output)

    # ---- pipeline stages -------------------------------------------------

    def _checked_token_count(
        self, hidden_states: torch.Tensor, batch: SglMoeLoraBatch
    ) -> int:
        num_tokens = hidden_states.shape[0]
        if batch.token_slots.shape[0] != num_tokens:
            raise RuntimeError(
                "sgl_lora token/adapter assignment does not match the MoE "
                f"token domain: mapping has {batch.token_slots.shape[0]} rows "
                f"but the runner received {num_tokens}. Gather/remap "
                "assignments before MoE-DP execution."
            )
        return num_tokens

    def _route_views(
        self, batch: SglMoeLoraBatch, topk_ids: torch.Tensor
    ) -> tuple[RouteView, RouteView]:
        """Return ``(per_expert, outer)`` — only the views a site consumes.

        Without shared-outer factors both are the same object, so no second
        plan is built.
        """
        token_slots = _mask_disabled_adapter_slots(batch)
        # Both LoRA kernels in this serial topology are grouped GEMMs, so this
        # schedule consumes the aligned plan. An indexed schedule would request
        # ROUTE_RAW here and derive the key inline instead.
        expert = build_virtual_expert_routing(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=self.provider.num_local_experts,
            max_loras=batch.slot_capacity,
            block_size=self.launch_config.routing_block_size,
            view=ROUTE_ALIGNED,
        )
        outer = expert
        if batch.shared_outer:
            # Section 60.5 form: the shared factor is slot 0 for every
            # EP-owned routed expert — a constexpr in the key derivation,
            # not a map gather (admission guarantees local-domain ids).
            outer = build_virtual_expert_routing(
                topk_ids,
                token_slots,
                lora_experts_per_adapter=1,
                max_loras=batch.slot_capacity,
                block_size=self.launch_config.routing_block_size,
                shared_outer_local_expert_count=self.provider.num_local_experts,
            )
        return expert, outer

    def _gate_up_lora(
        self,
        hidden_states: torch.Tensor,
        batch: SglMoeLoraBatch,
        num_tokens: int,
        *,
        a_route: RouteView,
        b_route: RouteView,
    ) -> torch.Tensor:
        """Canonical ``[gate | up]`` pair-major delta, pre-activation."""
        from sglang.srt.utils import dispose_tensor

        device = hidden_states.device
        delta_dtype = self.provider.contract.lora_delta_dtype
        inter = self.provider.intermediate_size
        num_pairs = num_tokens * self.top_k

        rank_out = torch.empty(
            (num_pairs, batch.gate_up_lora_a.shape[2]),
            dtype=delta_dtype,
            device=device,
        )
        grouped_lora_a(
            hidden_states,
            batch.gate_up_lora_a.flatten(0, 1),
            rank_out,
            a_route,
            config=self.launch_config.lora_a,
        )
        delta = torch.empty((num_pairs, 2 * inter), dtype=delta_dtype, device=device)
        stock_grouped_lora_b(
            rank_out,
            batch.gate_up_lora_b.flatten(0, 1),
            delta,
            b_route,
            # Canonical [gate | up] regardless of the provider's base-output
            # column order: the S3 join reads the delta canonically and applies
            # the provider's gate_first/interleaved layout to the BASE rows.
            destination_offsets=(0, inter),
            config=self.launch_config.lora_b,
        )
        dispose_tensor(rank_out)
        return delta

    def _activate(
        self,
        ws,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor,
        topk_ids: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """S3: base + delta -> activation, plus the pair-major down-A source."""
        provider = self.provider
        inter = provider.intermediate_size
        act_dtype = provider.contract.lora_activation_dtype
        device = gateup_out.device

        act_out = torch.empty(
            provider.act_out_shape(ws), dtype=act_dtype, device=device
        )
        activation_lora_input = torch.empty(
            (num_tokens, self.top_k, inter), dtype=act_dtype, device=device
        )
        provider.act_with_delta(
            ws,
            gateup_out,
            gate_up_delta.view(num_tokens, self.top_k, 2 * inter),
            topk_ids,
            act_out,
            activation_lora_input,
        )
        return act_out, activation_lora_input

    def _down_lora(
        self,
        activation_lora_input: torch.Tensor,
        batch: SglMoeLoraBatch,
        num_tokens: int,
        *,
        a_route: RouteView,
        b_route: RouteView,
    ) -> torch.Tensor:
        """Unweighted canonical pair delta ``[T*K, H]`` for the combine."""
        from sglang.srt.utils import dispose_tensor

        device = activation_lora_input.device
        delta_dtype = self.provider.contract.lora_delta_dtype
        inter = self.provider.intermediate_size
        num_pairs = num_tokens * self.top_k

        rank_out = torch.empty(
            (num_pairs, batch.down_lora_a.shape[2]), dtype=delta_dtype, device=device
        )
        grouped_lora_a(
            activation_lora_input.view(num_pairs, inter),
            batch.down_lora_a.flatten(0, 1),
            rank_out,
            a_route,
            config=self.launch_config.lora_a,
            pair_input=True,
        )
        delta = torch.empty(
            (num_pairs, self.provider.hidden_size), dtype=delta_dtype, device=device
        )
        stock_grouped_lora_b(
            rank_out,
            batch.down_lora_b.flatten(0, 1),
            delta,
            b_route,
            destination_offsets=(0,),
            config=self.launch_config.lora_b,
        )
        dispose_tensor(rank_out)
        return delta

    def _finalize(
        self,
        ws,
        down_out: torch.Tensor,
        down_delta: torch.Tensor,
        topk_output,
        num_tokens: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """S5 combine into a fresh output buffer."""
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric

        hidden = self.provider.hidden_size
        # Allocate the returned tensor from the NCCL symmetric pool so a
        # downstream TP collective can use its low-latency algorithms, which
        # require matching addresses across ranks. DP attention with ragged
        # per-rank token counts breaks that symmetry, hence the disable.
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output = torch.empty(
                (num_tokens, hidden), dtype=output_dtype, device=down_out.device
            )
        self.provider.finalize(
            ws,
            down_out,
            topk_output.topk_ids,
            topk_output.topk_weights,
            self.routed_scaling_factor,
            output,
            pair_delta=down_delta.view(num_tokens, self.top_k, hidden),
        )
        return output


def _mask_disabled_adapter_slots(batch: SglMoeLoraBatch) -> torch.Tensor:
    """Map disabled resident slots to the ``-1`` execution sentinel.

    ``token_slots`` carries physical slot IDs, including the base model's own
    zero-filled slot; ``adapter_enabled[slot] == 0`` marks a slot that must not
    produce LoRA work.  Masking here keeps the sentinel contract in ONE place
    (device-side, graph-safe, no host sync) so every downstream route view and
    kernel shares a single notion of "inactive".
    """
    mapping = batch.token_slots
    adapter_enabled = batch.adapter_enabled
    if adapter_enabled is None:
        return mapping
    enabled = adapter_enabled.to(torch.bool)[mapping.clamp_min(0).long()]
    return torch.where((mapping >= 0) & enabled, mapping, torch.full_like(mapping, -1))
