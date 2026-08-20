"""The masked row layout that both BF16 base GEMM providers share.

The rows are ``[E_local, m_max, ·]``. ``masked_m`` holds the row count of each
expert. ``src2dst`` maps a routed pair to its row. The preprocess, the
activation, and the finalize are Triton kernels over this layout. They do not
depend on the GEMM engine, so this class runs them for every provider. The
DeepGEMM subclass and the CuTeDSL subclass add only ``gateup`` and ``down``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MappedLoraAInput,
    MoeBaseProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class MaskedRowState(msgspec.Struct, kw_only=True):
    """``src2dst[t * top_k + k]`` is ``expert * m_max + offset``. A pair is
    valid only when ``topk_ids[t, k] >= 0``. A provider that needs more
    per-forward state, such as a tile schedule, subclasses this.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden]
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int
    retained_inputs: bool


class MaskedRowDomainProvider(MoeBaseProvider):
    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        self.quant_info = quant_info
        if quant_info.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        expected_w2 = (
            quant_info.num_local_experts,
            quant_info.hidden_size,
            quant_info.intermediate_size,
        )
        if quant_info.w2_weight.shape != expected_w2:
            raise ValueError(
                f"w2_weight must be {expected_w2}, got "
                f"{tuple(quant_info.w2_weight.shape)}"
            )
        if (
            quant_info.w13_weight.ndim != 3
            or quant_info.w13_weight.shape[0] != quant_info.num_local_experts
            or quant_info.w13_weight.shape[2] != quant_info.hidden_size
        ):
            raise ValueError(
                "w13_weight must be [num_local_experts, slices*intermediate, hidden]"
            )
        gateup_width = quant_info.w13_weight.shape[1]
        if gateup_width % quant_info.intermediate_size:
            raise ValueError(
                "w13 output width must be an integer multiple of intermediate_size"
            )
        self._gate_up_slices = gateup_width // quant_info.intermediate_size
        if self._gate_up_slices not in (1, 2):
            raise ValueError(
                "masked BF16 provider supports one non-gated slice or two "
                f"gated gate/up slices, got {self._gate_up_slices}"
            )

        # This constructor runs once, when the LoRA attaches. These imports run
        # here so that no forward pass runs an import.
        from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
        from sglang.srt.lora.moe.base_gemm_provider.masked_activation import (
            act_delta_masked,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_dispatch import (
            fused_masked_preprocess,
        )

        self._preprocess = fused_masked_preprocess
        self._post_reorder = post_reorder_deepgemm
        self._act_kernel = act_delta_masked

        from sglang.srt.lora.moe.base_gemm_provider.down_b_into_base import (
            invoke_down_b_into_base,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_finalize import (
            MASKED_FINALIZE_TRITON,
            invoke_shared_from_scratch_finalize,
            invoke_shared_rank_reduce,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_fused_act import (
            MASKED_ACT_FAMILIES,
            MASKED_ACT_TRITON,
            run_masked_fused_act,
        )

        # A port can install another callable under a new name. The method
        # signatures do not change.
        self._fused_act_impls: dict[tuple[str, str, str], Callable] = {
            (family, activation, MASKED_ACT_TRITON): run_masked_fused_act
            for family in MASKED_ACT_FAMILIES
            for activation in ActivationFn
        }
        self._shared_reduce_impls: dict[str, Callable] = {
            MASKED_FINALIZE_TRITON: invoke_shared_rank_reduce
        }
        self._shared_tail_impls: dict[str, Callable] = {
            MASKED_FINALIZE_TRITON: invoke_shared_from_scratch_finalize
        }
        self._down_b_into_base = invoke_down_b_into_base

    @property
    def num_local_experts(self) -> int:
        return self.quant_info.num_local_experts

    @property
    def intermediate_size(self) -> int:
        return self.quant_info.intermediate_size

    @property
    def hidden_size(self) -> int:
        return self.quant_info.hidden_size

    @property
    def gate_up_slices(self) -> int:
        return self._gate_up_slices

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> MaskedRowState:
        m_max = (hidden_states.size(0) // 256 + 1) * 256
        masked_m_out = None
        src2dst_out = None
        hidden_permuted_out = None
        if workspace is not None:
            masked_m_out = workspace.tensor(
                "base:masked_m",
                (self.quant_info.num_local_experts,),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            src2dst_out = workspace.tensor(
                "base:src2dst",
                (topk_ids.numel(),),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            hidden_permuted_out = workspace.tensor(
                "base:hidden_permuted",
                (
                    self.quant_info.num_local_experts,
                    m_max,
                    hidden_states.size(1),
                ),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        masked_m, expected_m, src2dst, hidden_permuted, _scale = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            None,
            output_dtype=torch.bfloat16,
            masked_m_out=masked_m_out,
            src2dst_out=src2dst_out,
            gateup_input_out=hidden_permuted_out,
        )
        return MaskedRowState(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, row_state: MaskedRowState) -> None:
        # The gate/up GEMM is the last reader of the permuted rows. This frees
        # them before the next stage allocates. A workspace tensor must keep
        # its address for CUDA-graph replay, so this never frees one.
        if row_state.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_permuted)

    def act_with_delta(
        self,
        row_state: MaskedRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        # ``act_delta_masked`` checks the activation name against the registry,
        # so this method does not check it.
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.src2dst,
            topk_ids,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def mapped_down_lora_a_input(
        self,
        row_state: MaskedRowState,
        activation: torch.Tensor,
    ) -> MappedLoraAInput:
        if not isinstance(row_state, MaskedRowState):
            raise TypeError("masked down-A input requires MaskedRowState")
        expected = self.act_out_shape(row_state)
        if tuple(activation.shape) != expected:
            raise ValueError(
                f"mapped down-A activation must be {expected}, got "
                f"{tuple(activation.shape)}"
            )
        if activation.dtype != self.contract.lora_activation_dtype:
            raise TypeError(
                "mapped down-A activation dtype must match the provider "
                f"contract {self.contract.lora_activation_dtype}"
            )
        if not activation.is_contiguous():
            raise ValueError("mapped down-A activation rows must be contiguous")
        if (
            row_state.src2dst.ndim != 1
            or row_state.src2dst.dtype != torch.int32
            or row_state.src2dst.device != activation.device
            or not row_state.src2dst.is_contiguous()
        ):
            raise ValueError(
                "mapped down-A pair-to-row metadata must be contiguous 1-D "
                "int32 on the activation device"
            )
        return MappedLoraAInput(
            rows=activation.view(-1, activation.shape[-1]),
            pair_to_row=row_state.src2dst,
        )

    def install_fused_act_implementation(
        self,
        family: str,
        activation: str,
        name: str,
        implementation: Callable,
    ) -> None:
        if family != "b_activation":
            raise ValueError(f"unknown fused-act family {family!r}")
        if not name or not callable(implementation):
            raise ValueError("a fused-act implementation needs a name and callable")
        self._fused_act_impls[(family, activation, name)] = implementation

    def install_fused_finalize_implementation(
        self,
        family: str,
        ownership: str,
        name: str,
        implementation: Callable | tuple[Callable, Callable],
    ) -> None:
        if not name:
            raise ValueError("a fused-finalize implementation needs a name")
        if ownership not in ("per_expert", "shared"):
            raise ValueError(f"unknown fused-finalize ownership {ownership!r}")
        if family == "shared_rank_reduce":
            if ownership != "shared":
                raise ValueError("shared_rank_reduce requires shared ownership")
            if (
                not isinstance(implementation, tuple)
                or len(implementation) != 2
                or not all(callable(item) for item in implementation)
            ):
                raise ValueError(
                    "shared_rank_reduce implementation must be a "
                    "(reduce, tail) callable pair"
                )
            self._shared_reduce_impls[name], self._shared_tail_impls[name] = (
                implementation
            )
        else:
            raise ValueError(f"unknown fused-finalize family {family!r}")

    def fused_act_implementations(self, family: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    name
                    for candidate_family, _activation, name in self._fused_act_impls
                    if candidate_family == family
                }
            )
        )

    def supports_fused_act(
        self,
        family: str,
        *,
        activation: str,
        implementation: str = "triton",
    ) -> bool:
        return (family, activation, implementation) in self._fused_act_impls

    def _fused_act_implementation(
        self,
        family: str,
        activation: str,
        implementation: str,
    ) -> Callable:
        try:
            return self._fused_act_impls[(family, activation, implementation)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} masked-act "
                f"implementation for {family!r}/{activation!r}"
            ) from exc

    def run_fused_act(
        self,
        row_state: MaskedRowState,
        family: str,
        *,
        implementation: str,
        activation: str,
        base_gateup: torch.Tensor,
        act_masked: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
        consume_base_pdl: bool = False,
    ) -> None:
        invoke = self._fused_act_implementation(
            family,
            activation,
            implementation,
        )
        invoke(
            family,
            activation=activation,
            base_gateup=base_gateup,
            act_masked=act_masked,
            act_pairs=act_pairs,
            src2dst=row_state.src2dst,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            consume_base_pdl=consume_base_pdl,
        )

    def fused_finalize_implementations(
        self, family: str, ownership: str
    ) -> tuple[str, ...]:
        if family == "shared_rank_reduce" and ownership == "shared":
            return tuple(
                name
                for name in self._shared_reduce_impls
                if name in self._shared_tail_impls
            )
        return ()

    def supports_down_b_into_base(self) -> bool:
        return True

    def run_down_b_into_base(
        self,
        row_state: MaskedRowState,
        *,
        down_out: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        config: Mapping[str, int],
    ) -> None:
        self._down_b_into_base(
            down_rows=down_out.view(-1, self.hidden_size),
            src2dst=row_state.src2dst,
            bridge=bridge,
            b_down=b_down,
            routing=routing,
            config=config,
        )

    def run_shared_rank_finalize(
        self,
        row_state: MaskedRowState,
        *,
        implementation: str,
        down_masked: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        config: Mapping[str, Mapping[str, int]],
    ) -> None:
        self.run_shared_rank_reduce(
            row_state,
            implementation=implementation,
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            token_rank=token_rank,
            config=config["reduce"],
        )
        self.finish_shared_rank_finalize(
            row_state,
            implementation=implementation,
            down_masked=down_masked,
            b_down=b_down,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            token_rank=token_rank,
            config=config["tail"],
        )

    def run_shared_rank_reduce(
        self,
        row_state: MaskedRowState,
        *,
        implementation: str,
        bridge: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        try:
            invoke = self._shared_reduce_impls[implementation]
        except KeyError as exc:
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} shared-rank reduction"
            ) from exc
        # This launch reads pair data only. ``row_state`` stays in the
        # signature so that every stage takes the same arguments.
        del row_state
        invoke(
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            token_rank=token_rank,
            config=config,
        )

    def finish_shared_rank_finalize(
        self,
        row_state: MaskedRowState,
        *,
        implementation: str,
        down_masked: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        try:
            invoke = self._shared_tail_impls[implementation]
        except KeyError as exc:
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} shared B tail"
            ) from exc
        invoke(
            down_masked=down_masked,
            src2dst=row_state.src2dst,
            token_rank=token_rank,
            b_down=b_down,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            num_local_experts=self.num_local_experts,
            config=config,
        )

    def finalize(
        self,
        row_state: MaskedRowState,
        down_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        *,
        lora_delta: torch.Tensor | None = None,
    ) -> None:
        num_tokens, hidden = output.shape
        self._post_reorder(
            down_out.view(-1, hidden),
            output,
            row_state.src2dst,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            num_tokens,
            hidden,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            lora_delta=lora_delta,
        )

    def gateup_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.quant_info.hidden_size,
        )
