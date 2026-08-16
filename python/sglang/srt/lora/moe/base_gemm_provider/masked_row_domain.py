"""The masked-row-domain half of a BF16 MoE provider, GEMM-engine-agnostic.

S1 preprocess (the engine-local `fused_masked_preprocess`), the S3 activation
join (`silu_mul_delta_masked`), and the S5 finalize (`post_reorder_deepgemm`)
are Triton kernels over one physical layout — rows in ``[E_local, m_max,·]``
with ``masked_m`` counts and ``src2dst`` pair mapping — and carry nothing
specific to any GEMM engine. Both shipped providers (DeepGEMM and CuTeDSL)
consume exactly this layout and differ ONLY in how S2/S4 are executed, so the
domain lives here once and each engine subclass implements just ``gateup`` and
``down``.

Extracted per Yanbin's review (plan section 51): before this, the CuTeDSL
provider inherited from the DeepGEMM provider, which read as "CuTeDSL depends
on DeepGEMM" when the true relationship is "both specialize the masked row
domain".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MappedLoraAInput,
    MoeBaseProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.routing import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class MaskedRowWorkspace(msgspec.Struct, kw_only=True):
    """Per-forward state of the masked row domain.

    Rows are ``[E_local, m_max, ·]`` with
    ``src2dst[t * top_k + k] = expert * m_max + offset``; validity is carried by
    ``topk_ids >= 0`` for the activation join and final reordering. Providers
    with extra per-forward state (e.g. tile schedules) subclass this.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden]
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int
    retained_inputs: bool


class MaskedRowDomainProvider(MoeBaseProvider):
    """S1/S3/S5 plus geometry over the masked row domain; S2/S4 stay abstract."""

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
                "masked BF16 provider supports one ReLU2 slice or two "
                f"gate/up slices, got {self._gate_up_slices}"
            )

        # Bind callees once: this instance is constructed at LoRA-attach time
        # and lives for the layer's lifetime, so no per-forward imports.
        from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
        from sglang.srt.lora.moe.base_gemm_provider.masked_activation import (
            silu_mul_delta_masked,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_dispatch import (
            fused_masked_preprocess,
        )

        self._preprocess = fused_masked_preprocess
        self._post_reorder = post_reorder_deepgemm
        self._act_kernel = silu_mul_delta_masked

        from sglang.srt.lora.moe.base_gemm_provider.down_b_scatter import (
            invoke_down_b_scatter,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_finalize import (
            MASKED_FINALIZE_TRITON,
            invoke_shared_from_scratch_finalize,
            invoke_shared_rank_reduce,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_fused_middle import (
            MASKED_MIDDLE_ACTIVATIONS,
            MASKED_MIDDLE_FAMILIES,
            MASKED_MIDDLE_TRITON,
            run_masked_fused_middle,
        )

        # Named, forceable implementations. A provider-specific CuTe port can
        # inject another callable without changing the semantic method ABI.
        self._fused_middle_impls: dict[tuple[str, str, str], Callable] = {
            (family, activation, MASKED_MIDDLE_TRITON): run_masked_fused_middle
            for family in MASKED_MIDDLE_FAMILIES
            for activation in MASKED_MIDDLE_ACTIVATIONS
        }
        self._shared_reduce_impls: dict[str, Callable] = {
            MASKED_FINALIZE_TRITON: invoke_shared_rank_reduce
        }
        self._shared_tail_impls: dict[str, Callable] = {
            MASKED_FINALIZE_TRITON: invoke_shared_from_scratch_finalize
        }
        self._down_b_scatter = invoke_down_b_scatter

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
    ) -> MaskedRowWorkspace:
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
        return MaskedRowWorkspace(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, ws: MaskedRowWorkspace) -> None:
        # The permuted hidden rows are dead after the gate/up GEMM; free them
        # before the S3/S4 buffers are allocated when this provider owns the
        # allocation. Runner-workspace inputs remain address-stable for graph
        # replay and are reclaimed with the runner workspace.
        if ws.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(ws.hidden_permuted)

    def act_with_delta(
        self,
        ws: MaskedRowWorkspace,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        if activation not in ("silu", "relu2"):
            raise ValueError(f"activation={activation!r} is not 'silu' or 'relu2'")
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            ws.src2dst,
            topk_ids,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def mapped_down_lora_a_input(
        self,
        ws: MaskedRowWorkspace,
        activation: torch.Tensor,
    ) -> MappedLoraAInput:
        """Expose masked activation rows without leaking workspace internals.

        The mapping is the provider's semantic pair-to-physical-row ABI.  The
        runner consumes only this descriptor and never reaches into ``ws``.
        """

        if not isinstance(ws, MaskedRowWorkspace):
            raise TypeError("masked down-A input requires MaskedRowWorkspace")
        expected = self.act_out_shape(ws)
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
            ws.src2dst.ndim != 1
            or ws.src2dst.dtype != torch.int32
            or ws.src2dst.device != activation.device
            or not ws.src2dst.is_contiguous()
        ):
            raise ValueError(
                "mapped down-A pair-to-row metadata must be contiguous 1-D "
                "int32 on the activation device"
            )
        return MappedLoraAInput(
            rows=activation.view(-1, activation.shape[-1]),
            pair_to_row=ws.src2dst,
        )

    def install_fused_middle_implementation(
        self,
        family: str,
        activation: str,
        name: str,
        implementation: Callable,
    ) -> None:
        """Inject an explicitly forceable provider-local implementation."""
        if family != "b_activation":
            raise ValueError(f"unknown fused-middle family {family!r}")
        if activation not in ("silu", "relu2"):
            raise ValueError(f"unknown fused-middle activation {activation!r}")
        if not name or not callable(implementation):
            raise ValueError("a fused-middle implementation needs a name and callable")
        self._fused_middle_impls[(family, activation, name)] = implementation

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

    def fused_middle_implementations(self, family: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    name
                    for candidate_family, _activation, name in self._fused_middle_impls
                    if candidate_family == family
                }
            )
        )

    def supports_fused_middle(
        self,
        family: str,
        *,
        activation: str,
        implementation: str = "triton",
    ) -> bool:
        return (family, activation, implementation) in self._fused_middle_impls

    def _fused_middle_implementation(
        self,
        family: str,
        activation: str,
        implementation: str,
    ) -> Callable:
        try:
            return self._fused_middle_impls[(family, activation, implementation)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} masked-middle "
                f"implementation for {family!r}/{activation!r}"
            ) from exc

    def run_fused_middle(
        self,
        ws: MaskedRowWorkspace,
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
        invoke = self._fused_middle_implementation(
            family,
            activation,
            implementation,
        )
        if activation not in ("silu", "relu2"):
            raise ValueError(f"activation={activation!r} is not 'silu' or 'relu2'")
        invoke(
            family,
            activation=activation,
            base_gateup=base_gateup,
            act_masked=act_masked,
            act_pairs=act_pairs,
            src2dst=ws.src2dst,
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

    def supports_down_b_scatter(self) -> bool:
        return True

    def run_down_b_scatter(
        self,
        ws: MaskedRowWorkspace,
        *,
        down_out: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        config: Mapping[str, int],
    ) -> None:
        # Same row-domain lever as post_reorder: rows are
        # addressed only through src2dst over the flat [rows, H] view (masked
        # rows e * m_max + slot).  src2dst is only READ, so the documented
        # in-place-src2dst-store hazard does not apply.
        self._down_b_scatter(
            down_rows=down_out.view(-1, self.hidden_size),
            src2dst=ws.src2dst,
            bridge=bridge,
            b_down=b_down,
            routing=routing,
            config=config,
        )

    def run_shared_rank_finalize(
        self,
        ws: MaskedRowWorkspace,
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
        if set(config) != {"reduce", "tail"}:
            raise ValueError("shared-rank config must contain exactly reduce and tail")
        self.run_shared_rank_reduce(
            ws,
            implementation=implementation,
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            token_rank=token_rank,
            config=config["reduce"],
        )
        self.finish_shared_rank_finalize(
            ws,
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
        ws: MaskedRowWorkspace,
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
        # `ws` is deliberately opaque and unused by this pair-domain launch;
        # retaining it in the provider ABI lets every scheduled stage be
        # invoked uniformly.
        del ws
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
        ws: MaskedRowWorkspace,
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
            src2dst=ws.src2dst,
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
        ws: MaskedRowWorkspace,
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
            ws.src2dst,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            num_tokens,
            hidden,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            lora_delta=lora_delta,
        )

    def gateup_out_shape(self, ws: MaskedRowWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, ws: MaskedRowWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, ws: MaskedRowWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.hidden_size,
        )
