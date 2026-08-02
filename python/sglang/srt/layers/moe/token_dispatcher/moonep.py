from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, NoReturn, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode


_MOONEP_UNSUPPORTED_MESSAGE = (
    "MoonEP MoE A2A is recognized by SGLang, but the runtime dispatcher is not "
    "implemented yet. MoonEP is not a drop-in DeepEP-compatible backend: it "
    "returns a MoonEPCommPlan/cu_seqlens and requires MoonEP-compatible "
    "contiguous symmetric-memory expert weights plus a VM-group expert GEMM."
)


class MoonEPDispatchOutput(NamedTuple):
    """MoonEP dispatch output.

    ``plan`` is intentionally typed as ``Any`` so this module can define the
    SGLang-side contract without importing the optional ``moonep`` package at
    module import time.
    """

    hidden_states: torch.Tensor
    route_weights_nvs: Optional[torch.Tensor]
    cu_seqlens: torch.Tensor
    plan: Any

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MOONEP


assert isinstance(MoonEPDispatchOutput, DispatchOutput)


class MoonEPCombineInput(NamedTuple):
    """MoonEP combine input."""

    hidden_states: torch.Tensor
    route_weights_nvs: Optional[torch.Tensor]
    plan: Any

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MOONEP


assert isinstance(MoonEPCombineInput, CombineInput)


@dataclass(frozen=True)
class MoonEPBufferKey:
    """Static MoonEP buffer dimensions.

    MoonEP allocates its communication buffers from static shape parameters,
    unlike DeepEP's normal-dispatch path.  Keep the dimensions explicit so the
    process-wide facade never reuses a buffer with incompatible token capacity,
    model shape, EP topology, or prefetch-slot layout.
    """

    num_max_dispatch_tokens_per_rank: int
    hidden_size: int
    router_topk: int
    num_experts: int
    num_ep_ranks: int
    group_id: int
    num_prefetch_slots: int
    token_padding: int
    num_sms: int


class MoonEPBuffer:
    """Process-wide facade for MoonEP communication buffers.

    The underlying ``moonep.Buffer`` owns NVLink/VMM allocations and is keyed by
    MoonEP's static allocation dimensions.  The state lives on
    ``ctx.resources.buffers`` so tests can reset it with ``reset_context()`` and
    future runtime code has one lifecycle hook per process.
    """

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        state = buffers.get("moonep_ep_state")
        if state is None:
            state = SimpleNamespace(
                buffers={},
                active_key=None,
            )
            buffers["moonep_ep_state"] = state
        return state

    @staticmethod
    def _require_positive_int(name: str, value: int) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _resolve_num_ep_ranks(group: dist.ProcessGroup) -> int:
        try:
            num_ep_ranks = dist.get_world_size(group=group)
        except (AssertionError, RuntimeError, TypeError, ValueError):
            group_size = getattr(group, "size", None)
            if not callable(group_size):
                raise
            num_ep_ranks = group_size()
        return MoonEPBuffer._require_positive_int("num_ep_ranks", int(num_ep_ranks))

    @staticmethod
    def _resolve_num_prefetch_slots(
        num_prefetch_slots: int | None,
        num_experts: int,
        num_ep_ranks: int,
    ) -> int:
        if num_experts % num_ep_ranks != 0:
            raise ValueError(
                "MoonEP requires num_experts to be divisible by the EP group size: "
                f"num_experts={num_experts}, num_ep_ranks={num_ep_ranks}"
            )

        if num_prefetch_slots is None:
            num_prefetch_slots = envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.get()
        num_prefetch_slots = int(num_prefetch_slots)
        if num_prefetch_slots <= 0:
            return num_experts // num_ep_ranks
        return MoonEPBuffer._require_positive_int(
            "num_prefetch_slots", num_prefetch_slots
        )

    @classmethod
    def build_key(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int | None = None,
        num_prefetch_slots: int | None = None,
        token_padding: int | None = None,
        num_sms: int | None = None,
    ) -> MoonEPBufferKey:
        if num_max_dispatch_tokens_per_rank is None:
            num_max_dispatch_tokens_per_rank = (
                envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
            )
        if token_padding is None:
            token_padding = envs.SGLANG_MOONEP_TOKEN_PADDING.get()
        if num_sms is None:
            num_sms = envs.SGLANG_MOONEP_NUM_SMS.get()

        num_ep_ranks = cls._resolve_num_ep_ranks(group)
        num_experts = cls._require_positive_int("num_experts", int(num_experts))
        num_prefetch_slots = cls._resolve_num_prefetch_slots(
            num_prefetch_slots,
            num_experts,
            num_ep_ranks,
        )

        return MoonEPBufferKey(
            num_max_dispatch_tokens_per_rank=cls._require_positive_int(
                "num_max_dispatch_tokens_per_rank",
                int(num_max_dispatch_tokens_per_rank),
            ),
            hidden_size=cls._require_positive_int("hidden_size", int(hidden_size)),
            router_topk=cls._require_positive_int("router_topk", int(router_topk)),
            num_experts=num_experts,
            num_ep_ranks=num_ep_ranks,
            group_id=id(group),
            num_prefetch_slots=num_prefetch_slots,
            token_padding=cls._require_positive_int(
                "token_padding", int(token_padding)
            ),
            num_sms=cls._require_positive_int("num_sms", int(num_sms)),
        )

    @classmethod
    def get_existing_buffer(
        cls,
        key: MoonEPBufferKey | None = None,
    ):
        """Return an already-created buffer, if any.

        Without a key this returns the most recently requested MoonEP buffer.
        Future runtime code should pass an explicit key when multiple static
        capacities are in play.
        """

        state = cls._state()
        if key is None:
            key = state.active_key
        if key is None:
            return None
        return state.buffers.get(key)

    @classmethod
    def get_moonep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int | None = None,
        num_prefetch_slots: int | None = None,
        token_padding: int | None = None,
        num_sms: int | None = None,
    ):
        key = cls.build_key(
            group=group,
            hidden_size=hidden_size,
            router_topk=router_topk,
            num_experts=num_experts,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_prefetch_slots=num_prefetch_slots,
            token_padding=token_padding,
            num_sms=num_sms,
        )

        state = cls._state()
        buffer = state.buffers.get(key)
        if buffer is not None:
            state.active_key = key
            return buffer

        try:
            from moonep import Buffer
        except ImportError as exc:
            raise ImportError(
                "MoonEP is not installed. Install MoonEP before running SGLang "
                "with --moe-a2a-backend moonep."
            ) from exc

        buffer = Buffer(
            S=key.num_max_dispatch_tokens_per_rank,
            H=key.hidden_size,
            K=key.router_topk,
            E=key.num_experts,
            num_ep_ranks=key.num_ep_ranks,
            num_sms=key.num_sms,
            token_padding=key.token_padding,
            B=key.num_prefetch_slots,
            group=group,
        )
        state.buffers[key] = buffer
        state.active_key = key
        return buffer

    @classmethod
    def destroy_buffer(cls, key: MoonEPBufferKey | None = None) -> None:
        state = cls._state()
        if key is None:
            key = state.active_key
        if key is None:
            return

        buffer = state.buffers.pop(key, None)
        destroy = getattr(buffer, "destroy", None)
        if callable(destroy):
            destroy()
        if state.active_key == key:
            state.active_key = next(reversed(state.buffers), None)

    @classmethod
    def destroy_all_buffers(cls) -> None:
        state = cls._state()
        for key in list(state.buffers):
            cls.destroy_buffer(key)
        state.active_key = None


class MoonEPDispatcher(BaseDispatcher):
    """Placeholder dispatcher for MoonEP.

    This keeps backend selection explicit while preventing accidental execution
    through the DeepEP/Mooncake/NIXL dispatcher contracts, which use different
    dispatch output formats from MoonEP.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.async_finish = async_finish
        self.return_recv_hook = return_recv_hook
        self.expert_mask_gpu = None

    @staticmethod
    def _raise_unimplemented() -> NoReturn:
        raise NotImplementedError(_MOONEP_UNSUPPORTED_MESSAGE)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self._raise_unimplemented()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._raise_unimplemented()

    def dispatch_b(self):
        self._raise_unimplemented()

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self._raise_unimplemented()

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        self._raise_unimplemented()

    def combine_b(self):
        self._raise_unimplemented()

    def register_deepep_dispatch_hook(self, hook):
        self._raise_unimplemented()
