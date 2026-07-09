from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import (
    get_attention_dp_size,
    get_attention_tp_size,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DeepEPOutputDtype,
    get_deepep_output_dtype,
)
from sglang.srt.environ import envs

# Block size used by pplx-kernels for FP8 block-wise scales, matching the
# DeepSeek / DeepGEMM block quantization convention.
_FP8_BLOCK_SIZE = 128

try:
    from pplx_kernels import AllToAll, nvshmem_init
    from pplx_kernels.nvshmem import PyTorchStreamWrapper  # noqa: F401

    use_pplx = True
except ImportError:
    use_pplx = False


# ------------------------------ Dispatch / Combine formats ------------------
#
# pplx-kernels dispatch produces per-expert masked/batched tensors, which are
# semantically identical to DeepEP's low-latency ("LL") format. We therefore
# reuse DEEPEP_LL so the existing masked expert-compute + combine path
# (DeepEPMoE / moe_runner) handles pplx with no new permute registration.


class PplxDispatchOutput(NamedTuple):
    """PPLX EP dispatch output (masked / per-expert batched)."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(PplxDispatchOutput, DispatchOutput)


class PplxCombineInput(NamedTuple):
    """PPLX EP combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(PplxCombineInput, CombineInput)


# ------------------------------ NVSHMEM / AllToAll manager ------------------


class PplxAllToAllManager:
    """Process-wide owner of NVSHMEM init and the pplx-kernels ``AllToAll``.

    Analogous to ``DeepEPBuffer`` / ``EPBuffer``: it is a lazily-initialized
    singleton so the (expensive) NVSHMEM symmetric-memory workspace is
    allocated exactly once per process, keyed by the dispatch dtype (bf16 vs
    fp8-block-scale, which changes ``hidden_dim_scale_bytes``).
    """

    _nvshmem_initialized = False
    _all_to_all: Optional["AllToAll"] = None
    _key: Optional[tuple] = None
    _group_name: Optional[str] = None
    _combined_group: Optional[dist.ProcessGroup] = None

    # Name under which the EP process group is registered with c10d so the
    # pplx intranode kernel can resolve it via resolve_process_group().
    _GROUP_NAME = "pplx_ep"

    @classmethod
    def _ensure_nvshmem(cls, group: dist.ProcessGroup) -> None:
        if cls._nvshmem_initialized:
            return
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        # pplx-kernels resolves the local device from the current CUDA device.
        device = torch.device("cuda", torch.cuda.current_device())
        local_rank = torch.cuda.current_device()
        nvshmem_init(
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        cls._nvshmem_initialized = True

    @classmethod
    def _register_group(cls, group: dist.ProcessGroup) -> str:
        """Register a combined-backend EP group with c10d and return its name.

        The pplx intranode kernel resolves the process group by name via
        ``c10d::resolve_process_group`` and drives ``alltoall_base`` on it to
        exchange CUDA IPC handles (CPU tensors) as well as device data. It
        therefore needs a single group carrying BOTH a CPU (gloo) and a CUDA
        (nccl) backend. SGLang's EP ``device_group`` is nccl-only, so we build
        a fresh ``cpu:gloo,cuda:nccl`` group over the same ranks and register
        that. Must be called collectively by all ranks of the enclosing world
        group (``new_group`` is a collective).
        """
        if cls._group_name is not None:
            return cls._group_name
        ranks = dist.get_process_group_ranks(group)
        combined = dist.new_group(ranks=ranks, backend="cpu:gloo,cuda:nccl")
        torch._C._distributed_c10d._register_process_group(cls._GROUP_NAME, combined)
        cls._group_name = cls._GROUP_NAME
        cls._combined_group = combined
        return cls._group_name

    @classmethod
    def barrier(cls) -> None:
        """Barrier across the EP group to keep ranks in lockstep.

        pplx dispatch/combine share one fixed-buffer AllToAll instance across
        all MoE layers. The calls enqueue asynchronously, so without a
        per-layer rendezvous a fast (e.g. idle DP) rank can run ahead and
        overwrite the shared counter/data buffers that a slower rank's in-flight
        collective is still reading, deadlocking the device-side handshake. A
        device barrier at each layer boundary forces all ranks past layer K
        before any starts layer K+1.
        """
        if cls._combined_group is not None:
            dist.barrier(group=cls._combined_group, device_ids=[torch.cuda.current_device()])

    @classmethod
    def get_all_to_all(
        cls,
        group: dist.ProcessGroup,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
    ) -> "AllToAll":
        world_size = group.size()
        rank = group.rank()
        # pplx dpSize == number of ranks per DP group == attention TP size.
        # numDPGroups == worldSize / dpSize == attention DP size (must be > 1).
        dp_size = get_attention_tp_size()

        key = (
            max_num_tokens,
            num_experts,
            experts_per_token,
            hidden_dim,
            hidden_dim_bytes,
            hidden_dim_scale_bytes,
            world_size,
            dp_size,
        )
        if cls._all_to_all is not None:
            assert cls._key == key, (
                "PplxAllToAllManager already initialized with a different "
                f"configuration: {cls._key} != {key}"
            )
            return cls._all_to_all

        cls._ensure_nvshmem(group)

        # Use the single-node NVLink path when the EP group fits on one node,
        # otherwise the NVSHMEM internode path.
        is_internode = world_size > torch.cuda.device_count()

        if is_internode:
            cls._all_to_all = AllToAll.internode(
                max_num_tokens=max_num_tokens,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                rank=rank,
                world_size=world_size,
                dp_size=dp_size,
                hidden_dim=hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=hidden_dim_scale_bytes,
            )
        else:
            group_name = cls._register_group(group)
            cls._all_to_all = AllToAll.intranode(
                max_num_tokens=max_num_tokens,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                rank=rank,
                world_size=world_size,
                dp_size=dp_size,
                hidden_dim=hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=hidden_dim_scale_bytes,
                group_name=group_name,
            )
        cls._key = key
        return cls._all_to_all


# ------------------------------ Dispatcher implementation -------------------


class _PplxEPDispatcherImpl:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_pplx:
            raise ImportError(
                "pplx-kernels is not installed. Please build and install it "
                "from https://github.com/perplexityai/pplx-kernels (e.g. "
                "`TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel && "
                "pip install dist/*.whl`) to run SGLang with the pplx MoE A2A "
                "backend."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.params_bytes = 2
        self.deepep_mode = deepep_mode

        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

        # Resolved lazily from the quant config (see set_dispatch_dtype).
        self.quant_config: dict = {}
        self.use_fp8 = False
        self.set_dispatch_dtype()

    def set_dispatch_dtype(self) -> None:
        output_dtype = get_deepep_output_dtype(self)
        if output_dtype == DeepEPOutputDtype.BF16:
            self.use_fp8 = False
        elif output_dtype == DeepEPOutputDtype.FP8:
            self.use_fp8 = True
        else:
            raise NotImplementedError(
                f"pplx MoE A2A backend does not support dispatch dtype "
                f"{output_dtype}; use bf16 or fp8."
            )

    def _hidden_dim_scale_bytes(self) -> int:
        if not self.use_fp8:
            return 0
        return (
            (self.hidden_size + _FP8_BLOCK_SIZE - 1)
            // _FP8_BLOCK_SIZE
            * torch.float32.itemsize
        )

    def _get_all_to_all(self) -> "AllToAll":
        itemsize = 1 if self.use_fp8 else self.params_bytes
        return PplxAllToAllManager.get_all_to_all(
            group=self.group,
            max_num_tokens=self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts,
            experts_per_token=self.router_topk,
            hidden_dim=self.hidden_size,
            hidden_dim_bytes=self.hidden_size * itemsize,
            hidden_dim_scale_bytes=self._hidden_dim_scale_bytes(),
        )

    def _quantize(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (dp_x, dp_x_scale) matching the pplx dispatch contract."""
        if not self.use_fp8:
            return hidden_states, None

        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        x_q, x_s = sglang_per_token_group_quant_fp8(
            hidden_states,
            group_size=_FP8_BLOCK_SIZE,
        )
        # pplx expects float32 scales.
        return x_q, x_s.to(torch.float32)

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        ata = self._get_all_to_all()

        num_tokens = hidden_states.shape[0]
        num_dp_groups = get_attention_dp_size()
        max_batch_tokens = self.num_max_dispatch_tokens_per_rank * num_dp_groups
        device = hidden_states.device

        dp_x, dp_x_scale = self._quantize(hidden_states)

        # pplx pre-allocates dispatch outputs (masked / per-expert batched).
        # Zero-init: unwritten padding slots (beyond masked_m per expert) must
        # be zero so the expert GEMM + combine never read uninitialized memory.
        out_expert_num_tokens = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device=device
        )
        out_expert_x = torch.zeros(
            (self.num_local_experts, max_batch_tokens, self.hidden_size),
            dtype=dp_x.dtype,
            device=device,
        )
        out_expert_x_scale = None
        if self.use_fp8:
            scale_dim = self._hidden_dim_scale_bytes() // torch.float32.itemsize
            out_expert_x_scale = torch.empty(
                (self.num_local_experts, max_batch_tokens, scale_dim),
                dtype=torch.float32,
                device=device,
            )

        # bound_m carries the token count. Build it with torch.full (a
        # scalar-fill kernel) rather than torch.tensor([...]) so it is
        # CUDA-graph-capturable (a host->device copy of a Python list is not).
        bound_m = torch.full((1,), num_tokens, dtype=torch.uint32, device=device)
        indices = topk_ids.to(torch.uint32)

        ata.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            bound_m=bound_m,
        )

        expected_m = (
            num_tokens * num_dp_groups * self.router_topk + self.num_experts
        ) // self.num_experts

        return (
            out_expert_x,
            out_expert_x_scale,
            topk_ids,
            topk_weights,
            out_expert_num_tokens,
            expected_m,
        )

    def dispatch_b(
        self,
        out_expert_x,
        out_expert_x_scale,
        topk_ids,
        topk_weights,
        out_expert_num_tokens,
        expected_m,
    ):
        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            out_expert_num_tokens
        )
        return PplxDispatchOutput(
            out_expert_x,
            out_expert_x_scale,
            topk_ids,
            topk_weights,
            out_expert_num_tokens,
            expected_m,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        ata = self._get_all_to_all()
        num_tokens = topk_ids.shape[0]
        device = topk_ids.device

        out_tokens = torch.zeros(
            (self.num_max_dispatch_tokens_per_rank, self.hidden_size),
            dtype=self.params_dtype,
            device=device,
        )
        # torch.full (scalar-fill kernel) is CUDA-graph-capturable; a
        # torch.tensor([...]) host->device copy is not.
        bound_m = torch.full((1,), num_tokens, dtype=torch.uint32, device=device)

        ata.combine(
            out_tokens=out_tokens,
            indices=topk_ids.to(torch.uint32),
            weights=topk_weights.to(torch.float32),
            expert_y=hidden_states,
            bound_m=bound_m,
        )
        # pplx.combine applies routing weights internally; slice to real tokens.
        return (out_tokens[:num_tokens],)

    def combine_b(self, hidden_states):
        return hidden_states

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_dispatch_dtype()
        # Build the NVSHMEM workspace eagerly (during weight post-processing,
        # in eager mode) so nvshmem_init's CPU collectives do not run inside
        # CUDA graph capture, which fails with "No backend type associated with
        # device type cpu". The dispatch dtype is now known.
        self._get_all_to_all()


# ------------------------------ Public dispatcher ---------------------------


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class PplxEPDispatcher(BaseDispatcher):
    """MoE all-to-all dispatcher backed by Perplexity's pplx-kernels.

    Low-latency (masked) only, mirroring ``MooncakeEPDispatcher``. Emits the
    DEEPEP_LL dispatch/combine format so the existing masked expert-compute
    path is reused unchanged.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        if self.deepep_mode.enable_normal():
            raise NotImplementedError(
                "pplx MoE A2A backend supports low-latency mode only."
            )

        self._low_latency_dispatcher = _PplxEPDispatcherImpl(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        self._stage = _Stage.INITIAL

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self._low_latency_dispatcher.set_quant_config(quant_config)

    def _get_impl(self) -> _PplxEPDispatcherImpl:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            raise NotImplementedError(
                "pplx MoE A2A backend supports low-latency mode only."
            )
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage
