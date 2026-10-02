"""NCCL EP low-latency MoE dispatch/combine backend (``--moe-a2a-backend nccl_ep``).

Mirrors the DeepEP LL contract (DEEPEP_LL output, reuses DeepEPMoE.compute).
NCCL EP carries bf16 on the wire, so we quantize the bf16 recv to fp8 (group 128)
after dispatch. LL (decode) only; HT (prefill) is a follow-up.
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import nullcontext
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPPDispatchHooks,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    get_nccl_ep_layout,
    get_nccl_ep_mode,
    get_nccl_ep_num_max_dispatch_tokens_per_rank,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.moe.fused_moe_triton.layer import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutput

logger = logging.getLogger(__name__)

# NCCL EP LL dispatch accepts only these hidden sizes (hardcoded in SWITCH_HIDDEN)
# and requires hidden % 128 == 0.
_NCCL_EP_LL_SUPPORTED_HIDDEN = {2048, 2560, 4096, 5120, 6144, 7168, 8192}

# LL top-k upper bound (kNumMaxTopK). Models with topk>9 (e.g. Nemotron Super)
# fall back to DeepEP LL — upstream issue #2103.
_NCCL_EP_LL_MAX_TOPK = 9

# Mirror DeepEP's per-rank dispatch budget cap.
_NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP = 1024
_NCCL_EP_DEFAULT_MAX_DISPATCH_TOKENS_PER_RANK = 1024

# fp8 per-token-group quant block size for the LL GEMM path (cutlass w4a8 moe).
_NCCL_EP_FP8_GROUP_SIZE = 128


class NcclEpRankMajorDispatchOutput(NamedTuple):
    """Packed local-expert compute view of a NCCL EP rank-major receive.

    NCCL itself writes BF16 ``[world, max_dispatch, hidden]`` slots.  The
    dispatcher turns those into the existing masked expert-major FP8 compute
    layout, while retaining the source-slot mapping needed to pre-reduce
    locally before NCCL's rank-major combine.
    """

    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int
    src2dst: torch.Tensor
    local_topk_ids: torch.Tensor
    recv_topk_weights: torch.Tensor
    slot_shape: tuple[int, int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL_EP_RANK_MAJOR


class NcclEpRankMajorCombineInput(NamedTuple):
    """Local expert outputs plus the metadata needed for RM pre-reduction."""

    hidden_states: torch.Tensor
    src2dst: torch.Tensor
    local_topk_ids: torch.Tensor
    recv_topk_weights: torch.Tensor
    slot_shape: tuple[int, int]

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.NCCL_EP_RANK_MAJOR


def rank_major_local_expert_ids(
    recv_topk_ids: torch.Tensor, num_local_experts: int
) -> torch.Tensor:
    """Validate NCCL RM's local expert ids, preserving -1 padding."""
    return torch.where(
        (recv_topk_ids >= 0) & (recv_topk_ids < num_local_experts),
        recv_topk_ids,
        torch.full_like(recv_topk_ids, -1),
    ).to(torch.int32)


def reduce_rank_major_expert_outputs(
    expert_outputs: torch.Tensor,
    src2dst: torch.Tensor,
    local_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    slot_shape: tuple[int, int],
    *,
    route_buffer: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight and reduce local expert rows into NCCL rank-major combine slots."""
    num_slots = slot_shape[0] * slot_shape[1]
    topk = local_topk_ids.shape[-1]
    if local_topk_ids.numel() != num_slots * topk:
        raise ValueError("rank-major metadata does not match the receive slot shape")
    if expert_outputs.is_cuda:
        if output is None:
            output = torch.empty(
                (*slot_shape, expert_outputs.shape[-1]),
                dtype=expert_outputs.dtype,
                device=expert_outputs.device,
            )
        from sglang.kernels.ops.moe.ep_moe_kernels import rank_major_weighted_reduce

        return rank_major_weighted_reduce(
            expert_outputs,
            src2dst,
            local_topk_ids,
            recv_topk_weights,
            output,
            slot_shape,
        )

    # CPU reference implementation, used by unit tests.
    flat_output = expert_outputs.flatten(0, 1)
    valid = local_topk_ids.reshape(-1) >= 0
    route_shape = (num_slots * topk, expert_outputs.shape[-1])
    if route_buffer is None:
        routes = torch.zeros(
            route_shape, dtype=expert_outputs.dtype, device=expert_outputs.device
        )
    else:
        if tuple(route_buffer.shape) != route_shape:
            raise ValueError("rank-major route scratch has the wrong shape")
        routes = route_buffer
        routes.zero_()
    routes[valid] = flat_output[src2dst[valid].to(torch.long)]
    # NCCL's rank-major combine accumulates each rank contribution in fp32.
    # Keep that precision here; casting router weights to bf16 before the
    # local reduction is enough to diverge from the expert-major path at
    # greedy decode.
    weights = recv_topk_weights.reshape(-1, 1).float()
    reduced = (
        (routes.float() * weights)
        .reshape(num_slots, topk, -1)
        .sum(dim=1)
        .to(expert_outputs.dtype)
        .reshape(*slot_shape, -1)
    )
    if output is not None:
        if tuple(output.shape) != tuple(reduced.shape):
            raise ValueError("rank-major pre-reduce scratch has the wrong shape")
        output.copy_(reduced)
        return output
    return reduced


def _parse_ver_str(s: str):
    """Parse the leading ``X.Y.Z`` from a version string into a 3-tuple, or None."""
    parts = s.split(".")
    try:
        return tuple(int(x) for x in parts[:3])
    except ValueError:
        return None


def _nccl_runtime_version():
    """Return the linked NCCL version as a (major, minor, patch) tuple, or None.

    Prefers nccl4py's version query (may differ from torch's bundled NCCL via
    LD_PRELOAD); falls back to torch.
    """
    try:
        import nccl.core as nccl_core

        vi = nccl_core.get_version()
        lib_info = getattr(vi, "libnccl", None)
        ver_obj = getattr(lib_info, "version", None) if lib_info is not None else None
        if ver_obj is not None:
            return _parse_ver_str(str(ver_obj))
    except Exception:
        pass
    try:
        v = torch.cuda.nccl.version()
        if isinstance(v, (tuple, list)):
            return tuple(int(x) for x in v[:3])
    except Exception:
        pass
    return None


def _nccl_ep_scale_unavailable_reason() -> str | None:
    from sglang.srt.layers import deep_gemm_wrapper

    if deep_gemm_wrapper.DEEPGEMM_BLACKWELL:
        return (
            "NCCL EP LL emits float32 FP8 group scales and does not support "
            "UE8M0 scales required by DEEPGEMM_BLACKWELL"
        )
    return None


def nccl_ep_unavailable_reason(*, require_graph: bool = False) -> str | None:
    """Return why NCCL EP is unavailable (None = available), for fallback logs."""
    try:
        installed = importlib.util.find_spec("nccl.ep") is not None
    except (ModuleNotFoundError, ValueError):
        installed = False
    if not installed:
        return "nccl4py (nccl.ep) not importable"
    if not torch.cuda.is_available():
        return "no CUDA device"
    cc = torch.cuda.get_device_capability(0)[0]  # Hopper (sm_90) or Blackwell.
    if cc < 9:
        return f"GPU arch sm_{cc}x not supported (need Hopper/Blackwell sm_90+)"
    if reason := _nccl_ep_scale_unavailable_reason():
        return reason
    ver = _nccl_runtime_version()
    if ver is None:
        return "could not read NCCL version"
    if (ver[0], ver[1]) < (2, 29):
        return f"NCCL {'.'.join(map(str, ver))} < 2.29 (need Device API/GIN)"
    if require_graph:
        try:
            _, ep = _load_nccl_ep()
        except ImportError as error:
            return f"could not load nccl.ep: {error}"
        if not callable(getattr(ep.Handle, "update", None)):
            return "the installed nccl.ep binding lacks Handle.update"
    return None


def is_nccl_ep_available() -> bool:
    """Whether nccl4py + NCCL >= 2.29 + Hopper/Blackwell are usable here."""
    return nccl_ep_unavailable_reason() is None


# ----------------------------- Buffer (Group/Handle lifecycle) -----------------------------

_nccl_ep_runtime = None  # cached import of nccl.ep / nccl.core


def _load_nccl_ep():
    global _nccl_ep_runtime
    if _nccl_ep_runtime is not None:
        return _nccl_ep_runtime
    import nccl.core as nccl_core
    import nccl.ep as nccl_ep

    _nccl_ep_runtime = (nccl_core, nccl_ep)
    return _nccl_ep_runtime


class NcclEpBuffer:
    """Process-wide NCCL EP group + static recv buffers (mirrors DeepEPBuffer).

    State lives on ``ctx.resources.buffers["nccl_ep_state"]`` and, for TBO,
    ``nccl_ep_state_1``. Each lane creates its group once from the EP
    communicator and retains its own scratch and persistent eager handles.
    """

    @classmethod
    def _state(cls, instance_id=0):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_resources

        if instance_id not in (0, 1):
            raise ValueError("NCCL EP supports two TBO subbatches")
        buffers = get_resources().buffers
        key = "nccl_ep_state" if instance_id == 0 else "nccl_ep_state_1"
        state = buffers.get(key)
        if state is None:
            state = SimpleNamespace(
                group=None,
                dispatchers=set(),
                borrower=None,
                configuration=None,
                num_experts=None,
                num_local_experts=None,
                nccl_num_local_experts=None,
                hidden_size=None,
                max_dispatch_tokens_per_rank=None,
                max_recv_tokens_per_rank=None,
                router_topk=None,
                layout=None,
                # Pre-allocated scratch (fixed shapes); reused per dispatch/combine step.
                recv_tokens=None,
                expert_counters=None,
                expert_offsets=None,
                recv_total=None,
                combined=None,
                # Rank-major low-latency receive scratch. Kept separate from
                # expert-major buffers so the default path is byte-for-byte
                # unchanged and a graph handle never changes layout in place.
                rm_recv_tokens=None,
                rm_recv_topk_ids=None,
                rm_recv_topk_weights=None,
                rm_src_rank_counters=None,
                rm_route_buffer=None,
                rm_pre_reduced=None,
                rm_combined=None,
            )
            buffers[key] = state
        return state

    @classmethod
    def get_buffer(
        cls,
        ep_group: GroupCoordinator,
        hidden_size: int,
        num_experts: int,
        num_local_experts: int,
        max_dispatch_tokens_per_rank: int,
        router_topk: int,
        layout,
        instance_id: int = 0,
    ) -> NcclEpBuffer:
        state = cls._state(instance_id)
        pynccl = ep_group.pynccl_comm
        if pynccl is None or not getattr(pynccl, "available", False):
            raise RuntimeError("NCCL EP requires a live PyNccl communicator")
        configuration = (
            pynccl.comm.value,
            ep_group.device,
            ep_group.world_size,
            hidden_size,
            num_experts,
            num_local_experts,
            max_dispatch_tokens_per_rank,
            router_topk,
            layout,
        )
        if state.group is not None and state.configuration != configuration:
            raise ValueError(
                "Incompatible MoE layer for the shared NCCL EP eager group"
            )
        if state.group is None:
            state.num_experts = num_experts
            state.num_local_experts = num_local_experts
            if num_experts % ep_group.world_size:
                raise ValueError(
                    "NCCL EP requires num_experts to be divisible by EP world size "
                    f"(got {num_experts} / {ep_group.world_size})."
                )
            # NCCL EP's expert-major ABI indexes experts owned by this NCCL
            # rank, not the model runner's local-expert configuration.  Those
            # can differ for TP-sharded model implementations (GLM is one).
            state.nccl_num_local_experts = num_experts // ep_group.world_size
            state.hidden_size = hidden_size
            state.max_dispatch_tokens_per_rank = max_dispatch_tokens_per_rank
            state.router_topk = router_topk
            state.layout = layout
            # LL auto = nRanks * max_dispatch_tokens_per_rank.
            world_size = ep_group.world_size
            state.max_recv_tokens_per_rank = world_size * max_dispatch_tokens_per_rank
            cls._create_group(state, ep_group)
            state.configuration = configuration
        return state

    @classmethod
    def _create_group(cls, state, ep_group: GroupCoordinator):
        nccl_core, nccl_ep = _load_nccl_ep()

        pynccl = ep_group.pynccl_comm
        if pynccl is None or not getattr(pynccl, "available", False):
            raise RuntimeError(
                "NCCL EP requires a live PyNccl communicator on the EP group "
                "(pynccl_comm unavailable)."
            )
        comm_ptr = pynccl.comm.value  # ncclComm_t (c_void_p) -> int
        wrapped_comm = nccl_core.Communicator(ptr=comm_ptr)

        # Both layouts defer RDMA sizing to the first concrete handle.  The
        # old, verified expert-major path used this NCCL auto mode; applying a
        # generic worst-case hint changes its allocation/ABI behavior.
        rdma_buffer_size = 0
        cfg = nccl_ep.GroupConfig(
            algorithm=nccl_ep.Algorithm.LOW_LATENCY,
            num_experts=state.num_experts,
            max_dispatch_tokens_per_rank=state.max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank=0,  # 0 = auto = nRanks * max_dispatch
            max_token_bytes=state.hidden_size * 2,  # bf16 = 2 bytes/elem
            rdma_buffer_size=rdma_buffer_size,
            max_num_sms=cls._resolve_max_num_sms(state.num_experts),
        )
        state.group = nccl_ep.Group.create(wrapped_comm, cfg)
        logger.info(
            "NCCL EP group created: num_experts=%d world_size=%d hidden=%d "
            "max_dispatch_tokens_per_rank=%d rdma_buffer_size=%d",
            state.num_experts,
            ep_group.world_size,
            state.hidden_size,
            state.max_dispatch_tokens_per_rank,
            rdma_buffer_size,
        )
        if not state.layout.is_rank_major():
            cls._alloc_scratch(state, ep_group.device)

    @staticmethod
    def _alloc_scratch(state, device: torch.device):
        """Allocate fixed-shape dispatch/combine scratch once, reused across steps
        (counters re-zeroed per dispatch). recv_tokens need not be cleared — the
        kernel writes the valid region, bounded downstream by expert_counters.
        """
        # Expert-major compute consumes the model runner's group count.  This
        # is deliberately distinct from rank-major's NCCL-local metadata.
        e_local = state.num_local_experts
        h = state.hidden_size
        max_recv = state.max_recv_tokens_per_rank
        max_send = state.max_dispatch_tokens_per_rank
        state.recv_tokens = torch.empty(
            (e_local, max_recv, h), dtype=torch.bfloat16, device=device
        )
        state.expert_counters = torch.zeros(
            (e_local,), dtype=torch.int32, device=device
        )
        state.expert_offsets = torch.zeros(
            (e_local + 1,), dtype=torch.int32, device=device
        )
        state.recv_total = torch.zeros((1,), dtype=torch.int32, device=device)
        # Upper bound = max dispatch tokens per rank; sliced to [:t] per combine.
        state.combined = torch.empty((max_send, h), dtype=torch.bfloat16, device=device)

    @staticmethod
    def _alloc_rank_major_scratch(state, device: torch.device):
        """Allocate RM-only scratch lazily so expert-major keeps its baseline VRAM."""
        if state.rm_recv_tokens is not None:
            return
        max_send = state.max_dispatch_tokens_per_rank
        h = state.hidden_size
        world_size = state.max_recv_tokens_per_rank // max_send
        state.rm_recv_tokens = torch.empty(
            (world_size, max_send, h), dtype=torch.bfloat16, device=device
        )
        state.rm_recv_topk_ids = torch.empty(
            (world_size, max_send, state.router_topk),
            dtype=torch.int32,
            device=device,
        )
        state.rm_recv_topk_weights = torch.empty(
            (world_size, max_send, state.router_topk),
            dtype=torch.float32,
            device=device,
        )
        state.rm_src_rank_counters = torch.zeros(
            (world_size,), dtype=torch.int32, device=device
        )
        # The fused weighted-reduction kernel gathers expert rows directly;
        # it does not need the old [world, D, topk, H] route materialization.
        state.rm_route_buffer = None
        state.rm_pre_reduced = torch.empty_like(state.rm_recv_tokens)
        # Rank-major combines restore the sender's original token order, so
        # their output remains a 2D [max_dispatch, hidden] tensor.
        state.rm_combined = torch.empty(
            (max_send, h), dtype=torch.bfloat16, device=device
        )

    @staticmethod
    def _low_latency_rdma_size_hint(nccl_ep, state, world_size: int) -> int:
        """Worst-case RDMA buffer size for LL (explicit so handle init is local).

        Prefers a nccl4py native hint; falls back to a conservative upper bound
        from the max recv footprint (E_local * max_recv * H * 2 bytes bf16), x4
        for send+recv+slack.
        """
        hint_fn = getattr(nccl_ep, "get_low_latency_rdma_size_hint", None)
        if callable(hint_fn):
            try:
                return int(
                    hint_fn(
                        state.max_dispatch_tokens_per_rank,
                        state.hidden_size,
                        world_size,
                        state.num_experts,
                    )
                )
            except Exception:
                pass  # signature mismatch; fall through to the bound below
        per_slot = state.num_local_experts * state.max_recv_tokens_per_rank
        per_slot *= state.hidden_size * 2  # bf16 bytes
        return int(per_slot * 4)

    @staticmethod
    def _resolve_max_num_sms(num_experts: int) -> int:
        """Cap LL EP-group SMs so overlap compute can run alongside dispatch.

        LL requires ceil(num_experts / max_num_sms) <= 14, i.e. a floor of
        ceil(num_experts / 14). Default 20 (leaves most SMs for compute, like
        DeepEP's communicate_num_sms); overridable via SGLANG_NCCL_EP_MAX_NUM_SMS.
        """
        from sglang.srt.environ import envs

        if (
            envs.SGLANG_NCCL_EP_MAX_NUM_SMS.is_set()
            and envs.SGLANG_NCCL_EP_MAX_NUM_SMS.get() > 0
        ):
            val = envs.SGLANG_NCCL_EP_MAX_NUM_SMS.get()
        else:
            val = 20  # conservative default for H100 (132 SMs)

        min_required = (num_experts + 13) // 14  # ceil(num_experts / 14)
        if val < min_required:
            logger.warning(
                "NCCL EP max_num_sms=%d below LL minimum %d (num_experts=%d); "
                "clamping to %d.",
                val,
                min_required,
                num_experts,
                min_required,
            )
            val = min_required
        return val

    @classmethod
    def destroy(cls):
        from sglang.srt.runtime_context import get_resources

        states = [
            get_resources().buffers[key]
            for key in ("nccl_ep_state", "nccl_ep_state_1")
            if key in get_resources().buffers
        ]
        # Check all lanes before releasing any resource.
        if any(state.borrower is not None for state in states):
            raise RuntimeError("NCCL EP eager group has an incomplete transaction")
        dispatchers = {
            dispatcher for state in states for dispatcher in state.dispatchers
        }
        if any(dispatcher._stage != _Stage.INITIAL for dispatcher in dispatchers):
            raise RuntimeError("Cannot close NCCL EP during an active transaction")
        for dispatcher in dispatchers:
            dispatcher.close()
        for state in states:
            if state.group is not None:
                state.group.destroy()
                state.group = None
            state.dispatchers.clear()


# ----------------------------- Dispatcher (LL path) -----------------------------


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class NcclEpDispatcher(BaseDispatcher):
    """NCCL EP low-latency token dispatcher.

    Mirrors the DeepEP LL contract: ``dispatch`` returns a
    ``DeepEPLLDispatchOutput`` (DEEPEP_LL format), reusing the DeepEP compute
    path. NCCL EP carries bf16 on the wire; we quantize bf16->fp8 (group 128)
    after dispatch to feed ``apply_deepep_ll``.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        ep_group: GroupCoordinator,
        *,
        instance_id: int | None = None,
    ):
        super().__init__()
        if instance_id not in (None, 0, 1):
            raise ValueError("NCCL EP supports two TBO subbatches")
        self.instance_id = instance_id
        if moe_runner_config.params_dtype != torch.bfloat16:
            raise ValueError(
                "NCCL EP LL requires bfloat16 parameters (--dtype bfloat16)"
            )
        nccl_core, nccl_ep = _load_nccl_ep()
        self._nccl_ep = nccl_ep

        self.ep_group = ep_group
        self.router_topk = moe_runner_config.top_k
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.hidden_size = moe_runner_config.hidden_size
        self.params_dtype = moe_runner_config.params_dtype
        self.world_size = ep_group.world_size

        # Resolve dispatch mode. LOW_LATENCY only this PR; HT raises at resolve() time.
        self.mode = get_nccl_ep_mode().resolve(is_extend_in_batch=False)
        self.layout = get_nccl_ep_layout()

        # Keep direct dispatcher construction consistent with the public gate.
        if reason := _nccl_ep_scale_unavailable_reason():
            raise NotImplementedError(reason)

        # Deterministic-inference guard: NCCL EP is not determinism-audited yet.
        try:
            from sglang.srt.runtime_context import get_exec

            if get_exec().deterministic.enable_deterministic_inference:
                raise NotImplementedError(
                    "NCCL EP is not deterministic-inference-audited yet. Use "
                    "--moe-a2a-backend deepep with --enable-deterministic-inference, "
                    "or remove the flag (deterministic support tracked as a follow-up)."
                )
        except (ImportError, AttributeError, ValueError):
            pass  # exec flags not materialized yet; gated in server_args post-processing.

        # Per-rank dispatch budget.
        budget = get_nccl_ep_num_max_dispatch_tokens_per_rank()
        if budget <= 0:
            budget = _NCCL_EP_DEFAULT_MAX_DISPATCH_TOKENS_PER_RANK
        assert budget <= _NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP, (
            f"NCCL EP max_dispatch_tokens_per_rank {budget} exceeds cap {_NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP}"
        )
        self.num_max_dispatch_tokens_per_rank = budget

        # Validate LL hard constraints early (fail fast at init, not at first dispatch).
        if self.hidden_size not in _NCCL_EP_LL_SUPPORTED_HIDDEN:
            raise ValueError(
                f"NCCL EP LL only supports hidden in {sorted(_NCCL_EP_LL_SUPPORTED_HIDDEN)} "
                f"(got {self.hidden_size}); use --moe-a2a-backend deepep for this model."
            )
        if self.router_topk > _NCCL_EP_LL_MAX_TOPK:
            raise ValueError(
                f"NCCL EP LL supports topk <= {_NCCL_EP_LL_MAX_TOPK} (got {self.router_topk}); "
                f"fall back to DeepEP LL (upstream nccl_ep issue #2103)."
            )

        self.buffer = None
        self._comm_initialized = False

        self.handle = None
        self._handle_persistent = False
        self._graph_resources = None
        self._eager_session = None
        self._active_buffer = self.buffer
        self._saved_eager_handle = None

        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        from .nccl_ep_stream import get_nccl_ep_stream

        self._comm = (
            get_nccl_ep_stream(ep_group.device, instance_id=instance_id or 0)
            if is_sbo_enabled() or is_tbo_enabled()
            else None
        )

        # Staged execution state machine (mirrors deepep.py _Stage).
        self._stage = _Stage.INITIAL
        self._dispatch_intermediate_state = None
        self._combine_intermediate_state = None

        # SBO dispatch hook: deepseek_v2.py registers a hook via
        # register_deepep_dispatch_hook to run shared experts between
        # dispatch_a (send_only) and dispatch_b (complete).
        self._dispatch_hooks = DeepEPPDispatchHooks()

        # Lazily imported; fp8 quant is only needed for the fp8 (w4afp8) GEMM path.
        self._quant_fp8 = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def _send_context(self, *inputs):
        from .nccl_ep_graph import is_nccl_ep_graph_capture

        return (
            self._comm.send(*inputs)
            if self._comm is not None and is_nccl_ep_graph_capture()
            else nullcontext()
        )

    def _complete(self):
        from .nccl_ep_graph import is_nccl_ep_graph_capture

        if self._comm is not None and is_nccl_ep_graph_capture():
            self._comm.complete(self.handle)
        else:
            self.handle.complete(
                config=0, stream=torch.cuda.current_stream().cuda_stream
            )

    # _Stage state machine: guards a/b call order. handle's continue_fn is a
    # single slot — calling combine(send_only=1) before dispatch_b (complete)
    # would overwrite the dispatch continue_fn.
    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage, (
            f"NCCL EP stage mismatch: expected {old_stage}, got {self._stage}"
        )
        self._stage = new_stage

    def register_deepep_dispatch_hook(self, hook):
        return self._dispatch_hooks.register_hook(hook)

    def _ensure_buffer(self):
        if self.buffer is None:
            self.buffer = NcclEpBuffer.get_buffer(
                self.ep_group,
                self.hidden_size,
                self.num_experts,
                self.num_local_experts,
                self.num_max_dispatch_tokens_per_rank,
                self.router_topk,
                self.layout,
                instance_id=self.instance_id or 0,
            )
            self.buffer.dispatchers.add(self)
            self._comm_initialized = True

    def close(self):
        """Release native handles before their shared EP group is destroyed."""
        if self._stage != _Stage.INITIAL:
            raise RuntimeError("Cannot close NCCL EP during an active transaction")
        if self.handle is not None:
            self.handle.destroy()
            self.handle = None
        self._handle_persistent = False
        self._comm_initialized = False
        self.buffer = None

    def init_comm_resources(self):
        if self._comm_initialized:
            return
        self._ensure_buffer()

    def init_handle_for_graph(self):
        if self._handle_persistent:
            return
        if self.layout.is_rank_major():
            raise NotImplementedError(
                "NCCL EP rank-major CUDA graph capture is not enabled yet; "
                "use --disable-cuda-graph for this first implementation."
            )
        self.init_comm_resources()
        nccl_ep = self._nccl_ep
        ptr = nccl_ep._ep_bindings.init_handle(
            self.buffer.group.ptr,
            int(
                nccl_ep.Layout.RANK_MAJOR
                if self.layout.is_rank_major()
                else nccl_ep.Layout.EXPERT_MAJOR
            ),
            0,
            self.router_topk,
            0,
        )
        self.handle = nccl_ep.Handle(ptr)
        self._handle_persistent = True

    # ---- three-phase dispatch/combine (staged execution) ----
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._dispatch_hooks is not None:
            self._dispatch_hooks(self)
        return self.dispatch_b()

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        from .nccl_ep_graph import get_nccl_ep_graph_resources

        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("NCCL EP LL dispatch requires bfloat16 hidden states")
        owner = get_nccl_ep_graph_resources()
        graph_owner = owner if owner is not None and owner.capturing else None
        if torch.cuda.is_current_stream_capturing() and graph_owner is None:
            raise RuntimeError(
                "NCCL EP capture requires the opt-in full decode Graph backend"
            )
        if self._stage != _Stage.INITIAL:
            raise RuntimeError("NCCL EP has an incomplete dispatch/combine transaction")

        nccl_ep = self._nccl_ep
        topk_weights = topk_output.topk_weights.to(torch.float32)
        topk_ids = topk_output.topk_ids.to(torch.int64)

        t = hidden_states.shape[0]
        if t > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"NCCL EP: batch ({t}) exceeds per-rank dispatch budget "
                f"{self.num_max_dispatch_tokens_per_rank}; reduce the decode batch "
                f"or --chunked-prefill-size. The LL budget is capped at "
                f"{_NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP}."
            )

        with self._send_context(hidden_states, topk_ids, topk_weights):
            stream = torch.cuda.current_stream()
            if graph_owner is not None:
                if self.layout.is_rank_major():
                    raise ValueError("NCCL EP CUDA Graph requires expert_major layout")
                state, handle, hidden_states, topk_ids, topk_weights = (
                    graph_owner.prepare(self, hidden_states, topk_ids, topk_weights)
                )
                self._saved_eager_handle = self.handle
                self._graph_resources = graph_owner
            else:
                if owner is not None:
                    if self.instance_id is not None:
                        # Staged lanes finish in non-LIFO order; the runner owns
                        # the outer submission session.
                        owner.require_session("eager")
                    else:
                        session = owner.submission_session("transaction")
                        session.__enter__()
                        self._eager_session = session
                try:
                    self._ensure_buffer()
                    state = self.buffer
                    if state.borrower is not None:
                        raise RuntimeError(
                            "NCCL EP eager group has an incomplete transaction"
                        )
                    if self._handle_persistent:
                        handle = self.handle
                        handle.update(
                            topk_idx=nccl_ep.Tensor(topk_ids),
                            layout_info=None,
                            stream=stream.cuda_stream,
                        )
                    else:
                        handle = state.group.create_handle(
                            layout=(
                                nccl_ep.Layout.RANK_MAJOR
                                if self.layout.is_rank_major()
                                else nccl_ep.Layout.EXPERT_MAJOR
                            ),
                            topk_idx=nccl_ep.Tensor(topk_ids),
                            config=nccl_ep.HandleConfig(),
                            stream=stream.cuda_stream,
                        )
                    state.borrower = self
                except BaseException:
                    if self._eager_session is not None:
                        self._eager_session.__exit__(None, None, None)
                        self._eager_session = None
                    raise
            self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
            self.handle = handle
            self._active_buffer = state
            self._dispatched_topk_ids = topk_ids
            if self.layout.is_rank_major():
                NcclEpBuffer._alloc_rank_major_scratch(state, self.ep_group.device)
                self._dispatch_a_rank_major(
                    hidden_states, topk_ids, topk_weights, t, state, stream
                )
                return

            # Reuse pre-allocated scratch; counters re-zeroed each dispatch.
            recv_tokens = state.recv_tokens
            expert_counters = state.expert_counters.zero_()
            expert_offsets = state.expert_offsets.zero_()
            recv_total = state.recv_total.zero_()

            inputs = nccl_ep.DispatchInputs(tokens=nccl_ep.Tensor(hidden_states))
            outputs = nccl_ep.DispatchOutputs(tokens=nccl_ep.Tensor(recv_tokens))
            layout_info = nccl_ep.LayoutInfo(
                expert_counters=nccl_ep.Tensor(expert_counters),
                expert_offsets=nccl_ep.Tensor(expert_offsets),
                recv_total_counter=nccl_ep.Tensor(recv_total),
            )
            self.handle.dispatch(
                inputs,
                outputs,
                layout_info=layout_info,
                config=nccl_ep.DispatchConfig(send_only=1),
                stream=stream.cuda_stream,
            )

            self._dispatch_intermediate_state = (
                recv_tokens,
                expert_counters,
                topk_ids,
                topk_weights,
                t,
                hidden_states,
                # Native send_only borrows the tensor descriptors until complete.
                # Retaining the Torch allocations alone does not keep these alive.
                inputs,
                outputs,
                layout_info,
            )

    def _dispatch_a_rank_major(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        t: int,
        state,
        stream,
    ):
        """Start an NCCL RM dispatch with the API's mandatory routing metadata."""
        nccl_ep = self._nccl_ep
        recv_tokens = state.rm_recv_tokens
        recv_ids = state.rm_recv_topk_ids
        recv_weights = state.rm_recv_topk_weights
        src_rank_counters = state.rm_src_rank_counters.zero_()
        inputs = nccl_ep.DispatchInputs(
            tokens=nccl_ep.Tensor(hidden_states),
            topk_weights=nccl_ep.Tensor(topk_weights),
        )
        outputs = nccl_ep.DispatchOutputs(
            tokens=nccl_ep.Tensor(recv_tokens),
            topk_idx=nccl_ep.Tensor(recv_ids),
            topk_weights=nccl_ep.Tensor(recv_weights),
        )
        layout_info = nccl_ep.LayoutInfo(
            src_rank_counters=nccl_ep.Tensor(src_rank_counters),
        )
        self.handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=nccl_ep.DispatchConfig(send_only=1),
            stream=stream.cuda_stream,
        )
        self._dispatch_intermediate_state = (
            recv_tokens,
            recv_ids,
            recv_weights,
            src_rank_counters,
            t,
        )

    def dispatch_b(self) -> DispatchOutput:
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)

        if self.layout.is_rank_major():
            return self._dispatch_b_rank_major()

        (
            recv_tokens,
            expert_counters,
            topk_ids,
            topk_weights,
            t,
            hidden_states,
            _inputs,
            _outputs,
            _layout_info,
        ) = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state

        self._complete()

        hs_fp8, hs_scale = self._quantize_fp8(recv_tokens, expert_counters)

        expected_m = (
            hidden_states.shape[0] * self.world_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts

        self._dispatched_topk_ids = topk_ids
        self._dispatched_topk_weights = topk_weights
        self._dispatched_t = t

        return DeepEPLLDispatchOutput(
            hs_fp8,
            hs_scale,
            topk_ids,
            topk_weights,
            expert_counters,
            expected_m,
        )

    def _dispatch_b_rank_major(self) -> NcclEpRankMajorDispatchOutput:
        recv_tokens, recv_ids, recv_weights, src_rank_counters, t = (
            self._dispatch_intermediate_state
        )
        del self._dispatch_intermediate_state

        self._complete()

        # A source rank cannot contribute more slots than its local input
        # token count ``t``. NCCL lays each source rank's valid receive slots
        # contiguously, so trim the fixed D capacity to [world, t] before the
        # MoE preprocess. This avoids scanning world*D*topk metadata during
        # low-concurrency decode without a dynamic-size GPU->CPU sync.
        compact_width = t
        if compact_width > recv_tokens.shape[1]:
            raise ValueError("rank-major receive count exceeds the slot capacity")
        compact_tokens = recv_tokens[:, :compact_width].contiguous()
        compact_ids = recv_ids[:, :compact_width].contiguous()
        compact_weights = recv_weights[:, :compact_width].contiguous()

        # NCCL RM output already stores local expert ids (-1 for nonlocal
        # routes). Clamp defensively before passing them to the existing
        # masked W4AFP8 GEMM preparation kernel. Each valid
        # local route duplicates its source slot once, which is exactly the
        # rank-major API's required local pre-reduction semantics.
        slot_valid = (
            torch.arange(compact_width, device=recv_tokens.device, dtype=torch.int32)[
                None, :
            ]
            < src_rank_counters[:, None]
        )
        local_ids = rank_major_local_expert_ids(
            torch.where(
                slot_valid[:, :, None],
                compact_ids,
                torch.full_like(compact_ids, -1),
            ),
            num_local_experts=self.num_local_experts,
        )
        compact_weights = torch.where(
            slot_valid[:, :, None],
            compact_weights,
            torch.zeros_like(compact_weights),
        )
        from sglang.kernels.ops.moe.ep_moe_kernels import moe_ep_deepgemm_preprocess

        slot_shape = (recv_tokens.shape[0], compact_width)
        # Quantize while scattering into the expert-major compute view.  Doing
        # this in preprocess avoids materializing a packed BF16 tensor followed
        # by a second full-tensor BF16->FP8 pass.
        masked_m, expected_m, src2dst, packed_fp8, packed_scale = (
            moe_ep_deepgemm_preprocess(
                local_ids.reshape(-1, self.router_topk),
                self.num_local_experts,
                compact_tokens.reshape(-1, self.hidden_size),
                self.router_topk,
                block_shape=[128, _NCCL_EP_FP8_GROUP_SIZE],
                output_dtype=torch.float8_e4m3fn,
            )
        )
        self._dispatched_t = t
        return NcclEpRankMajorDispatchOutput(
            packed_fp8,
            packed_scale,
            local_ids.reshape(-1, self.router_topk),
            masked_m,
            expected_m,
            src2dst,
            local_ids.reshape(-1, self.router_topk),
            compact_weights.reshape(-1, self.router_topk),
            slot_shape,
        )

    def _quantize_fp8(self, recv_tokens_bf16: torch.Tensor, masked_m: torch.Tensor):
        if self._quant_fp8 is None:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            self._quant_fp8 = sglang_per_token_group_quant_fp8
        # recv_tokens_bf16 is 3D [E_local, max_recv, H]; the quant helper supports
        # 3D + masked_m. masked_m=expert_counters matches DeepEPLLDispatchOutput.
        return self._quant_fp8(
            recv_tokens_bf16,
            group_size=_NCCL_EP_FP8_GROUP_SIZE,
            masked_m=masked_m,
        )

    def combine(self, combine_input) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(self, combine_input):
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)

        nccl_ep = self._nccl_ep
        if isinstance(combine_input, NcclEpRankMajorCombineInput):
            self._combine_a_rank_major(combine_input)
            return
        if isinstance(combine_input, DeepEPLLCombineInput):
            expert_outputs = combine_input.hidden_states
            topk_weights = combine_input.topk_weights
        else:
            expert_outputs = combine_input
            topk_weights = self._dispatched_topk_weights

        t = self._dispatched_t  # bounded in dispatch_a.

        # Reuse pre-allocated [max_send, H] buffer.
        combined = self._active_buffer.combined[:t]

        with self._send_context(expert_outputs, topk_weights):
            stream = torch.cuda.current_stream()
            inputs = nccl_ep.CombineInputs(tokens=nccl_ep.Tensor(expert_outputs))
            outputs = nccl_ep.CombineOutputs(
                tokens=nccl_ep.Tensor(combined),
                topk_weights=nccl_ep.Tensor(topk_weights),
            )
            self.handle.combine(
                inputs,
                outputs,
                config=nccl_ep.CombineConfig(send_only=1),
                stream=stream.cuda_stream,
            )

        self._combine_intermediate_state = (combined, inputs, outputs)

    def _combine_a_rank_major(self, combine_input: NcclEpRankMajorCombineInput):
        pre_reduced = reduce_rank_major_expert_outputs(
            combine_input.hidden_states,
            combine_input.src2dst,
            combine_input.local_topk_ids,
            combine_input.recv_topk_weights,
            combine_input.slot_shape,
            route_buffer=self.buffer.rm_route_buffer,
            output=self.buffer.rm_pre_reduced,
        )
        # NCCL's RM combine consumes already weighted/reduced source slots;
        # passing topk_weights here would double-apply routing weights.
        combined = self.buffer.rm_combined[: self._dispatched_t]
        stream = torch.cuda.current_stream()
        inputs = self._nccl_ep.CombineInputs(tokens=self._nccl_ep.Tensor(pre_reduced))
        outputs = self._nccl_ep.CombineOutputs(tokens=self._nccl_ep.Tensor(combined))
        self.handle.combine(
            inputs,
            outputs,
            config=self._nccl_ep.CombineConfig(send_only=1),
            stream=stream.cuda_stream,
        )
        self._combine_intermediate_state = (combined, inputs, outputs)

    def combine_b(self) -> torch.Tensor:
        if self._stage != _Stage.AFTER_COMBINE_A:
            raise RuntimeError("NCCL EP combine_b requires a pending combine")
        combined, _inputs, _outputs = self._combine_intermediate_state
        # Release ownership only after successful completion. Native/GPU faults
        # leave an incomplete transaction and require process restart, rather
        # than allowing another layer to reuse uncertain communication state.
        self._complete()
        if self._graph_resources is not None:
            # Later MoE layers reuse the group's combine scratch. Keep this
            # layer's live output in the captured graph's allocator pool.
            combined = combined.clone()
            self._graph_resources.release(self)
            self._graph_resources = None
            self.handle = self._saved_eager_handle
            self._saved_eager_handle = None
        else:
            # The next staged layer may reuse this lane before its output is consumed.
            if self.instance_id is not None:
                combined = combined.clone()
            self._active_buffer.borrower = None
            if not self._handle_persistent:
                self.handle.destroy()
                self.handle = None
            if self._eager_session is not None:
                self._eager_session.__exit__(None, None, None)
                self._eager_session = None
        del self._dispatched_topk_ids
        del self._combine_intermediate_state
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        return combined
