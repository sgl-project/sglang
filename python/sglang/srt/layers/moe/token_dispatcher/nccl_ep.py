"""NCCL EP low-latency MoE dispatch/combine backend (``--moe-a2a-backend nccl_ep``).

Mirrors the DeepEP LL contract (DEEPEP_LL output, reuses DeepEPMoE.compute).
NCCL EP carries bf16 on the wire, so we quantize the bf16 recv to fp8 (group 128)
after dispatch. LL (decode) only; HT (prefill) is a follow-up.
"""

from __future__ import annotations

import importlib.util
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPPDispatchHooks,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
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
        import nccl

        # nccl.get_version() -> VersionInfo with a .nccl LibraryInfo carrying .version.
        vi = nccl.get_version()
        lib_info = getattr(vi, "nccl", None)
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


def nccl_ep_unavailable_reason() -> str | None:
    """Return why NCCL EP is unavailable (None = available), for fallback logs."""
    if importlib.util.find_spec("nccl.ep") is None:
        return "nccl4py (nccl.ep) not importable"
    if not torch.cuda.is_available():
        return "no CUDA device"
    cc = torch.cuda.get_device_capability(0)[0]  # Hopper (sm_90) or Blackwell.
    if cc < 9:
        return f"GPU arch sm_{cc}x not supported (need Hopper/Blackwell sm_90+)"
    ver = _nccl_runtime_version()
    if ver is None:
        return "could not read NCCL version"
    if (ver[0], ver[1]) < (2, 29):
        return f"NCCL {'.'.join(map(str, ver))} < 2.29 (need Device API/GIN)"
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

    State lives on ``ctx.resources.buffers["nccl_ep_state"]``; the group is
    created once from the sglang EP group's ``ncclComm_t``.
    """

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        state = buffers.get("nccl_ep_state")
        if state is None:
            state = SimpleNamespace(
                group=None,
                num_experts=None,
                num_local_experts=None,
                hidden_size=None,
                max_dispatch_tokens_per_rank=None,
                max_recv_tokens_per_rank=None,
                # Pre-allocated scratch (fixed shapes); reused per dispatch/combine step.
                recv_tokens=None,
                expert_counters=None,
                expert_offsets=None,
                recv_total=None,
                combined=None,
            )
            buffers["nccl_ep_state"] = state
        return state

    @classmethod
    def get_buffer(
        cls,
        ep_group: "GroupCoordinator",
        hidden_size: int,
        num_experts: int,
        num_local_experts: int,
        max_dispatch_tokens_per_rank: int,
    ) -> "NcclEpBuffer":
        state = cls._state()
        if state.group is None:
            state.num_experts = num_experts
            state.num_local_experts = num_local_experts
            state.hidden_size = hidden_size
            state.max_dispatch_tokens_per_rank = max_dispatch_tokens_per_rank
            # LL auto = nRanks * max_dispatch_tokens_per_rank.
            world_size = ep_group.world_size
            state.max_recv_tokens_per_rank = world_size * max_dispatch_tokens_per_rank
            cls._create_group(state, ep_group)
        return state

    @classmethod
    def _create_group(cls, state, ep_group: "GroupCoordinator"):
        nccl_core, nccl_ep = _load_nccl_ep()

        pynccl = ep_group.pynccl_comm
        if pynccl is None or not getattr(pynccl, "available", False):
            raise RuntimeError(
                "NCCL EP requires a live PyNccl communicator on the EP group "
                "(pynccl_comm unavailable)."
            )
        comm_ptr = pynccl.comm.value  # ncclComm_t (c_void_p) -> int
        wrapped_comm = nccl_core.Communicator(ptr=comm_ptr)

        # Explicit rdma_buffer_size keeps create_handle local (no collective/realloc).
        rdma_buffer_size = cls._low_latency_rdma_size_hint(
            nccl_ep, state, ep_group.world_size
        )
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
        cls._alloc_scratch(state, ep_group.device)

    @staticmethod
    def _alloc_scratch(state, device: torch.device):
        """Allocate fixed-shape dispatch/combine scratch once, reused across steps
        (counters re-zeroed per dispatch). recv_tokens need not be cleared — the
        kernel writes the valid region, bounded downstream by expert_counters.
        """
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
        state.combined = torch.empty(
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

        if envs.SGLANG_NCCL_EP_MAX_NUM_SMS.is_set() and envs.SGLANG_NCCL_EP_MAX_NUM_SMS.get() > 0:
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
        state = cls._state()
        g = state.group
        if g is not None:
            try:
                g.destroy()
            except Exception:
                pass
            state.group = None


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

    def __init__(self, moe_runner_config: "MoeRunnerConfig", ep_group: "GroupCoordinator"):
        super().__init__()
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

        # Blackwell guard: our fp8 post-quant emits float32 group scales, which
        # diverge from DeepEP's UE8M0 scales under DEEPGEMM_BLACKWELL. Fail fast
        # rather than silently mis-quantize.
        try:
            from sglang.srt.layers import deep_gemm_wrapper

            if getattr(deep_gemm_wrapper, "DEEPGEMM_BLACKWELL", False):
                raise NotImplementedError(
                    "NCCL EP LL fp8 post-quant emits float32 group scales, which diverge "
                    "from DeepEP's UE8M0 scales when DEEPGEMM_BLACKWELL is set. Fall back "
                    "to --moe-a2a-backend deepep on Blackwell for now."
                )
        except ImportError:
            pass  # deep_gemm_wrapper absent -> DeepGEMM-Blackwell path not active.

        # Deterministic-inference guard: NCCL EP is not determinism-audited yet.
        try:
            from sglang.srt.runtime_context import get_exec

            if get_exec().deterministic.enable_deterministic_inference:
                raise NotImplementedError(
                    "NCCL EP is not deterministic-inference-audited yet. Use "
                    "--moe-a2a-backend deepep with --enable-deterministic-inference, "
                    "or remove the flag (deterministic support tracked as a follow-up)."
                )
        except (ImportError, AttributeError):
            pass  # exec flags not materialized yet; gated in server_args post-processing.

        # Per-rank dispatch budget.
        budget = get_nccl_ep_num_max_dispatch_tokens_per_rank()
        if budget <= 0:
            budget = _NCCL_EP_DEFAULT_MAX_DISPATCH_TOKENS_PER_RANK
        assert (
            budget <= _NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP
        ), f"NCCL EP max_dispatch_tokens_per_rank {budget} exceeds cap {_NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP}"
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

        self.buffer = NcclEpBuffer.get_buffer(
            ep_group,
            self.hidden_size,
            self.num_experts,
            self.num_local_experts,
            self.num_max_dispatch_tokens_per_rank,
        )

        self.handle = None

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

    # ---- three-phase dispatch/combine (staged execution) ----
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._dispatch_hooks is not None:
            self._dispatch_hooks(self)
        return self.dispatch_b()

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)

        nccl_ep = self._nccl_ep
        topk_weights = topk_output.topk_weights.to(torch.float32)
        topk_ids = topk_output.topk_ids.to(torch.int64)

        t = hidden_states.shape[0]
        if t > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"NCCL EP: decode batch ({t}) exceeds per-rank dispatch budget "
                f"{self.num_max_dispatch_tokens_per_rank}; increase "
                f"--nccl-ep-num-max-dispatch-tokens-per-rank or reduce batch."
            )

        if self.handle is not None:
            try:
                self.handle.destroy()
            except Exception:
                pass
            self.handle = None

        state = self.buffer

        stream = torch.cuda.current_stream()

        handle = state.group.create_handle(
            layout=nccl_ep.Layout.EXPERT_MAJOR,
            topk_idx=nccl_ep.Tensor(topk_ids),
            config=nccl_ep.HandleConfig(),
            stream=stream.cuda_stream,
        )
        self.handle = handle

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
        handle.dispatch(
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
        )

    def dispatch_b(self) -> DispatchOutput:
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)

        (
            recv_tokens,
            expert_counters,
            topk_ids,
            topk_weights,
            t,
            hidden_states,
        ) = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state

        stream = torch.cuda.current_stream()
        self.handle.complete(config=0, stream=stream.cuda_stream)

        hs_fp8, hs_scale = self._quantize_fp8(recv_tokens, expert_counters)

        expected_m = (
            hidden_states.shape[0] * self.world_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts

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
        if isinstance(combine_input, DeepEPLLCombineInput):
            expert_outputs = combine_input.hidden_states
            topk_weights = combine_input.topk_weights
        else:
            expert_outputs = combine_input
            topk_weights = self._dispatched_topk_weights

        t = self._dispatched_t  # bounded in dispatch_a.

        # Reuse pre-allocated [max_send, H] buffer.
        combined = self.buffer.combined[:t]

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

        self._combine_intermediate_state = combined

    def combine_b(self) -> torch.Tensor:
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)

        combined = self._combine_intermediate_state
        del self._combine_intermediate_state

        stream = torch.cuda.current_stream()
        try:
            self.handle.complete(config=0, stream=stream.cuda_stream)
        finally:
            if self.handle is not None:
                try:
                    self.handle.destroy()
                except Exception:
                    pass
                self.handle = None
        return combined
