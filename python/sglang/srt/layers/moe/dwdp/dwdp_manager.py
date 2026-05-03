# Copyright 2024-2025 SGLang Team
# Licensed under the Apache License, Version 2.0

"""DWDP Manager: Expert layout, IPC handle exchange, weight views, lifecycle.

Uses PyTorch native IPC APIs (_share_cuda_ / _new_shared_cuda) for cross-process
GPU memory sharing — this is portable across CUDA and ROCm/HIP.
"""

import ctypes
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class _HsaAgent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _HsaSignal(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _HsaDirectCopier:
    """Thin ctypes wrapper around hsa_amd_memory_async_copy_on_engine()."""

    HSA_STATUS_SUCCESS = 0
    HSA_DEVICE_TYPE_GPU = 1
    HSA_AGENT_INFO_DEVICE = 17
    HSA_SIGNAL_CONDITION_LT = 2
    HSA_WAIT_STATE_BLOCKED = 0
    HSA_WAIT_TIMEOUT = 0xFFFFFFFFFFFFFFFF

    def __init__(self):
        self.hsa = ctypes.CDLL("/opt/rocm/lib/libhsa-runtime64.so")
        self.roctx = None
        self.roctx_enabled = os.environ.get("DWDP_HSA_ROCTX", "0") == "1"
        if self.roctx_enabled:
            try:
                self.roctx = ctypes.CDLL("/opt/rocm/lib/libroctx64.so")
                self.roctx.roctxMarkA.argtypes = [ctypes.c_char_p]
                self.roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
                self.roctx.roctxRangePushA.restype = ctypes.c_int
                self.roctx.roctxRangePop.argtypes = []
                self.roctx.roctxRangePop.restype = ctypes.c_int
            except Exception:
                logger.exception("[DWDP] Failed to load ROCTX; markers disabled")
                self.roctx_enabled = False
                self.roctx = None

        self.timing_enabled = os.environ.get("DWDP_HSA_TIMING", "1") != "0"
        self.torch_profiler_enabled = (
            os.environ.get("DWDP_HSA_TORCH_PROFILER", "0") == "1"
        )
        self._torch_profiler_warned = False
        self._signal_meta: Dict[int, Tuple[int, int, int, str, Any]] = {}
        self._timing_samples: List[Tuple[int, float, float, int, str]] = []
        self._init_symbols()
        self._check(self.hsa.hsa_init(), "hsa_init")
        self.agents = self._get_gpu_agents()
        self._preferred_cache: Dict[Tuple[int, int], int] = {}

    def _roctx_mark(self, message: str):
        if self.roctx_enabled and self.roctx is not None:
            self.roctx.roctxMarkA(message.encode("utf-8", errors="replace"))

    def _roctx_push(self, message: str):
        if self.roctx_enabled and self.roctx is not None:
            self.roctx.roctxRangePushA(message.encode("utf-8", errors="replace"))

    def _roctx_pop(self):
        if self.roctx_enabled and self.roctx is not None:
            self.roctx.roctxRangePop()

    def _torch_profiler_enter(self, message: str):
        if not self.torch_profiler_enabled:
            return None
        try:
            return torch.ops.profiler._record_function_enter_new(message, "")
        except Exception:
            if not self._torch_profiler_warned:
                logger.exception("[DWDP] Failed to start PyTorch profiler range")
                self._torch_profiler_warned = True
            self.torch_profiler_enabled = False
            return None

    def _torch_profiler_exit(self, handle):
        if handle is None:
            return
        try:
            torch.ops.profiler._record_function_exit(handle)
        except Exception:
            if not self._torch_profiler_warned:
                logger.exception("[DWDP] Failed to end PyTorch profiler range")
                self._torch_profiler_warned = True
            self.torch_profiler_enabled = False

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(len(values) * pct)))
        return sorted(values)[idx]

    def timing_summary_and_reset(self) -> Optional[str]:
        if not self.timing_enabled or not self._timing_samples:
            return None

        samples = self._timing_samples
        self._timing_samples = []
        bytes_total = sum(s[0] for s in samples)
        latency_us = [s[1] for s in samples]
        wait_us = [s[2] for s in samples]
        engine_counts: Dict[int, int] = {}
        for _, _, _, engine, _ in samples:
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
        engine_str = ",".join(
            f"0x{engine:x}:{count}" for engine, count in sorted(engine_counts.items())
        )
        return (
            f"samples={len(samples)}, bytes={bytes_total / 1024**3:.2f}GB, "
            f"copy_latency_us[p50={self._percentile(latency_us, 0.50):.1f}, "
            f"p95={self._percentile(latency_us, 0.95):.1f}, "
            f"max={max(latency_us):.1f}], "
            f"host_wait_us[p50={self._percentile(wait_us, 0.50):.1f}, "
            f"p95={self._percentile(wait_us, 0.95):.1f}, "
            f"max={max(wait_us):.1f}], engines=[{engine_str}]"
        )

    def _init_symbols(self):
        self.hsa.hsa_init.restype = ctypes.c_int

        self.hsa.hsa_agent_get_info.argtypes = [
            _HsaAgent,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.hsa.hsa_agent_get_info.restype = ctypes.c_int

        self.hsa.hsa_signal_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(_HsaSignal),
        ]
        self.hsa.hsa_signal_create.restype = ctypes.c_int

        self.hsa.hsa_signal_destroy.argtypes = [_HsaSignal]
        self.hsa.hsa_signal_destroy.restype = ctypes.c_int

        self.hsa.hsa_signal_wait_scacquire.argtypes = [
            _HsaSignal,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        self.hsa.hsa_signal_wait_scacquire.restype = ctypes.c_int64

        self.hsa.hsa_amd_memory_get_preferred_copy_engine.argtypes = [
            _HsaAgent,
            _HsaAgent,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.hsa.hsa_amd_memory_get_preferred_copy_engine.restype = ctypes.c_int

        self.hsa.hsa_amd_memory_async_copy_on_engine.argtypes = [
            ctypes.c_void_p,
            _HsaAgent,
            ctypes.c_void_p,
            _HsaAgent,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
            _HsaSignal,
            ctypes.c_uint32,
            ctypes.c_bool,
        ]
        self.hsa.hsa_amd_memory_async_copy_on_engine.restype = ctypes.c_int

    def _check(self, status: int, name: str):
        if status != self.HSA_STATUS_SUCCESS:
            raise RuntimeError(f"{name} failed with HSA status {status}")

    def _get_gpu_agents(self) -> List[_HsaAgent]:
        agents: List[_HsaAgent] = []

        @ctypes.CFUNCTYPE(ctypes.c_int, _HsaAgent, ctypes.c_void_p)
        def callback(agent, data):
            device_type = ctypes.c_uint32(0)
            self.hsa.hsa_agent_get_info(
                agent, self.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type)
            )
            if device_type.value == self.HSA_DEVICE_TYPE_GPU:
                agents.append(agent)
            return self.HSA_STATUS_SUCCESS

        self.hsa.hsa_iterate_agents(callback, None)
        return agents

    def _preferred_engine(self, dst_device: int, src_device: int) -> int:
        key = (dst_device, src_device)
        if key in self._preferred_cache:
            return self._preferred_cache[key]

        preferred = ctypes.c_uint32(0)
        self._check(
            self.hsa.hsa_amd_memory_get_preferred_copy_engine(
                self.agents[dst_device],
                self.agents[src_device],
                ctypes.byref(preferred),
            ),
            "hsa_amd_memory_get_preferred_copy_engine",
        )
        if preferred.value == 0:
            raise RuntimeError(
                f"No preferred SDMA engine for GPU{dst_device}<-GPU{src_device}"
            )
        self._preferred_cache[key] = preferred.value
        return preferred.value

    def copy_async(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        dst_device: int,
        src_device: int,
        tag: str = "",
    ) -> _HsaSignal:
        if not dst.is_contiguous() or not src.is_contiguous():
            raise RuntimeError("HSA direct copy requires contiguous tensors")
        if dst.numel() != src.numel() or dst.element_size() != src.element_size():
            raise RuntimeError(
                f"HSA direct copy shape mismatch: dst={tuple(dst.shape)}, src={tuple(src.shape)}"
            )

        signal = _HsaSignal()
        self._check(
            self.hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal)),
            "hsa_signal_create",
        )
        size_bytes = dst.numel() * dst.element_size()
        engine = self._preferred_engine(dst_device, src_device)
        submit_ns = time.perf_counter_ns()
        profile_handle = self._torch_profiler_enter(
            f"DWDP_HSA_COPY {tag} bytes={size_bytes} engine=0x{engine:x}"
        )
        self._roctx_mark(
            f"DWDP_HSA_COPY_SUBMIT {tag} bytes={size_bytes} engine=0x{engine:x}"
        )
        self._roctx_push(
            f"DWDP_HSA_COPY_LAUNCH {tag} bytes={size_bytes} engine=0x{engine:x}"
        )
        try:
            status = self.hsa.hsa_amd_memory_async_copy_on_engine(
                ctypes.c_void_p(dst.data_ptr()),
                self.agents[dst_device],
                ctypes.c_void_p(src.data_ptr()),
                self.agents[src_device],
                ctypes.c_size_t(size_bytes),
                ctypes.c_uint32(0),
                None,
                signal,
                ctypes.c_uint32(engine),
                ctypes.c_bool(True),
            )
        finally:
            self._roctx_pop()
        if status != self.HSA_STATUS_SUCCESS:
            self._torch_profiler_exit(profile_handle)
            self.hsa.hsa_signal_destroy(signal)
            raise RuntimeError(
                f"hsa_amd_memory_async_copy_on_engine failed with HSA status {status}"
            )
        if self.timing_enabled or profile_handle is not None:
            self._signal_meta[signal.handle] = (
                submit_ns,
                size_bytes,
                engine,
                tag,
                profile_handle,
            )
        return signal

    def wait_and_destroy(self, signal: _HsaSignal):
        meta = self._signal_meta.pop(signal.handle, None)
        wait_start_ns = time.perf_counter_ns()
        wait_profile_handle = None
        if meta is not None:
            _, size_bytes, engine, tag, _ = meta
            self._roctx_push(
                f"DWDP_HSA_WAIT {tag} bytes={size_bytes} engine=0x{engine:x}"
            )
            wait_profile_handle = self._torch_profiler_enter(
                f"DWDP_HSA_WAIT {tag} bytes={size_bytes} engine=0x{engine:x}"
            )
        try:
            self.hsa.hsa_signal_wait_scacquire(
                signal,
                self.HSA_SIGNAL_CONDITION_LT,
                1,
                self.HSA_WAIT_TIMEOUT,
                self.HSA_WAIT_STATE_BLOCKED,
            )
        finally:
            if meta is not None:
                self._roctx_pop()
                self._torch_profiler_exit(wait_profile_handle)
        wait_end_ns = time.perf_counter_ns()
        if meta is not None and self.timing_enabled:
            submit_ns, size_bytes, engine, tag, profile_handle = meta
            self._timing_samples.append(
                (
                    size_bytes,
                    (wait_end_ns - submit_ns) / 1000.0,
                    (wait_end_ns - wait_start_ns) / 1000.0,
                    engine,
                    tag,
                )
            )
        elif meta is not None:
            _, _, _, _, profile_handle = meta
        else:
            profile_handle = None
        self._torch_profiler_exit(profile_handle)
        self.hsa.hsa_signal_destroy(signal)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_global_dwdp_manager: Optional["DwdpManager"] = None


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


def set_global_dwdp_manager(manager: Optional["DwdpManager"]):
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def enable_dwdp() -> bool:
    if _global_dwdp_manager is not None:
        return True
    try:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        return args is not None and getattr(args, "dwdp_size", 1) > 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DwdpExpertLayout
# ---------------------------------------------------------------------------
@dataclass
class DwdpExpertLayout:
    """Expert-to-rank mapping with optional overlapping allocation."""

    num_routed_experts: int  # e.g. 256 for DeepSeek V3
    dwdp_size: int  # number of GPUs in a DWDP group
    dwdp_rank: int  # 0..dwdp_size-1
    num_experts_per_worker: int  # experts stored locally per rank

    def __post_init__(self):
        assert self.num_experts_per_worker >= self.num_routed_experts // self.dwdp_size
        assert self.num_experts_per_worker <= self.num_routed_experts

        # Number of experts to prefetch from each peer
        self.num_prefetch_experts = math.ceil(
            (self.num_routed_experts - self.num_experts_per_worker)
            / (self.dwdp_size - 1)
        )

        # Local expert range [start, end)
        self.local_expert_start = min(
            self.num_prefetch_experts * self.dwdp_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )
        self.local_expert_end = self.local_expert_start + self.num_experts_per_worker

        # Precompute peer expert ranges
        self.peer_expert_ranges: Dict[int, Tuple[int, int]] = {}
        for r in range(self.dwdp_size):
            start = min(
                self.num_prefetch_experts * r,
                self.num_routed_experts - self.num_experts_per_worker,
            )
            end = start + self.num_experts_per_worker
            self.peer_expert_ranges[r] = (start, end)

    def get_prefetch_src_offset(self, peer_rank: int) -> int:
        """Get the expert offset within a peer's local weight tensor for prefetch."""
        peer_start, peer_end = self.peer_expert_ranges[peer_rank]
        if self.dwdp_rank < peer_rank:
            # Need tail of peer's experts
            prefetch_start = peer_end - self.num_prefetch_experts
        else:
            # Need head of peer's experts
            prefetch_start = peer_start
        return prefetch_start - peer_start  # offset in number of experts


# ---------------------------------------------------------------------------
# DwdpWeightView — weight bundle returned by get_weight_view()
# ---------------------------------------------------------------------------
@dataclass
class DwdpWeightView:
    """Dict-based multi-B weight tensor bundle for the MoE kernel.

    weights: {param_name: [rank0_tensor, rank1_tensor, ...]}
    Each list has dwdp_size entries in rank order. The entry at dwdp_rank
    is the local weight tensor (shape [num_experts_per_worker, ...]);
    others are prefetch buffer views (shape [num_prefetch_experts, ...]).
    """

    weights: Dict[str, List[torch.Tensor]]
    expert_size_per_partition: int  # num_experts_per_worker
    local_expert_start: int  # global expert ID offset


# ---------------------------------------------------------------------------
# DwdpManager
# ---------------------------------------------------------------------------
class DwdpManager:
    """Lifecycle manager for DWDP: init, prefetch, wait, get_weight_view."""

    def __init__(
        self,
        layout: DwdpExpertLayout,
        num_moe_layers: int,
        first_moe_layer_id: int,
        process_group: Any,  # torch ProcessGroup or GroupCoordinator
        num_fused_shared_experts: int = 0,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.first_moe_layer_id = first_moe_layer_id
        self.dwdp_size = layout.dwdp_size
        self.dwdp_rank = layout.dwdp_rank
        self._num_fused_shared = num_fused_shared_experts
        self._last_rank = self.dwdp_size - 1

        # Extract the raw ProcessGroup for dist.all_gather_object
        if hasattr(process_group, "cpu_group"):
            self._cpu_group = process_group.cpu_group
        elif hasattr(process_group, "device_group"):
            self._cpu_group = process_group.device_group
        else:
            self._cpu_group = process_group

        # Per-layer registered weights: {layer_id: {param_name: Tensor}}
        self._layer_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Per-layer shared expert weights: {layer_id: {param_name: Tensor}}
        self._shared_expert_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Pre-fused local tensor (routed+shared) for last rank: {layer_id: {param_name: Tensor}}
        self._local_fused: Dict[int, Dict[str, torch.Tensor]] = {}

        # Peer IPC opened tensors: {(peer_rank, layer_id, param_name): Tensor}
        self._peer_tensors: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._peer_device_indices: Dict[Tuple[int, int, str], int] = {}

        # Prefetch buffers (ping-pong double buffer)
        self._prefetch_buffers: Optional[
            List[Dict[str, List[Optional[torch.Tensor]]]]
        ] = None

        # CUDA streams and events — one stream per peer for parallel D2D copies
        self._prefetch_streams: Dict[int, torch.cuda.Stream] = {}
        # Per-layer, per-peer events: events[buf_idx][layer_slot][peer_rank]
        self._prefetch_events: Optional[List[List[Dict[int, torch.cuda.Event]]]] = None
        self._compute_events: Optional[List[List[torch.cuda.Event]]] = None
        self._hsa_direct_enabled = (
            os.environ.get("DWDP_USE_HSA_DIRECT_COPY", "0") == "1"
        )
        self._hsa_copier: Optional[_HsaDirectCopier] = None
        self._hsa_prefetch_signals: Optional[
            List[List[Dict[int, List[_HsaSignal]]]]
        ] = None
        self._hsa_wait_group_us: List[float] = []
        self._hsa_wait_group_count = 0
        self._hsa_timing_log_interval = int(
            os.environ.get("DWDP_HSA_TIMING_LOG_INTERVAL", "58")
        )

        self._initialized = False

        logger.info(
            f"[DWDP] Created DwdpManager: rank={self.dwdp_rank}, "
            f"size={self.dwdp_size}, "
            f"experts_per_worker={layout.num_experts_per_worker}, "
            f"prefetch_experts={layout.num_prefetch_experts}, "
            f"local_range=[{layout.local_expert_start}, {layout.local_expert_end}), "
            f"num_moe_layers={num_moe_layers}, first_moe_layer={first_moe_layer_id}"
        )

    # ----- Weight Registration -----

    def register_layer_weights(self, layer_id: int, weights: Dict[str, torch.Tensor]):
        """Register weight tensors for a MoE layer after model loading."""
        self._layer_weights[layer_id] = weights
        logger.debug(
            f"[DWDP] Registered weights for layer {layer_id}: "
            f"{list(weights.keys())}"
        )

    def register_shared_expert_weights(
        self, layer_id: int, weights: Dict[str, torch.Tensor]
    ):
        """Register shared expert weight slices for fused-shared-expert mode."""
        self._shared_expert_weights[layer_id] = weights

    # ----- IPC Handle Exchange -----

    def exchange_ipc_handles(self):
        """Exchange CUDA IPC handles for all registered weight tensors across ranks.

        Uses PyTorch native IPC: _share_cuda_() to get handles,
        _new_shared_cuda() to open peer handles. This is portable across
        CUDA and ROCm/HIP.
        """
        logger.info(
            f"[DWDP] Exchanging IPC handles for {len(self._layer_weights)} layers..."
        )

        for layer_id in sorted(self._layer_weights.keys()):
            weights = self._layer_weights[layer_id]
            for param_name, tensor in weights.items():
                # Get local IPC handle via PyTorch native API
                storage = tensor.untyped_storage()
                local_handle = storage._share_cuda_()

                # All-gather handles across DWDP group
                all_handles = [None] * self.dwdp_size
                dist.all_gather_object(all_handles, local_handle, group=self._cpu_group)

                # Open peer handles
                for peer_rank in range(self.dwdp_size):
                    if peer_rank == self.dwdp_rank:
                        continue

                    peer_handle = all_handles[peer_rank]
                    peer_device_idx = int(tuple(peer_handle)[0])
                    # Redirect to local device
                    device_idx = torch.cuda.current_device()
                    redirected_handle = (device_idx,) + tuple(peer_handle)[1:]

                    target_device = torch.device(f"cuda:{device_idx}")
                    with torch.cuda.device(target_device):
                        peer_storage = torch.UntypedStorage._new_shared_cuda(
                            *redirected_handle
                        )
                        # Reconstruct tensor with same shape/dtype/stride
                        peer_tensor = torch.empty(
                            0, dtype=tensor.dtype, device=target_device
                        ).set_(
                            peer_storage,
                            storage_offset=0,
                            size=tensor.shape,
                            stride=tensor.stride(),
                        )

                    self._peer_tensors[(peer_rank, layer_id, param_name)] = peer_tensor
                    self._peer_device_indices[(peer_rank, layer_id, param_name)] = (
                        peer_device_idx
                    )

        total_peers = len(self._peer_tensors)
        logger.info(
            f"[DWDP] IPC handle exchange complete: "
            f"{total_peers} peer tensor pointers opened"
        )

    # ----- Prefetch Buffer Allocation -----

    def init_prefetch_buffers(self):
        """Allocate double-buffered prefetch staging buffers."""
        if not self._layer_weights:
            logger.warning("[DWDP] No layer weights registered, skipping buffer init")
            return

        # Determine param shapes/dtypes from first registered layer
        first_layer_id = min(self._layer_weights.keys())
        first_weights = self._layer_weights[first_layer_id]
        if not first_weights:
            logger.warning("[DWDP] First layer has no weights, skipping buffer init")
            return

        device = next(iter(first_weights.values())).device
        num_prefetch = self.layout.num_prefetch_experts

        # Allocate ping-pong double buffers
        # buffers[buf_idx][param_name][peer_rank] = Tensor or None
        self._prefetch_buffers = []
        total_buffer_bytes = 0
        for buf_idx in range(2):
            buffer = {}
            for param_name, ref_tensor in first_weights.items():
                per_expert_shape = ref_tensor.shape[1:]  # strip expert dim
                tensor_list: List[Optional[torch.Tensor]] = [None] * self.dwdp_size
                for peer_rank in range(self.dwdp_size):
                    if peer_rank != self.dwdp_rank:
                        # Last rank buffer includes fused shared expert slot(s)
                        if peer_rank == self._last_rank and self._num_fused_shared > 0:
                            buf_shape = (
                                num_prefetch + self._num_fused_shared,
                                *per_expert_shape,
                            )
                        else:
                            buf_shape = (num_prefetch, *per_expert_shape)
                        t = torch.empty(
                            buf_shape, dtype=ref_tensor.dtype, device=device
                        )
                        tensor_list[peer_rank] = t
                        total_buffer_bytes += t.numel() * t.element_size()
                buffer[param_name] = tensor_list
            self._prefetch_buffers.append(buffer)

        # Build pre-fused local tensors for the last rank (routed + shared, one-time cat)
        if self._num_fused_shared > 0 and self.dwdp_rank == self._last_rank:
            for layer_id in sorted(self._layer_weights.keys()):
                fused_dict: Dict[str, torch.Tensor] = {}
                routed_weights = self._layer_weights[layer_id]
                shared_weights = self._shared_expert_weights.get(layer_id, {})
                for param_name, routed_tensor in routed_weights.items():
                    shared_tensor = shared_weights.get(param_name)
                    if shared_tensor is not None:
                        fused_dict[param_name] = torch.cat(
                            [routed_tensor, shared_tensor], dim=0
                        )
                    else:
                        fused_dict[param_name] = routed_tensor
                self._local_fused[layer_id] = fused_dict
            logger.info(
                f"[DWDP] Built {len(self._local_fused)} pre-fused local tensors "
                f"(routed + {self._num_fused_shared} shared expert)"
            )

        # Create per-peer prefetch streams (one per peer → parallel SDMA engines)
        self._prefetch_streams = {}
        for peer_rank in range(self.dwdp_size):
            if peer_rank != self.dwdp_rank:
                self._prefetch_streams[peer_rank] = torch.cuda.Stream(device=device)

        # Create per-layer, per-peer events for each buffer slot
        num_slots_per_buf = math.ceil(self.num_moe_layers / 2)
        peer_ranks = sorted(self._prefetch_streams.keys())
        self._prefetch_events = [
            [
                {pr: torch.cuda.Event(enable_timing=False) for pr in peer_ranks}
                for _ in range(num_slots_per_buf)
            ]
            for _ in range(2)
        ]
        self._compute_events = [
            [torch.cuda.Event(enable_timing=False) for _ in range(num_slots_per_buf)]
            for _ in range(2)
        ]
        self._hsa_prefetch_signals = [
            [{pr: [] for pr in peer_ranks} for _ in range(num_slots_per_buf)]
            for _ in range(2)
        ]

        if self._hsa_direct_enabled:
            try:
                self._hsa_copier = _HsaDirectCopier()
                logger.info(
                    f"[DWDP] HSA direct copy enabled: "
                    f"{len(self._hsa_copier.agents)} GPU agents discovered"
                )
            except Exception:
                logger.exception(
                    "[DWDP] Failed to initialize HSA direct copy; falling back to tensor.copy_()"
                )
                self._hsa_direct_enabled = False
                self._hsa_copier = None

        self._initialized = True
        logger.info(
            f"[DWDP] Prefetch buffers allocated: "
            f"2 x {self.dwdp_size - 1} peers x {num_prefetch} experts, "
            f"total ~{total_buffer_bytes / 1024**3:.2f} GB, "
            f"{len(self._prefetch_streams)} per-peer streams (multi-stream D2D), "
            f"hsa_direct={self._hsa_direct_enabled}"
        )

    def initialize_compute_events(self):
        """Pre-record first compute event for each buffer slot.

        Must be called before prefetch_first_layers() so that any subsequent
        wait_compute_event() has a valid event to wait on.
        """
        if not self._initialized:
            return
        current_stream = torch.cuda.current_stream()
        for buf_idx in range(2):
            self._compute_events[buf_idx][0].record(current_stream)

    # ----- Prefetch Operations -----

    def _moe_layer_idx(self, layer_id: int) -> int:
        """Convert global layer_id to MoE-layer index (0-based)."""
        return layer_id - self.first_moe_layer_id

    def _wait_hsa_signals(self, buf_idx: int, layer_slot: int):
        if not self._hsa_direct_enabled or self._hsa_copier is None:
            return
        if self._hsa_prefetch_signals is None:
            return

        profile_handle = self._hsa_copier._torch_profiler_enter(
            f"DWDP_HSA_WAIT_GROUP buf={buf_idx} layer_slot={layer_slot}"
        )
        wait_start_ns = time.perf_counter_ns()
        try:
            for signals in self._hsa_prefetch_signals[buf_idx][layer_slot].values():
                while signals:
                    signal = signals.pop(0)
                    self._hsa_copier.wait_and_destroy(signal)
        finally:
            wait_end_ns = time.perf_counter_ns()
            self._hsa_copier._torch_profiler_exit(profile_handle)
        self._hsa_wait_group_us.append((wait_end_ns - wait_start_ns) / 1000.0)
        self._hsa_wait_group_count += 1

        if (
            self._hsa_timing_log_interval > 0
            and self._hsa_wait_group_count % self._hsa_timing_log_interval == 0
        ):
            wait_groups = self._hsa_wait_group_us
            self._hsa_wait_group_us = []
            copy_summary = self._hsa_copier.timing_summary_and_reset()
            logger.info(
                f"[DWDP][HSA_TIMING] rank={self.dwdp_rank} "
                f"wait_groups={len(wait_groups)}, "
                f"wait_group_us[p50={_HsaDirectCopier._percentile(wait_groups, 0.50):.1f}, "
                f"p95={_HsaDirectCopier._percentile(wait_groups, 0.95):.1f}, "
                f"max={max(wait_groups):.1f}], "
                f"copies=({copy_summary or 'none'})"
            )

    def _hsa_copy_or_fallback(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        peer_rank: int,
        layer_id: int,
        param_name: str,
        signals: List[_HsaSignal],
        use_hsa_direct: bool = True,
    ):
        if (
            not use_hsa_direct
            or not self._hsa_direct_enabled
            or self._hsa_copier is None
        ):
            dst.copy_(src, non_blocking=True)
            return

        dst_device = dst.device.index
        if dst_device is None:
            dst_device = torch.cuda.current_device()
        src_device = self._peer_device_indices.get((peer_rank, layer_id, param_name))
        if src_device is None:
            logger.warning(
                f"[DWDP] Missing peer source device for rank={peer_rank}, "
                f"layer={layer_id}, param={param_name}; falling back to tensor.copy_()"
            )
            dst.copy_(src, non_blocking=True)
            return

        try:
            tag = (
                f"rank={self.dwdp_rank} peer={peer_rank} "
                f"layer={layer_id} param={param_name}"
            )
            signals.append(
                self._hsa_copier.copy_async(dst, src, dst_device, src_device, tag=tag)
            )
        except Exception:
            logger.exception(
                f"[DWDP] HSA direct copy failed for rank={peer_rank}, "
                f"layer={layer_id}, param={param_name}; falling back to tensor.copy_()"
            )
            dst.copy_(src, non_blocking=True)

    def _prefetch_layer(
        self, layer_id: int, wait_compute_layer_id: Optional[int] = None
    ):
        """Async prefetch peer weights for a single MoE layer into double buffer.

        Uses per-peer dedicated streams so D2D copies from different peers
        execute in parallel on independent SDMA engines.
        """
        if not self._initialized:
            return

        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2
        layer_slot = moe_idx // 2

        weights = self._layer_weights.get(layer_id)
        if weights is None:
            return

        # Precompute wait event (shared by all per-peer streams)
        wait_event = None
        if wait_compute_layer_id is not None:
            wait_moe_idx = self._moe_layer_idx(wait_compute_layer_id)
            wait_buf_idx = wait_moe_idx % 2
            wait_slot = wait_moe_idx // 2
            wait_event = self._compute_events[wait_buf_idx][wait_slot]

        in_graph_capture = torch.cuda.is_current_stream_capturing()
        use_hsa_direct = self._hsa_direct_enabled and not in_graph_capture

        # HSA copies cannot wait on a torch.cuda.Event directly. For the
        # experimental direct-copy path, conservatively wait on the host before
        # overwriting a buffer that was just consumed by compute.
        if use_hsa_direct and wait_event is not None:
            wait_event.synchronize()

        # Dispatch per-peer copies on per-peer streams (parallel SDMA)
        for peer_rank in range(self.dwdp_size):
            if peer_rank == self.dwdp_rank:
                continue

            stream = self._prefetch_streams[peer_rank]
            with torch.cuda.stream(stream):
                # Wait for compute to release this buffer slot (from 2 layers ago)
                if wait_event is not None:
                    stream.wait_event(wait_event)

                hsa_signals: List[_HsaSignal] = []
                if self._hsa_prefetch_signals is not None:
                    hsa_signals = self._hsa_prefetch_signals[buf_idx][layer_slot][
                        peer_rank
                    ]
                    hsa_signals.clear()

                src_expert_offset = self.layout.get_prefetch_src_offset(peer_rank)
                num_prefetch = self.layout.num_prefetch_experts

                for param_name in weights:
                    peer_tensor = self._peer_tensors.get(
                        (peer_rank, layer_id, param_name)
                    )
                    if peer_tensor is None:
                        continue

                    dst_buf = self._prefetch_buffers[buf_idx][param_name][peer_rank]
                    src_slice = peer_tensor.narrow(0, src_expert_offset, num_prefetch)

                    # Last rank with fused shared expert: copy routed into [:num_prefetch],
                    # then copy local shared expert into the extra slot(s)
                    if peer_rank == self._last_rank and self._num_fused_shared > 0:
                        routed_dst = dst_buf.narrow(0, 0, num_prefetch)
                        self._hsa_copy_or_fallback(
                            routed_dst,
                            src_slice,
                            peer_rank,
                            layer_id,
                            param_name,
                            hsa_signals,
                            use_hsa_direct=use_hsa_direct,
                        )
                        shared_src = self._shared_expert_weights.get(layer_id, {}).get(
                            param_name
                        )
                        if shared_src is not None:
                            dst_buf.narrow(
                                0, num_prefetch, self._num_fused_shared
                            ).copy_(shared_src, non_blocking=True)
                    else:
                        self._hsa_copy_or_fallback(
                            dst_buf,
                            src_slice,
                            peer_rank,
                            layer_id,
                            param_name,
                            hsa_signals,
                            use_hsa_direct=use_hsa_direct,
                        )

                # Signal per-peer completion
                self._prefetch_events[buf_idx][layer_slot][peer_rank].record(stream)

    def prefetch_first_layers(self):
        """Trigger async prefetch for the first 2 MoE layers.

        Called at the start of forward_extend(), before dense layers execute.
        The dense layers provide the initial prefetch hidden window.
        """
        if not self._initialized:
            return

        sorted_layer_ids = sorted(self._layer_weights.keys())
        if len(sorted_layer_ids) == 0:
            return

        # Prefetch first MoE layer into buf[0]
        self._prefetch_layer(sorted_layer_ids[0], wait_compute_layer_id=None)

        # Prefetch second MoE layer into buf[1] if exists
        if len(sorted_layer_ids) > 1:
            self._prefetch_layer(sorted_layer_ids[1], wait_compute_layer_id=None)

    def wait_prefetch(self, layer_id: int):
        """Wait for prefetch completion for a given MoE layer.

        Called on the default stream before MoE compute.
        Waits on all per-peer prefetch events for this layer.
        """
        if not self._initialized:
            return

        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2
        layer_slot = moe_idx // 2

        current_stream = torch.cuda.current_stream()
        self._wait_hsa_signals(buf_idx, layer_slot)
        for event in self._prefetch_events[buf_idx][layer_slot].values():
            current_stream.wait_event(event)

    def record_compute_and_prefetch_next(self, layer_id: int):
        """Record compute completion and trigger next layer's prefetch.

        Called on the default stream after MoE compute.
        """
        if not self._initialized:
            return

        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2
        layer_slot = moe_idx // 2

        # Record compute done on default stream
        torch.cuda.current_stream().record_event(
            self._compute_events[buf_idx][layer_slot]
        )

        # Trigger prefetch for layer_id + 2 (same buffer slot)
        sorted_layer_ids = sorted(self._layer_weights.keys())
        current_pos = sorted_layer_ids.index(layer_id)
        next_pos = current_pos + 2
        if next_pos < len(sorted_layer_ids):
            next_layer_id = sorted_layer_ids[next_pos]
            self._prefetch_layer(
                next_layer_id,
                wait_compute_layer_id=layer_id,
            )

    # ----- Weight View Assembly -----

    def get_weight_view(self, layer_id: int) -> DwdpWeightView:
        """Assemble multi-B weight view from local weights + prefetch buffers.

        Returns a DwdpWeightView with dict-based weight lists that can be
        concatenated or passed to multi-B kernel APIs.
        """
        local_weights = self._layer_weights.get(layer_id, {})
        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2

        local_fused = self._local_fused.get(layer_id, {})

        result_weights: Dict[str, List[torch.Tensor]] = {}
        for param_name, local_tensor in local_weights.items():
            tensor_list = []
            for rank in range(self.dwdp_size):
                if rank == self.dwdp_rank:
                    # Use pre-fused tensor if this is the last rank with shared expert
                    fused = local_fused.get(param_name)
                    if fused is not None:
                        tensor_list.append(fused)
                    else:
                        tensor_list.append(local_tensor)
                else:
                    if self._prefetch_buffers is not None:
                        buf_tensor = self._prefetch_buffers[buf_idx][param_name][rank]
                        tensor_list.append(buf_tensor)
                    else:
                        # Fallback: use peer tensor directly (no prefetch buffer)
                        peer_tensor = self._peer_tensors.get(
                            (rank, layer_id, param_name)
                        )
                        if peer_tensor is not None:
                            tensor_list.append(peer_tensor)
            result_weights[param_name] = tensor_list

        return DwdpWeightView(
            weights=result_weights,
            expert_size_per_partition=self.layout.num_experts_per_worker,
            local_expert_start=self.layout.local_expert_start,
        )

    # ----- Cleanup -----

    def cleanup(self):
        """Release IPC resources and buffers."""
        if self._hsa_prefetch_signals is not None and self._hsa_copier is not None:
            for buf_signals in self._hsa_prefetch_signals:
                for layer_signals in buf_signals:
                    for signals in layer_signals.values():
                        while signals:
                            self._hsa_copier.wait_and_destroy(signals.pop(0))
        self._peer_tensors.clear()
        self._peer_device_indices.clear()
        self._prefetch_buffers = None
        self._prefetch_streams.clear()
        self._prefetch_events = None
        self._compute_events = None
        self._hsa_prefetch_signals = None
        self._hsa_copier = None
        self._layer_weights.clear()
        self._shared_expert_weights.clear()
        self._local_fused.clear()
        self._initialized = False
        logger.info("[DWDP] Cleaned up DWDP resources")
