# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Set, Tuple

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_compiler_backend
from sglang.srt.utils.kt_accel import (
    kt_current_stream,
    kt_current_stream_handle,
    kt_device_synchronize,
)

from sglang.srt.layers.moe.kt_expert_masks import (
    ensure_kt_layer_masks,
    get_layer_gpu_experts_mask,
    get_layer_logical_to_gpu_index,
)
from sglang.srt.layers.moe.kt_stream_prefill import maybe_streaming_forward


if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)

# Streams subscribed for ACL aclrtProcessReport (torch_npu graph host callbacks).
_npu_report_subscribed: Set[int] = set()


def _npu_use_graph_host_callback(device: torch.device) -> bool:
    """True when MoE CPU work must be enqueued via ``_launch_host_func`` (graph capture)."""
    if device.type != "npu":
        return False
    try:
        if torch.npu.is_current_stream_capturing():
            return True
    except Exception:
        pass
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return get_is_capture_mode()
    except Exception:
        return False


def _ensure_npu_subscribe_report(stream) -> None:
    """Register ``stream`` with torch_npu ACL report thread (idempotent)."""
    key = int(stream.npu_stream)
    if key in _npu_report_subscribed:
        return
    import torch_npu

    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as e:
        # torch_npu >= 2.10 pre-subscribes capture streams inside NPUGraph, so a
        # second AclrtSubscribeReport on the same stream fails with error 107011.
        # The stream IS subscribed in that case; only swallow that specific error.
        if "107011" not in str(e):
            raise
    _npu_report_subscribed.add(key)


# KT_SIDE_STREAM=1 (experiment): run the CPU-MoE host callback on a dedicated
# side stream so the GPU-experts GroupedMatmul on the compute stream overlaps
# with the callback round trip / CPU MoE window, instead of being stream-order
# blocked behind it. fork/join events express the dependency:
#   compute: ...router... fork_ev ------------------- GPU experts -- wait(join_ev) -- merge
#   side:                   wait(fork_ev) D2H + host callback  join_ev
# Isolated microbenchmarks measured the fork/join machinery at ~6us/layer under
# graph replay and ~96% ideal hiding of min(callback, GPU work). Gain scales
# inversely with the off_cpu floor (judge by TPOT floor/p10, not off_cpu, which
# is invariant to overlap). One shared side stream; events created fresh per
# capture and kept alive for the lifetime of the captured graph.
_KT_SIDE_STREAM = os.environ.get("KT_SIDE_STREAM", "") == "1"
_kt_side_stream = None


def _get_kt_side_stream():
    global _kt_side_stream
    if _kt_side_stream is None:
        _kt_side_stream = torch.npu.Stream()
    return _kt_side_stream


# KT_SHARED_EXPERTS_STREAM=1 (experiment): run shared experts on their own
# stream, concurrent with the whole routed-experts span (which contains the
# CPU-MoE callback window). Sibling-fork topology (both side streams fork
# directly from the main stream); unlike SGLANG_NPU_USE_MULTI_STREAM's
# fork-from-fork nesting, this replays under NPUGraph. Used by
# DeepseekV2MoE.forward_normal.
KT_SHARED_EXPERTS_STREAM = os.environ.get("KT_SHARED_EXPERTS_STREAM", "") == "1"
_kt_shared_stream = None


def get_kt_shared_experts_stream():
    global _kt_shared_stream
    if _kt_shared_stream is None:
        _kt_shared_stream = torch.npu.Stream()
    return _kt_shared_stream


@torch.no_grad()
def _kt_npu_graph_host_forward(args) -> None:
    """Host callback: run CPU MoE on pinned buffers (used during NPUGraph capture)."""
    wrapper, hidden_states, stream_handle = args
    wrapper.run_pinned_forward_sync(hidden_states, stream_handle)


def resolve_kt_weight_path_for_layer(weight_path_template: str, layer_idx: int) -> str:
    """Expand ``--kt-weight-path`` for per-layer split GGUFs (e.g. DeepSeek-V4-Flash).

    Supported placeholders:

    - ``{layer_idx}`` — named, recommended (e.g. ``.../dsv4_layer{layer_idx}.gguf``).
    - A single anonymous ``{}`` — filled with the integer ``layer_idx``
      (e.g. ``.../dsv4_layer{}.gguf``).

    If there is no placeholder, the path is returned unchanged (one shared weight path).
    """
    if "{layer_idx}" in weight_path_template:
        return weight_path_template.format(layer_idx=layer_idx)
    if weight_path_template.count("{}") == 1:
        # Use replace (not str.format) so a lone ``{}`` is unambiguous. Some
        # deployment/YAML layers mangle ``{}``; ``{layer_idx}`` is safer (see launch script).
        return weight_path_template.replace("{}", str(layer_idx), 1)
    if weight_path_template.count("{}") > 1:
        logger.warning(
            "[KT] kt_weight_path contains multiple `{}` placeholders; "
            "using literal path without per-layer expansion."
        )
    return weight_path_template


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None
    gpu_experts_mask: Optional[torch.Tensor] = None
    logical_to_gpu_index: Optional[torch.Tensor] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if server_args.kt_weight_path is None:
        return None

    # Try to get num_layers from model config
    num_layers = None
    try:
        hf_config = server_args.get_model_config().hf_config
        num_layers = getattr(hf_config, "num_hidden_layers", None)
    except Exception:
        # If we can't get the config, num_layers will be None
        pass

    ensure_kt_layer_masks(server_args)
    gpu_experts_mask = get_layer_gpu_experts_mask(layer_idx)
    logical_to_gpu_index = get_layer_logical_to_gpu_index(layer_idx)
    num_gpu_experts = int(gpu_experts_mask.sum().item())

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=num_gpu_experts,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        gpu_experts_mask=gpu_experts_mask,
        logical_to_gpu_index=logical_to_gpu_index,
    )


# NOTE: deliberately NOT @torch.compile'd. On NPU ``get_compiler_backend()``
# returns the torchair backend, which lowers this into a *separate* NPU graph
# bound to the stream it was traced on. Inside sglang's outer NPU graph capture
# the model runs on a different stream at replay, so a torchair-compiled inner
# graph raises ``Unsupport run graph with different stream``. The ops below are
# trivial on-device indexing/``where`` (masks are pre-moved to device in
# ``process_weights_after_loading``), so eager execution folds cleanly into the
# outer graph with negligible cost.
def mask_cpu_expert_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make ``topk_ids/topk_weights`` safe for NPU grouped matmul.

    GPU slots use ``logical_to_gpu_index[logical_id]`` (supports prefix and
    frequency placement). CPU experts map to ``(expert 0, weight 0)`` — see
    docstring in prior revision for why ``-1`` is unsafe on NPU routing kernels.
    """
    mask_on_device = gpu_experts_mask.to(topk_ids.device)
    l2g_on_device = logical_to_gpu_index.to(topk_ids.device)
    is_gpu = mask_on_device[topk_ids]
    gpu_slots = l2g_on_device[topk_ids]
    safe_ids = torch.where(is_gpu, gpu_slots, torch.zeros_like(topk_ids))
    safe_weights = torch.where(is_gpu, topk_weights, torch.zeros_like(topk_weights))
    return safe_ids, safe_weights


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask
        self.logical_to_gpu_index = kt_config.logical_to_gpu_index
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            resolved_weight_path = resolve_kt_weight_path_for_layer(
                self.kt_config.weight_path, self.kt_config.layer_idx
            )
            gpu_experts_mask = self.gpu_experts_mask
            if gpu_experts_mask is None:
                gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
                gpu_experts_mask[: self.num_gpu_experts] = True
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=gpu_experts_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=None,
                weight_path=resolved_weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights (NZ-cast the 32 resident experts)
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # Reserve the streaming HBM slot now (model-load time, before KV-pool sizing) so the
        # KV pool auto-accounts for it; also registers this layer for the dynamic decode-
        # resident pool. No-op unless KT_PREFILL_STREAM=1. See kt_stream_prefill.
        from sglang.srt.layers.moe.kt_stream_prefill import maybe_reserve_slot

        maybe_reserve_slot(self, layer.w13_weight.device, layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            kt_device_synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

        # 3. Pre-move GPU-expert placement masks to the device ONCE, here at a
        # deterministic pre-capture point. On NPU there is NO eager warmup
        # forward before graph capture, so the very first forward runs under
        # capture — a per-forward ``.to(topk_ids.device)`` inside
        # ``mask_cpu_expert_routing`` would do a synchronous host->device
        # ``aclrtMemcpy`` that ACL capture mode forbids (error 107030). With the
        # masks already on-device, ``.to(same_device)`` short-circuits to a
        # no-op (no memcpy, no sync). Only needed when GPU experts exist; in
        # all-CPU mode (``num_gpu_experts == 0``) ``apply`` never calls
        # ``mask_cpu_expert_routing`` and the masks may be None.
        if self.num_gpu_experts > 0 and self.gpu_experts_mask is not None:
            device = layer.w13_weight.device
            if self.gpu_experts_mask.device != device:
                self.gpu_experts_mask = self.gpu_experts_mask.to(device)
            if (
                self.logical_to_gpu_index is not None
                and self.logical_to_gpu_index.device != device
            ):
                self.logical_to_gpu_index = self.logical_to_gpu_index.to(device)

        # 4. Pre-subscribe the ACL report stream before graph capture begins.
        # The graph host-callback path (``_submit_cpu_npu_graph``) issues
        # ``torch_npu.npu._subscribe_report`` lazily on first use; since the
        # first forward IS the capture forward on NPU, that subscription would
        # otherwise be issued during capture. Pre-warming it here (idempotent,
        # guarded by the ``_npu_report_subscribed`` set) keeps the capture-time
        # path free of first-time ACL control calls. Best-effort: failures are
        # non-fatal (the lazy path still runs).
        if self.tp_rank == 0 and self.wrapper is not None:
            try:
                device = layer.w13_weight.device
                if device.type == "npu":
                    _ensure_npu_subscribe_report(kt_current_stream(device))
                    # The side-stream variant launches the host callback on the
                    # side stream — pre-subscribe it too, for the same reason.
                    if _KT_SIDE_STREAM:
                        _ensure_npu_subscribe_report(_get_kt_side_stream())
            except Exception as exc:  # pragma: no cover - hardening only
                logger.warning(
                    "[KT] pre-capture ACL report subscribe failed (non-fatal): %s",
                    exc,
                )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, kt_current_stream_handle(x.device)
        )

    def sync(self, x: torch.Tensor, *, cpu_already_synced: bool = False) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information
            cpu_already_synced: If True, only copy pinned CPU output to device (graph host path).

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        if cpu_already_synced:
            return self.wrapper.copy_forward_output_to_device(x)
        return self.wrapper.sync_forward(x, kt_current_stream_handle(x.device))

    def _submit_cpu_npu_graph(
        self,
        dispatch_output: "StandardDispatchOutput",
        x: torch.Tensor,
    ) -> None:
        """Enqueue CPU MoE via ``_launch_host_func`` so it is captured in NPUGraph."""
        import torch_npu

        assert self.wrapper is not None
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        stream = kt_current_stream(x.device)
        _ensure_npu_subscribe_report(stream)
        stream_handle = kt_current_stream_handle(x.device)
        self.wrapper.copy_inputs_to_cpu_buffers(x, topk_ids, topk_weights)
        torch_npu.npu._launch_host_func(
            stream,
            _kt_npu_graph_host_forward,
            (self.wrapper, x, stream_handle),
        )

    def kt_ascend_pre_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_output: "TopKOutput",
    ):
        """Pre-dispatch seam for the AscendTP dispatcher.

        The AscendTP dispatcher permutes (and may quantize) tokens before the
        quant method runs, so both the CPU-side submission (which needs the
        raw token-major activations) and the CPU-expert routing mask (the
        dispatch permutation groups rows by expert id, so masking after the
        fact is too late) must happen BEFORE ``dispatcher.dispatch``.

        Returns ``(bypass_output, masked_topk_output, join_state)``:
        ``bypass_output`` is the finished [t, hidden] MoE output when the
        streaming-prefill path took the whole batch (skip dispatch entirely);
        otherwise it is None and ``masked_topk_output``/``join_state`` feed
        the dispatch and the post-combine ``kt_ascend_join``.
        """
        from types import SimpleNamespace

        x = hidden_states
        layer_idx = self.kt_config.layer_idx

        _stream_out = maybe_streaming_forward(self, layer_idx, x, topk_output)
        if _stream_out is not None:
            return _stream_out.hidden_states, None, None

        # Carrier so submit()/_submit_cpu_npu_graph() (written against
        # StandardDispatchOutput) run unchanged on the raw tensors.
        carrier = SimpleNamespace(hidden_states=x, topk_output=topk_output)

        use_npu_graph = (
            self.tp_rank == 0
            and self.wrapper is not None
            and _npu_use_graph_host_callback(x.device)
        )
        pending_join = None
        if use_npu_graph:
            if _KT_SIDE_STREAM:
                comp_stream = kt_current_stream(x.device)
                side_stream = _get_kt_side_stream()
                fork_ev = torch.npu.Event()
                join_ev = torch.npu.Event()
                if not hasattr(self, "_kt_side_events"):
                    self._kt_side_events = []
                self._kt_side_events.append((fork_ev, join_ev))
                fork_ev.record(comp_stream)
                side_stream.wait_event(fork_ev)
                with torch.npu.stream(side_stream):
                    self._submit_cpu_npu_graph(carrier, x)
                pending_join = (join_ev, side_stream, comp_stream)
            else:
                self._submit_cpu_npu_graph(carrier, x)
        elif self.tp_rank == 0:
            self.submit(layer, carrier)

        masked_topk_output = topk_output
        if self.num_gpu_experts > 0:
            masked_topk_ids, masked_topk_weights = mask_cpu_expert_routing(
                topk_output.topk_ids,
                topk_output.topk_weights,
                self.gpu_experts_mask,
                self.logical_to_gpu_index,
            )
            masked_topk_output = topk_output._replace(
                topk_ids=masked_topk_ids,
                topk_weights=masked_topk_weights,
            )
        return None, masked_topk_output, (use_npu_graph, pending_join)

    def kt_ascend_join(self, x: torch.Tensor, join_state) -> Optional[torch.Tensor]:
        """Sync the CPU-side result submitted at ``kt_ascend_pre_dispatch``.

        Called after ``dispatcher.combine`` so both sides are token-major
        [t, hidden]. Returns None on ranks that did not submit.
        """
        if self.tp_rank != 0:
            return None
        use_npu_graph, pending_join = join_state
        if pending_join is not None:
            join_ev, side_stream, comp_stream = pending_join
            join_ev.record(side_stream)
            comp_stream.wait_event(join_ev)
        return self.sync(x, cpu_already_synced=use_npu_graph)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if not hasattr(dispatch_output, "topk_output"):
            # AscendTP dispatch: tokens are already permuted/quantized and the
            # CPU side was submitted at the kt_ascend_pre_dispatch seam in
            # FusedMoE.forward_impl — this call is purely the resident-expert
            # GPU pass.
            return self.gpu_method.apply(layer, dispatch_output)

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        layer_idx = self.kt_config.layer_idx

        # Streaming-prefill bypass (env-gated): long prefill streams all 256 experts on NPU.
        # Returns None when not applicable -> falls through to the hybrid path below (untouched).
        _stream_out = maybe_streaming_forward(self, layer_idx, x, topk_output)
        if _stream_out is not None:
            return _stream_out

        use_npu_graph = (
            self.tp_rank == 0
            and self.wrapper is not None
            and _npu_use_graph_host_callback(x.device)
        )

        # Step 1: Submit CPU expert computation (non-blocking, or graph host callback)
        pending_join = None
        if use_npu_graph:
            if _KT_SIDE_STREAM:
                comp_stream = kt_current_stream(x.device)
                side_stream = _get_kt_side_stream()
                fork_ev = torch.npu.Event()
                join_ev = torch.npu.Event()
                # Keep event objects alive as long as the captured graph: the
                # graph's record/wait nodes reference them.
                if not hasattr(self, "_kt_side_events"):
                    self._kt_side_events = []
                self._kt_side_events.append((fork_ev, join_ev))
                fork_ev.record(comp_stream)
                side_stream.wait_event(fork_ev)
                with torch.npu.stream(side_stream):
                    # D2H input copies + _launch_host_func all enqueue on the
                    # side stream (kt_current_stream resolves to it here).
                    self._submit_cpu_npu_graph(dispatch_output, x)
                pending_join = (join_ev, side_stream, comp_stream)
            else:
                self._submit_cpu_npu_graph(dispatch_output, x)
        elif self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2 & 3: GPU expert computation — only if we actually have GPU experts.
        #
        # In all-CPU mode (``--kt-num-gpu-experts 0``) the underlying quant
        # method created ``layer.w13_weight`` / ``layer.w2_weight`` with shape
        # ``[0, ...]`` (no GPU expert weights loaded → no HBM cost). Calling
        # ``self.gpu_method.apply`` would then invoke ``npu_grouped_matmul``
        # with an empty weight tensor and CANN raises
        # ``RuntimeError: Invalid inputs: neither x nor weight could be empty``.
        # This guard is symmetric to the early-return ``torch.zeros_like(x)``
        # in ``submit/sync`` when ``tp_rank != 0`` / ``wrapper is None``.
        if self.num_gpu_experts > 0:
            # CPU rows (id >= num_gpu_experts) are remapped to (expert 0, weight 0)
            # — see ``mask_cpu_expert_routing`` docstring for why we do not use
            # ``-1`` here (would crash NPU ``npu_moe_init_routing/compute_expert_tokens``).
            topk_ids = topk_output.topk_ids
            topk_weights = topk_output.topk_weights
            masked_topk_ids, masked_topk_weights = mask_cpu_expert_routing(
                topk_ids,
                topk_weights,
                self.gpu_experts_mask,
                self.logical_to_gpu_index,
            )

            masked_topk_output = topk_output._replace(
                topk_ids=masked_topk_ids,
                topk_weights=masked_topk_weights,
            )
            masked_dispatch_output = dispatch_output._replace(
                topk_output=masked_topk_output
            )

            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states
        else:
            output = torch.zeros_like(x)

        # Step 4: Synchronize CPU results and merge with GPU results
        if self.tp_rank == 0:
            if pending_join is not None:
                join_ev, side_stream, comp_stream = pending_join
                # Compute stream resumes only after the side stream's host
                # callback has returned (CPU output buffers are then valid for
                # the H2D copy inside ``sync``).
                join_ev.record(side_stream)
                comp_stream.wait_event(join_ev)
            cpu_output = self.sync(x, cpu_already_synced=use_npu_graph)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def map_logical_expert_id_for_gpu_load(self, logical_expert_id: int) -> int:
        """Map checkpoint logical expert id to NPU weight slot, or -1 if CPU-only."""
        if self.logical_to_gpu_index is None:
            if logical_expert_id < self.num_gpu_experts:
                return logical_expert_id
            return -1
        return int(self.logical_to_gpu_index[logical_expert_id].item())

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
