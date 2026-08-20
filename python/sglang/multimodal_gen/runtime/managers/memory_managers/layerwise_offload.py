import bisect
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Dict, List, Set, Tuple

import torch
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_RESIDENCY_GROUPS,
    LAYERWISE_OFFLOAD,
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
    describe_host_memory,
    module_weight_bytes,
    pin_benefit_bytes,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    RESIDENCY_POLICIES,
    RESIDENCY_POLICY_LEADING,
    RESIDENCY_POLICY_STRIDED,
    is_dit_component_name,
    layerwise_component_matches_any_selection,
    normalize_layerwise_offload_components,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def compute_streamed_layers(
    *, num_layers: int, resident_layers: int, policy: str
) -> tuple[int, ...]:
    """Which layer indices are streamed rather than held on the GPU.

    Both policies stream the same *count* of layers, so they cost the same
    memory and move the same bytes. They differ only in when those bytes move:

    ``leading``  keeps layers ``0..r-1`` and streams the tail. Every streamed
                 layer sits next to another streamed layer, so the transfers
                 arrive as one burst confined to the last ``(n-r)/n`` of the
                 step, and each has exactly one layer of compute to hide behind.

    ``strided``  spreads the streamed layers evenly across the whole step, so
                 the same bytes move over ``n`` layers instead of ``n-r`` and
                 the peak concurrent traffic drops by ``n/(n-r)``.

    What that buys is contention, not bandwidth and not stalls. Profiling the
    two policies on an 8-GPU run shows the same HtoD volume to within 0.1%, the
    copy engines about half idle in both, and only a handful of long gaps in
    either. What differs is how much traffic is in flight beside the compute:
    under ``strided`` the GEMM, the attention and the sequence-parallel
    all-to-all each run measurably faster (-1.5%, -0.7%, -0.5%) without any
    kernel changing, which is the whole of the -0.5% end to end.

    Returned sorted, and always exactly ``num_layers - resident_layers`` long.
    """
    if policy not in RESIDENCY_POLICIES:
        raise ValueError(
            f"unknown residency policy {policy!r}, expected one of {RESIDENCY_POLICIES}"
        )
    resident = min(max(0, resident_layers), num_layers)
    streamed_count = num_layers - resident
    if streamed_count <= 0:
        return ()
    if resident <= 0:
        return tuple(range(num_layers))

    if policy == RESIDENCY_POLICY_LEADING:
        return tuple(range(resident, num_layers))

    # The step num_layers / streamed_count is >= 1 here (resident > 0 was
    # handled above), so round() of the ramp is strictly increasing and the
    # indices cannot collide -- the partition is total by construction, which
    # test_both_policies_partition_the_stack pins.
    return tuple(
        round(index * num_layers / streamed_count) for index in range(streamed_count)
    )


# Adapted from skywork AI Infra diffusion optimize
# Below this a table is not worth a per-request round trip; above it the ratio
# of table size to rows actually read makes residency clearly wasteful.
HOST_RESIDENT_TABLE_MIN_BYTES = 256 * 1024**2


def _resolve_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module | None:
    current: Any = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, torch.nn.Module) else None


def _host_resident_tables(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Declared vocab tables large enough that device residency is waste.

    A table is read by gather, not by GEMM: one row per token, so a 512-token
    prompt touches 8 MiB of umT5-XXL's 3.91 GiB table. Streaming it layer by
    layer would be worse than resident -- 3.91 GiB moved to read 8 MiB -- so it
    belongs in host memory with the lookup running there.

    Opt-in per model rather than discovered by shape. The bridge is a forward
    hook, so it only covers the table's own ``__call__``; a model that also
    reads the weight directly -- a tied ``lm_head``, a functional gather inside
    a third-party backbone -- would see a host tensor mid-graph. Only a model
    whose table is reached solely through its forward may list it.
    """
    tables = []
    for module in model.modules():
        for path in getattr(module, "host_resident_table_names", ()) or ():
            table = _resolve_submodule(module, path)
            weight = getattr(table, "weight", None)
            if weight is None or not hasattr(weight, "dim") or weight.dim() != 2:
                continue
            # A sharded table is already divided by the world size, and its
            # output feeds an all-reduce that expects a device tensor.
            if getattr(table, "tp_size", 1) != 1:
                continue
            if weight.numel() * weight.element_size() < HOST_RESIDENT_TABLE_MIN_BYTES:
                continue
            if table not in tables:
                tables.append(table)
    return tables


def detach_host_resident_tables(
    model: torch.nn.Module,
) -> List[Tuple[torch.nn.Module, torch.Tensor]]:
    """Swap large vocab tables for placeholders so a `.to(device)` skips them."""
    detached = []
    for module in _host_resident_tables(model):
        weight = module.weight
        detached.append((module, weight.data))
        weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
    return detached


def restore_host_resident_tables(
    detached: List[Tuple[torch.nn.Module, torch.Tensor]],
    device: torch.device | str,
) -> None:
    for module, data in detached:
        module.weight.data = data
        _install_host_gather_hooks(module, device)
        logger.info(
            "Keeping %s (%.2f GiB) in host memory: a gather reads one row per "
            "token, so residency buys almost nothing.",
            type(module).__name__,
            data.numel() * data.element_size() / (1024**3),
        )


def _install_host_gather_hooks(
    module: torch.nn.Module, device: torch.device | str
) -> None:
    """Run this module's gather on the host, move only the result."""

    def _inputs_to_host(_module, args, kwargs):
        if not args or not torch.is_tensor(args[0]):
            return None
        return (args[0].to("cpu"),) + args[1:], kwargs

    def _output_to_device(_module, _args, output):
        if not torch.is_tensor(output):
            return output
        return output.to(device, non_blocking=True)

    module.register_forward_pre_hook(_inputs_to_host, with_kwargs=True)
    module.register_forward_hook(_output_to_device)


class LayerwiseOffloadManager:
    """A lightweight layerwise CPU offload manager.

    This utility offloads per-layer parameters/buffers from GPU to CPU, and
    supports async H2D prefetch using a dedicated CUDA stream. MPS uses
    synchronous per-layer transfers from the checkpoint-backed CPU tensors.

    Typical usage:
    - Construct the manager with the target model and the list-like module
      attribute that represents transformer blocks (e.g. ``blocks``).
    - Call :meth:`initialize` once to offload weights and prefetch layer 0.
    - During forward, call :meth:`prefetch_layer` for the next layer and
      :meth:`release_layer` for the finished layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layers_attr_str: str,
        num_layers: int,
        enabled: bool,
        pin_cpu_memory: bool = True,
        prefetch_size: int = 1,
        resident_layers: int = 0,
        initialize: bool = True,
        residency_policy: str = RESIDENCY_POLICY_LEADING,
    ) -> None:
        self.model = model
        self.layers_attr_str = layers_attr_str
        self.num_layers = num_layers
        self._synchronous_mps = current_platform.is_mps()
        # mps shares physical memory with the CPU and has no pinned host memory
        # or CUDA-style copy streams
        self.pin_cpu_memory = bool(pin_cpu_memory and not self._synchronous_mps)
        # an explicit MPS zero avoids staging the next layer alongside the
        # active one; MPS has no transfer overlap to recover from that cost
        self.prefetch_size = (
            0
            if current_platform.is_mps() and prefetch_size == 0
            else min(max(1, prefetch_size), self.num_layers)
        )
        # Layers held on GPU across denoise steps, instead of being re-streamed
        # every step. `residency_policy` picks *which* layers those are; see
        # compute_streamed_layers for why the choice is not cosmetic.
        self.resident_layers = min(max(0, int(resident_layers)), self.num_layers)
        self.residency_policy = residency_policy
        self._streamed_order = compute_streamed_layers(
            num_layers=self.num_layers,
            resident_layers=self.resident_layers,
            policy=residency_policy,
        )
        self._resident_set = frozenset(range(self.num_layers)) - set(
            self._streamed_order
        )
        # Armed on the first denoise forward, so that the load-time prefetch below
        # does not pin the whole resident set before the DiT is the active component.
        self._residency_active = False
        # True once _initialize builds the CPU buffers; unlike `enabled` it
        # never flips back, so disable_offload/enable_offload can toggle
        # `enabled` without losing track of which managers can be re-armed.
        self._configured = False
        self.enabled = bool(enabled and torch.get_device_module().is_available())
        if not self.enabled:
            return
        self.device = (
            current_platform.get_local_torch_device()
            if current_platform.is_mps()
            else torch.device(
                current_platform.device_type,
                torch.get_device_module().current_device(),
            )
        )
        self.copy_stream = (
            None if self._synchronous_mps else torch.get_device_module().Stream()
        )

        # ``named_parameters()`` is relative to ``model``, just like the path in
        # ``layers_attr_str``. Anchor the match so a manager for top-level
        # ``blocks`` cannot also capture an unrelated nested list such as
        # ``token_refiner.blocks`` whose forward hooks run at a different time.
        self._layer_name_re = re.compile(
            rf"^{re.escape(layers_attr_str)}\.(?P<layer_idx>\d+)(\.|$)"
        )

        # layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
        # stores the consolidated weight from a same layer, of same dtype
        self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        # layer_idx -> {name: pinned_cpu_tensor_with_original_stride}
        # stores tensors whose original non-contiguous stride/layout must be preserved
        self._strided_cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # mps keeps the original CPU tensor for each layer instead of building a
        # second flattened host copy
        self._mps_cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape}}
        # stores the offset and numel of each weight from a same layer, of same dtype
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()
        # layer_idx -> torch.get_device_module().Event for fine-grained sync, to make sure the weight is resident in pre-hook
        self._prefetch_events: Dict[int, torch.get_device_module().Event] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        self._offload_placeholders: Dict[torch.dtype, torch.Tensor] = {}
        self._has_dtensor_weights = False
        # Store forward hooks for removal
        self._forward_hooks: List[Any] = []

        if initialize:
            self._initialize()

    def initialize(self) -> None:
        self._initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group("layer_idx"))
        except Exception:
            return None

    def _managed_parameter_bytes(self) -> int:
        total_bytes = 0
        for name, tensor in self.model.named_parameters():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            local_tensor = self._to_local_tensor(tensor)
            total_bytes += local_tensor.numel() * local_tensor.element_size()
        return total_bytes

    def _get_shared_empty_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        placeholder = self._offload_placeholders.get(dtype)
        if placeholder is None:
            placeholder = torch.empty((1,), device=self.device, dtype=dtype)
            self._offload_placeholders[dtype] = placeholder
        return placeholder

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, DTensor):
            return tensor.to_local()
        return tensor

    def _wrap_for_target(
        self, target: torch.Tensor, local_tensor: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(target, DTensor):
            return DTensor.from_local(
                local_tensor, target.device_mesh, target.placements
            )
        return local_tensor

    def _get_shared_empty_tensor_for_target(
        self, target: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._wrap_for_target(target, self._get_shared_empty_tensor(dtype))

    @staticmethod
    def _get_alignment_numel(dtype: torch.dtype, alignment_bytes: int = 32) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        return max(1, alignment_bytes // element_size)

    @classmethod
    def _align_numel_offset(
        cls, offset: int, dtype: torch.dtype, alignment_bytes: int = 32
    ) -> int:
        alignment_numel = cls._get_alignment_numel(dtype, alignment_bytes)
        remainder = offset % alignment_numel
        if remainder == 0:
            return offset
        return offset + alignment_numel - remainder

    @torch.compiler.disable
    def _initialize(self) -> None:
        if not self.enabled:
            return

        if self._synchronous_mps:
            self._named_parameters = dict(self.model.named_parameters())
            self._named_buffers = dict(self.model.named_buffers())
            self._initialize_mps_cpu_weights()
            return

        self._initialize_layer_weights()

        # Keep non-layer parameters resident on GPU. Layer tensors have already
        # been replaced by tiny device placeholders, so this does not reload the
        # offloaded layer weights.
        host_resident = detach_host_resident_tables(self.model)
        if not self._has_dtensor_weights:
            self.model.to(self.device)
        restore_host_resident_tables(host_resident, self.device)

        self._finalize_initialization()

    def _initialize_layer_weights(self) -> None:
        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        # 1. collect and group layer parameters by dtype. Keep buffers resident:
        # shared buffers such as RoPE caches may be referenced by many layers.
        layer_groups: Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]] = {}
        for name, tensor in self._named_parameters.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            self._has_dtensor_weights = self._has_dtensor_weights or isinstance(
                tensor, DTensor
            )
            local_tensor = self._to_local_tensor(tensor)
            layer_groups.setdefault(layer_idx, {}).setdefault(
                local_tensor.dtype, []
            ).append((name, tensor))

        # 2. concat and offload (in pinned memory)
        for layer_idx, dtype_to_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._strided_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}

            for dtype, weights in dtype_to_params.items():
                contiguous_weights: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
                for name, weight in weights:
                    local_weight = self._to_local_tensor(weight)
                    if local_weight.is_contiguous():
                        contiguous_weights.append((name, weight, local_weight))
                        continue

                    # Preserve non-contiguous layouts such as the transposed FP8
                    # weight views expected by CUTLASS kernels.
                    cpu_tensor = torch.empty_strided(
                        size=local_weight.shape,
                        stride=local_weight.stride(),
                        dtype=dtype,
                        pin_memory=self.pin_cpu_memory,
                    )
                    cpu_tensor.copy_(local_weight)
                    self._strided_cpu_weights[layer_idx][name] = cpu_tensor
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "shape": local_weight.shape,
                        "stride": local_weight.stride(),
                        "preserve_strides": True,
                    }
                    weight.data = self._get_shared_empty_tensor_for_target(
                        weight, dtype
                    )

                if not contiguous_weights:
                    continue

                current_offset = 0
                aligned_offsets: Dict[str, int] = {}
                for name, weight, local_weight in contiguous_weights:
                    # Some fused diffusion kernels require tensor base pointers to
                    # satisfy a 32-byte alignment contract. Reusing one flat buffer
                    # is still fine, but each logical tensor slice must start on an
                    # aligned offset inside that buffer.
                    current_offset = self._align_numel_offset(current_offset, dtype)
                    aligned_offsets[name] = current_offset
                    current_offset += local_weight.numel()

                total_numel = current_offset

                # create concatenated CPU buffer (in pinned memory)
                cpu_buffer = torch.empty(
                    total_numel, dtype=dtype, pin_memory=self.pin_cpu_memory
                )

                # offload weights to the buffer
                for name, weight, local_weight in contiguous_weights:
                    current_offset = aligned_offsets[name]
                    numel = local_weight.numel()
                    cpu_buffer[current_offset : current_offset + numel].copy_(
                        local_weight.flatten()
                    )
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": local_weight.shape,
                        "stride": local_weight.stride(),
                        "preserve_strides": False,
                    }

                    weight.data = self._get_shared_empty_tensor_for_target(
                        weight, dtype
                    )

                    current_offset += numel

                self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

    def _finalize_initialization(self) -> None:
        # prefetch the head of the stream for warm-up; residency is not armed
        # yet, so this is layer 0 regardless of policy
        self.prepare_for_next_req(non_blocking=False)

        self.register_forward_hooks()
        self._configured = True
        logger.debug(
            f"LayerwiseOffloadManager initialized with num prefetched layer: {self.prefetch_size}, num resident layers: {self.resident_layers}, total num layers: {self.num_layers}, residency policy: {self.residency_policy}"
        )
        if self.residency_policy == RESIDENCY_POLICY_STRIDED and self._streamed_order:
            # Printed because the layout is the whole point of the policy, and
            # "did it actually stride?" is otherwise only answerable from a
            # profile.
            logger.debug(
                "Strided residency streams layers %s (%d of %d)",
                list(self._streamed_order),
                len(self._streamed_order),
                self.num_layers,
            )

    def _head_of_stream(self) -> list[int]:
        """The first layers the coming forward will have to stream in.

        Before residency is armed nothing is pinned, so the forward starts at
        layer 0 like any other; afterwards the first streamed layer is whichever
        the policy put first, which under `strided` is not necessarily layer 0.
        """
        count = min(self.prefetch_size, self.num_layers)
        if not self._residency_active:
            return list(range(count))
        return self._next_streamed(after=-1, count=count)

    @torch.compiler.disable
    def _initialize_mps_cpu_weights(self) -> None:
        for name, tensor in self._named_parameters.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            local_tensor = self._to_local_tensor(tensor).detach()
            cpu_tensor = (
                local_tensor
                if local_tensor.device.type == "cpu"
                else local_tensor.to("cpu")
            )
            self._mps_cpu_weights.setdefault(layer_idx, {})[name] = cpu_tensor
            self._weight_metadata.setdefault(layer_idx, {})[name] = {
                "dtype": cpu_tensor.dtype,
            }
            tensor.data = self._get_shared_empty_tensor_for_target(
                tensor, cpu_tensor.dtype
            )

        torch.mps.empty_cache()
        self.register_forward_hooks()
        self._configured = True
        logger.info(
            f"Initialized synchronous MPS layerwise offload with {self.num_layers} layers"
        )

    def prepare_for_next_req(self, non_blocking=True):
        """
        Prepare for the next round of denoising loop with prefetching the necessary layers
        """
        # The resident set first: it has to be there for the whole step, and the
        # caller decides whether to block on it.
        for layer_idx in sorted(self._retained_set):
            self.prefetch_layer(layer_idx, non_blocking=non_blocking)
        if not non_blocking and self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        # The head of the stream is issued after that wait, and always
        # asynchronously. wait_stream drains the whole copy stream, so issuing
        # it first would make the caller block on a layer it does not need yet:
        # this runs from the layer-0 pre-hook on every denoise step, and under
        # `leading` the first streamed layer is `resident_layers` away, one full
        # transfer (~48 ms on the 50-layer H3 DiT) ahead of a layer 0 that is
        # already pinned. The per-layer wait_event in the pre-hook blocks
        # exactly when the weights are needed and no earlier.
        for layer_idx in self._head_of_stream():
            self.prefetch_layer(layer_idx, non_blocking=True)

    @property
    def holds_residents(self) -> bool:
        """True if this manager keeps a resident layer set beyond the streaming
        prefetch window, so it must be denoise-stage-scoped."""
        return self.enabled and self.resident_layers > 0

    @property
    def _retained_layers(self) -> int:
        """How many layers are currently held across denoise steps; 0 until armed."""
        return self.resident_layers if self._residency_active else 0

    @property
    def _retained_set(self) -> frozenset[int]:
        """Which layers are currently held across denoise steps; empty until armed."""
        return self._resident_set if self._residency_active else frozenset()

    def _next_streamed(self, *, after: int, count: int) -> List[int]:
        """The next ``count`` streamed layers after ``after``, wrapping around.

        Under ``leading`` this is just the following indices, but under
        ``strided`` the immediate successor is usually resident, so prefetching
        ``after + 1`` would be a no-op and the real next transfer would not
        start until its own layer was already running.
        """
        total = len(self._streamed_order)
        if total == 0:
            return []
        start = bisect.bisect_right(self._streamed_order, after)
        return [
            self._streamed_order[(start + offset) % total]
            for offset in range(min(count, total))
        ]

    @torch.compiler.disable
    def _activate_residency(self) -> None:
        """Arm the resident set on the first denoise forward. The pinning itself is
        done by the ``prepare_for_next_req`` that follows in the same hook."""
        self._residency_active = True

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """get the target model weight/buffer to be replaced"""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """
        idempotent
        """
        if not self.enabled or self.device is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if self._synchronous_mps:
            cpu_weights = self._mps_cpu_weights.get(layer_idx)
            if not cpu_weights:
                return
            with torch.inference_mode(False), torch.no_grad():
                for name, cpu_tensor in cpu_weights.items():
                    target = self.get_target_with_name(name)
                    target.data = self._wrap_for_target(
                        target,
                        cpu_tensor.to(device=self.device, non_blocking=False),
                    )
            self._gpu_layers.add(layer_idx)
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return
        if self.copy_stream is not None:
            self.copy_stream.wait_stream(torch.get_device_module().current_stream())
            stream_context = torch.get_device_module().stream(self.copy_stream)
        else:
            # the device has no CUDA-like stream or pinned-memory support
            non_blocking = False
            stream_context = nullcontext()

        # create gpu buffer and load from CPU buffer
        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        with (
            torch.inference_mode(False),
            torch.no_grad(),
            stream_context,
        ):
            for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
                gpu_buffer = torch.empty(
                    cpu_buffer.shape, dtype=dtype, device=self.device
                )
                gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                gpu_buffers[dtype] = gpu_buffer

            # restore model's weights by their metadata using the same copy stream
            # so the recorded event covers both flat-buffer and stride-preserving copies.
            for name, meta in self._weight_metadata[layer_idx].items():
                target = self.get_target_with_name(name)
                if meta.get("preserve_strides", False):
                    # Recreate the original view layout instead of flatten+view.
                    # ModelOpt FP8 relies on a transposed runtime weight layout,
                    # so preserving stride is part of correctness, not just an
                    # optimization detail.
                    cpu_tensor = self._strided_cpu_weights[layer_idx][name]
                    gpu_tensor = torch.empty_strided(
                        size=meta["shape"],
                        stride=meta["stride"],
                        dtype=meta["dtype"],
                        device=self.device,
                    )
                    gpu_tensor.copy_(cpu_tensor, non_blocking=non_blocking)
                    target.data = self._wrap_for_target(target, gpu_tensor)
                    continue

                dtype = meta["dtype"]
                gpu_buffer = gpu_buffers[dtype]

                # map the parameter's data to the correct slice of the GPU buffer
                local_tensor = gpu_buffer[
                    meta["offset"] : meta["offset"] + meta["numel"]
                ].view(meta["shape"])
                target.data = self._wrap_for_target(target, local_tensor)

        if self.copy_stream is not None:
            # record after all copies so the consumer waits for every weight copy
            event = torch.get_device_module().Event()
            event.record(self.copy_stream)
            self._prefetch_events[layer_idx] = event

        self._gpu_layers.add(layer_idx)

    @torch.compiler.disable
    def release_layer(self, layer_idx: int, force: bool = False) -> None:
        """
        lightweight release layer weights
        Basically set the reference count to the gpu weight tensor to zero. The weights on cpu is untouched

        Resident layers are kept across denoise steps
        """
        if not self.enabled or self.device is None:
            return

        if not force and layer_idx in self._retained_set:
            return

        # clear prefetch event, since it's useless and needs to be reset
        self._prefetch_events.pop(layer_idx, None)

        if layer_idx not in self._gpu_layers:
            return

        with torch.inference_mode(False), torch.no_grad():
            for name, meta in self._weight_metadata.get(layer_idx, {}).items():
                target = self.get_target_with_name(name)
                # Wraparound prefetch will reload the layer when it is needed again
                target.data = self._get_shared_empty_tensor_for_target(
                    target, meta["dtype"]
                )

        self._gpu_layers.discard(layer_idx)
        if self._synchronous_mps:
            # mps dispatch is asynchronous, so a tensor rebinding alone leaves
            # prior layer allocations live until the command buffer drains
            torch.mps.synchronize()
            torch.mps.empty_cache()

    @torch.compiler.disable
    def release_all(self) -> None:
        """Release every layer, including the resident ones: this ends the
        denoise stage that the resident set is scoped to."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.release_layer(layer_idx, force=True)

    @torch.compiler.disable
    def load_all_layers(self) -> None:
        """Load all layers from CPU to GPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        for layer_idx in range(self.num_layers):
            if layer_idx not in self._gpu_layers:
                self.prefetch_layer(layer_idx, non_blocking=False)

    @torch.compiler.disable
    def sync_layer_to_cpu(self, layer_idx: int) -> None:
        """Sync a layer's weights from GPU back to CPU."""
        if not self.enabled or layer_idx not in self._gpu_layers:
            return
        if self._synchronous_mps:
            # inference does not mutate parameters; retain the original mapped
            # tensor instead of materializing a CPU copy after every layer
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return

        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        # Collect current GPU weights and write back to CPU buffer
        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            target_local = self._to_local_tensor(target)
            if meta.get("preserve_strides", False):
                self._strided_cpu_weights[layer_idx][name].copy_(target_local.cpu())
                continue

            gpu_weight = target_local.flatten().cpu()

            dtype = meta["dtype"]
            cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
            offset = meta["offset"]
            numel = meta["numel"]
            cpu_buffer[offset : offset + numel].copy_(gpu_weight)

    @torch.compiler.disable
    def sync_all_layers_to_cpu(self) -> None:
        """Sync all loaded layers' weights from GPU back to CPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.sync_layer_to_cpu(layer_idx)

    @torch.compiler.disable
    def update_cpu_weights(
        self, weight_dict: Dict[str, torch.Tensor]
    ) -> Set[str] | None:
        """Update consolidated CPU buffers with new weights.

        When layerwise offload (--dit-layerwise-offload) is enabled, the
        offload manager replaces GPU parameters with small torch.empty((1,))
        placeholders while real weights live in consolidated pinned CPU
        buffers.

        The refit process writes new weights directly into the CPU buffers,
        bypassing the placeholders.  For any layer that happens to be resident
        on the GPU at update time, the live GPU tensor is also updated.

        Args:
            weight_dict: Mapping of parameter name to new weight tensor.

        Returns:
            Set of parameter names that were successfully updated.

        Raises:
            ValueError: If a weight's shape does not match the recorded
                metadata (i.e., the real shape, not the placeholder shape).
        """
        if not self.enabled:
            return None

        updated_names: Set[str] = set()
        if self._synchronous_mps:
            for name, loaded_weight in weight_dict.items():
                layer_idx = self._match_layer_idx(name)
                if layer_idx is None:
                    continue
                cpu_tensor = self._mps_cpu_weights.get(layer_idx, {}).get(name)
                if cpu_tensor is None:
                    continue
                if tuple(cpu_tensor.shape) != tuple(loaded_weight.shape):
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"expected={tuple(cpu_tensor.shape)}, "
                        f"loaded={tuple(loaded_weight.shape)}"
                    )
                replacement = loaded_weight.to(
                    device="cpu", dtype=cpu_tensor.dtype
                ).detach()
                self._mps_cpu_weights[layer_idx][name] = replacement
                if layer_idx in self._gpu_layers:
                    target = self.get_target_with_name(name)
                    target.data = self._wrap_for_target(
                        target,
                        replacement.to(device=target.device, dtype=target.dtype),
                    )
                updated_names.add(name)
            return updated_names

        for name, loaded_weight in weight_dict.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None:
                continue
            meta_layer = self._weight_metadata.get(layer_idx)
            if meta_layer is None or name not in meta_layer:
                continue

            meta = meta_layer[name]
            local_loaded_weight = self._to_local_tensor(loaded_weight)
            if tuple(meta["shape"]) != tuple(local_loaded_weight.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={tuple(meta['shape'])}, "
                    f"loaded={tuple(local_loaded_weight.shape)}"
                )

            dtype = meta["dtype"]
            if meta.get("preserve_strides", False):
                self._strided_cpu_weights[layer_idx][name].copy_(
                    local_loaded_weight.to(dtype=dtype)
                )
            else:
                offset = meta["offset"]
                numel = meta["numel"]
                cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                cpu_buffer[offset : offset + numel].copy_(
                    local_loaded_weight.to(dtype=dtype).flatten()
                )

            # If this layer is currently on GPU, update the live parameter.
            if layer_idx in self._gpu_layers:
                target = self.get_target_with_name(name)
                target_local = self._to_local_tensor(target)
                target_local.copy_(local_loaded_weight.to(dtype=target_local.dtype))

            updated_names.add(name)

        return updated_names

    def iter_cpu_weights(self):
        """Yield (name, tensor) pairs from consolidated CPU buffers.

        This reconstructs the original weight tensors (with correct shapes)
        from the flat CPU buffers using stored metadata.  Unlike
        model.named_parameters(), which returns (1,) placeholders
        when offload is enabled, this method returns the real weights and
        can be used for checksum computation.
        """
        if self._synchronous_mps:
            for layer_idx in sorted(self._mps_cpu_weights):
                yield from self._mps_cpu_weights[layer_idx].items()
            return

        for layer_idx in sorted(self._weight_metadata):
            for name, meta in self._weight_metadata[layer_idx].items():
                if meta.get("preserve_strides", False):
                    # Some quantized weights rely on a non-contiguous layout.
                    # Yield the strided tensor directly instead of rebuilding it
                    # from the flat buffer, which would silently lose the
                    # original stride information.
                    yield name, self._strided_cpu_weights[layer_idx][name]
                    continue

                dtype = meta["dtype"]
                offset = meta["offset"]
                numel = meta["numel"]
                shape = meta["shape"]
                cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                yield name, cpu_buffer[offset : offset + numel].reshape(shape)

    def register_forward_hooks(self) -> None:
        if not self.enabled:
            return

        layers = dict(self.model.named_modules())[self.layers_attr_str]

        def make_pre_hook(i):
            def hook(module, input):
                if i == 0:
                    self._activate_residency()
                    self.prepare_for_next_req(non_blocking=False)
                if i not in self._gpu_layers:
                    # LTX audio VAE traverses decoder.up in reverse order
                    self.prefetch_layer(i, non_blocking=False)
                if i in self._prefetch_events and self.copy_stream is not None:
                    torch.get_device_module().current_stream().wait_event(
                        self._prefetch_events[i]
                    )

                if self.residency_policy == RESIDENCY_POLICY_STRIDED:
                    # Top up the stream at every layer rather than in bursts of
                    # prefetch_size. Under `strided` the next streamed layer can
                    # be several layers away, so a burst schedule keyed on index
                    # arithmetic would either skip it or issue it late; asking
                    # for "the next N streamed layers" is the same request every
                    # layer and prefetch_layer is idempotent, so the repeats are
                    # free. This is what buys the wider hiding window: the
                    # transfer is issued as soon as the previous streamed layer
                    # is done with, not one layer before it is needed.
                    for layer_to_prefetch in self._next_streamed(
                        after=i, count=self.prefetch_size
                    ):
                        self.prefetch_layer(layer_to_prefetch, non_blocking=True)
                # trigger batch prefetch (i + prefetch_size ~ i + 2 * prefetch_size) if needed
                elif self.prefetch_size and i % self.prefetch_size == 0:
                    for j in range(i + self.prefetch_size, i + 2 * self.prefetch_size):
                        layer_to_prefetch = j % self.num_layers
                        self.prefetch_layer(layer_to_prefetch, non_blocking=True)

            return hook

        def make_post_hook(i):
            def hook(module, input, output):
                # previous, we wait here, until the copy stream for next layer is finished,
                # now with any prefetch_size, only wait for the copy stream, when the copy stream is for the next layer
                self.release_layer(i)

            return hook

        # register prefetch & release hooks for each layer
        self._forward_hooks.clear()
        for i, layer in enumerate(layers):
            pre_hook_handle = layer.register_forward_pre_hook(make_pre_hook(i))
            post_hook_handle = layer.register_forward_hook(make_post_hook(i))
            self._forward_hooks.extend([pre_hook_handle, post_hook_handle])

    def remove_forward_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook_handle in self._forward_hooks:
            hook_handle.remove()
        self._forward_hooks.clear()


class LayerwiseOffloadableModuleMixin:
    """A mixin that registers forward hooks to enable layerwise offload."""

    # whether the current module is selected by the `dit` group
    layerwise_offload_dit_group_enabled: bool = True
    # H3 has a large packed-sequence working set on MPS, so its non-block
    # weights are materialized only for the subphase that consumes them
    mps_stream_non_layer_weights: bool = False

    # The list of names of this module's layer/block ModuleList or Sequential attributes.
    layer_names: List[str] = []

    # Dotted paths to gather-only vocab tables that may stay in host memory
    # under layerwise offload. See _host_resident_tables for what qualifies.
    host_resident_table_names: List[str] = []
    layerwise_offload_managers: list[LayerwiseOffloadManager] = []

    def _capture_mps_cpu_non_layer_weights(self) -> None:
        managed_names = {
            name
            for manager in self.layerwise_offload_managers
            for weights in manager._mps_cpu_weights.values()
            for name in weights
        }
        self._mps_cpu_non_layer_parameters = {
            name: parameter.detach()
            for name, parameter in self.named_parameters()
            if name not in managed_names
        }
        self._mps_cpu_buffers = {
            name: buffer.detach() for name, buffer in self.named_buffers()
        }
        if not self.mps_stream_non_layer_weights:
            return

        parameters = dict(self.named_parameters())
        for name, tensor in self._mps_cpu_non_layer_parameters.items():
            parameters[name].data = torch.empty(
                (1,),
                dtype=tensor.dtype,
                device=current_platform.get_local_torch_device(),
            )
        buffers = dict(self.named_buffers())
        for name, tensor in self._mps_cpu_buffers.items():
            buffers[name].data = torch.empty(
                (1,),
                dtype=tensor.dtype,
                device=current_platform.get_local_torch_device(),
            )

    @staticmethod
    def _matches_mps_weight_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
        return any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes
        )

    def materialize_mps_non_layer_weights(self, *prefixes: str) -> None:
        if not current_platform.is_mps() or not self.mps_stream_non_layer_weights:
            return
        selected_prefixes = tuple(prefixes)
        with torch.inference_mode(False), torch.no_grad():
            parameters = dict(self.named_parameters())
            for name, tensor in self._mps_cpu_non_layer_parameters.items():
                if self._matches_mps_weight_prefix(name, selected_prefixes):
                    parameters[name].data = tensor.to(
                        current_platform.get_local_torch_device()
                    )
            buffers = dict(self.named_buffers())
            for name, tensor in self._mps_cpu_buffers.items():
                if self._matches_mps_weight_prefix(name, selected_prefixes):
                    buffers[name].data = tensor.to(
                        current_platform.get_local_torch_device()
                    )

    def release_mps_non_layer_weights(self, *prefixes: str) -> None:
        if not current_platform.is_mps() or not self.mps_stream_non_layer_weights:
            return
        selected_prefixes = tuple(prefixes)
        with torch.inference_mode(False), torch.no_grad():
            parameters = dict(self.named_parameters())
            for name, tensor in self._mps_cpu_non_layer_parameters.items():
                if self._matches_mps_weight_prefix(name, selected_prefixes):
                    parameters[name].data = torch.empty(
                        (1,),
                        dtype=tensor.dtype,
                        device=current_platform.get_local_torch_device(),
                    )
            buffers = dict(self.named_buffers())
            for name, tensor in self._mps_cpu_buffers.items():
                if self._matches_mps_weight_prefix(name, selected_prefixes):
                    buffers[name].data = torch.empty(
                        (1,),
                        dtype=tensor.dtype,
                        device=current_platform.get_local_torch_device(),
                    )
        torch.mps.synchronize()
        torch.mps.empty_cache()

    def restore_mps_cpu_non_layer_weights(self) -> None:
        if not current_platform.is_mps():
            return
        with torch.inference_mode(False), torch.no_grad():
            parameters = dict(self.named_parameters())
            for name, tensor in self._mps_cpu_non_layer_parameters.items():
                parameters[name].data = tensor
            buffers = dict(self.named_buffers())
            for name, tensor in self._mps_cpu_buffers.items():
                buffers[name].data = tensor

    def configure_layerwise_offload(
        self,
        server_args: ServerArgs,
        *,
        pin_budget: HostPinBudget | None = None,
        component_name: str | None = None,
    ):
        self.layerwise_offload_managers = []
        named_modules = dict(self.named_modules())
        configured_layer_names = []
        # `--dit-*` is the group default these fall back to, not a scope.
        prefetch_value, resident_value, residency_policy = (
            server_args.layerwise_tuning_for(
                component_name,
                dit_group=self.layerwise_offload_dit_group_enabled,
            )
        )
        for layer_name in self.layer_names:
            module_list = named_modules.get(layer_name)
            if not isinstance(module_list, (torch.nn.ModuleList, torch.nn.Sequential)):
                continue
            if len(module_list) == 0:
                continue

            num_layers = len(module_list)
            if current_platform.is_mps() and prefetch_value == 0.0:
                prefetch_size = 0
            elif prefetch_value < 1.0:
                prefetch_size = 1 + int(round(prefetch_value * (num_layers - 1)))
            else:
                prefetch_size = int(prefetch_value)

            if resident_value <= 0:
                resident_layers = 0
            elif resident_value < 1.0:
                resident_layers = max(1, int(round(resident_value * num_layers)))
            else:
                resident_layers = min(num_layers, int(resident_value))

            # Pinning these weights is what lets the copy stream run ahead of
            # compute, but pinned pages are the ones the kernel cannot reclaim,
            # so a component only gets them while the budget lasts.
            pin_cpu_memory = server_args.pin_cpu_memory
            if pin_cpu_memory and pin_budget is not None:
                pin_cpu_memory = pin_budget.request(
                    component_name=f"{component_name or type(self).__name__}.{layer_name}",
                    weight_bytes=module_weight_bytes(module_list),
                )

            manager = LayerwiseOffloadManager(
                model=self,
                layers_attr_str=layer_name,
                num_layers=num_layers,
                enabled=True,
                pin_cpu_memory=pin_cpu_memory,
                prefetch_size=prefetch_size,
                resident_layers=resident_layers,
                initialize=False,
                residency_policy=residency_policy,
            )
            self.layerwise_offload_managers.append(manager)
            configured_layer_names.append(layer_name)

        if current_platform.is_mps():
            for manager in self.layerwise_offload_managers:
                manager.initialize()
            self._capture_mps_cpu_non_layer_weights()
        else:
            enabled_managers = [
                manager
                for manager in self.layerwise_offload_managers
                if manager.enabled
            ]
            initialization_order = sorted(
                enabled_managers,
                key=lambda manager: manager._managed_parameter_bytes(),
                reverse=True,
            )
            # release the largest managed groups first when checkpoint loading
            # already placed weights on the accelerator; keep the stored manager
            # order unchanged for prefetch and forward lifecycle semantics
            for manager in initialization_order:
                manager._initialize_layer_weights()

            # Every managed layer group must be replaced before moving the
            # remaining parameters, otherwise an earlier manager transiently
            # moves later groups to the device.
            if enabled_managers and not any(
                manager._has_dtensor_weights for manager in enabled_managers
            ):
                device = enabled_managers[0].device
                host_resident = detach_host_resident_tables(self)
                self.to(device)
                restore_host_resident_tables(host_resident, device)

            for manager in enabled_managers:
                manager._finalize_initialization()

        if configured_layer_names:
            logger.debug(
                "Enabled layerwise offload for %s on modules: %s",
                self.__class__.__name__,
                configured_layer_names,
            )
        else:
            logger.debug(
                "No layerwise-offloadable ModuleList found for %s. Candidates: %s",
                self.__class__.__name__,
                self.layer_names,
            )

    def prepare_for_next_req(self):
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            manager.prepare_for_next_req(non_blocking=True)

    def disable_offload(self) -> None:
        """Disable layerwise offload: load all layers to GPU and remove hooks.

        Also flips `manager.enabled` off so every layerwise path —
        is_layerwise_offloaded_module(), release_all(), prepare_for_next_req()
        — short-circuits until enable_offload() re-arms it. Without this, a
        residency strategy built while the module was offloaded (e.g. the
        temporary offload_during_compile window) keeps calling release_all()
        on use-site switches after the hooks are gone, replacing restored
        weights with (1,) placeholders that nothing swaps back in.
        """
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.remove_forward_hooks()
                manager.load_all_layers()
                manager.enabled = False

    def enable_offload(self) -> None:
        """Re-enable layerwise offload: sync weights to CPU, release layers, and restore hooks."""
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager._configured:
                manager.enabled = True
                manager.sync_all_layers_to_cpu()
                manager.release_all()
                manager.register_forward_hooks()


def iter_materialized_weights(module: torch.nn.Module):
    """Yield (name, tensor) pairs with materialized weights, even under offload.

    When layerwise offload is active, module.named_parameters() returns
    (1,) placeholders for offloaded layers.  This function reads the
    actual data from the offload manager's CPU buffers and chains it with
    the non-offloaded parameters.
    """
    offload_managers: list = []
    if is_layerwise_offloaded_module(module):
        offload_managers = [m for m in module.layerwise_offload_managers if m.enabled]

    if not offload_managers:
        yield from module.named_parameters()
        return

    # Collect offloaded names and their real tensors from CPU buffers.
    offloaded_names: set[str] = set()
    for manager in offload_managers:
        for name, tensor in manager.iter_cpu_weights():
            offloaded_names.add(name)
            yield name, tensor

    # Yield non-offloaded parameters (e.g. final norms, embeddings).
    for name, param in module.named_parameters():
        if name not in offloaded_names:
            yield name, param


def is_layerwise_offloaded_module(module: torch.nn.Module) -> bool:
    return isinstance(module, LayerwiseOffloadableModuleMixin) and any(
        manager.enabled for manager in module.layerwise_offload_managers
    )


def is_resident_layerwise_module(module: torch.nn.Module) -> bool:
    """True if the module keeps a resident DiT layer set beyond the streaming
    prefetch window.
    """
    return isinstance(module, LayerwiseOffloadableModuleMixin) and any(
        manager.holds_residents for manager in module.layerwise_offload_managers
    )


def get_layerwise_offload_component_names_for_pipeline(
    modules: Mapping[str, object],
    component_names: Sequence[str] | None = None,
) -> list[str]:
    """Resolve layerwise selectors against the current pipeline modules.

    Explicit unsupported component names are kept so callers can report them.
    """
    normalized_component_names = normalize_layerwise_offload_components(component_names)
    selected_component_names = (
        set(normalized_component_names)
        if normalized_component_names is not None
        else None
    )

    if selected_component_names is None:
        return [
            component_name
            for component_name, module in modules.items()
            if isinstance(module, LayerwiseOffloadableModuleMixin)
            and module.layerwise_offload_dit_group_enabled
        ]

    if LAYERWISE_OFFLOAD_ALL_COMPONENTS in selected_component_names:
        return [
            component_name
            for component_name, module in modules.items()
            if isinstance(module, torch.nn.Module)
        ]

    explicit_component_names = selected_component_names - {LAYERWISE_OFFLOAD_DIT_GROUP}
    select_dit_group = LAYERWISE_OFFLOAD_DIT_GROUP in selected_component_names
    selected_pipeline_component_names: list[str] = []
    for component_name, module in modules.items():
        if layerwise_component_matches_any_selection(
            component_name, explicit_component_names
        ):
            selected_pipeline_component_names.append(component_name)
            continue
        if select_dit_group and (
            is_dit_component_name(component_name)
            or (
                isinstance(module, LayerwiseOffloadableModuleMixin)
                and module.layerwise_offload_dit_group_enabled
            )
        ):
            selected_pipeline_component_names.append(component_name)
    return selected_pipeline_component_names


def configure_layerwise_offload_modules(
    modules: Mapping[str, object],
    server_args: ServerArgs,
    component_names: Sequence[str] | None = None,
    warn_missing: bool = True,
) -> list[str]:
    """Configure layerwise offload for the given modules, from the given component_names

    Args:
        modules: the dict of {component_name: component}, containing the components to be chosen from
        component_names: list of component names. component with names not in this list shouldn't be configured

    Returns a list of component names of modules configured to be layerwise-offload
    """

    # components which has already been configured to be layerwise-offload
    configured_component_names: list[str] = []
    configured_module_ids: set[int] = set()
    normalized_component_names = normalize_layerwise_offload_components(component_names)
    selected_component_names = (
        set(normalized_component_names)
        if normalized_component_names is not None
        else None
    )
    select_all = (
        selected_component_names is not None
        and LAYERWISE_OFFLOAD_ALL_COMPONENTS in selected_component_names
    )
    exact_layerwise_selectors = {
        selector
        for selector, mode in (server_args.component_residency or {}).items()
        if mode == LAYERWISE_OFFLOAD and selector not in COMPONENT_RESIDENCY_GROUPS
    }
    if server_args.component_residency is not None:
        selected_pipeline_component_names = [
            component_name
            for component_name, module in modules.items()
            if server_args.residency_mode(component_name) == LAYERWISE_OFFLOAD
            and (
                isinstance(module, torch.nn.Module)
                or component_name in exact_layerwise_selectors
            )
        ]
    else:
        selected_pipeline_component_names = (
            get_layerwise_offload_component_names_for_pipeline(
                modules,
                normalized_component_names,
            )
        )

    if (
        warn_missing
        and server_args.component_residency is not None
        and server_args.disagg_role == RoleType.MONOLITHIC
    ):
        missing_component_names = sorted(exact_layerwise_selectors - modules.keys())
        if missing_component_names:
            logger.warning(
                "Layerwise offload components are not currently loaded: %s. "
                "Available pipeline components: %s",
                missing_component_names,
                sorted(modules),
            )

    if warn_missing and selected_component_names is not None and not select_all:
        explicit_component_names = selected_component_names - {
            LAYERWISE_OFFLOAD_DIT_GROUP
        }
        missing_component_names = [
            selected_component_name
            for selected_component_name in explicit_component_names
            if not any(
                layerwise_component_matches_any_selection(
                    component_name, [selected_component_name]
                )
                for component_name in modules
            )
        ]
        if missing_component_names:
            logger.warning(
                "Layerwise offload components are not currently loaded: %s. "
                "Available pipeline components: %s",
                sorted(missing_component_names),
                sorted(modules),
            )

    unsupported_component_names = [
        component_name
        for component_name in selected_pipeline_component_names
        if not isinstance(modules[component_name], LayerwiseOffloadableModuleMixin)
    ]
    explicit_unsupported_component_names = [
        component_name
        for component_name in unsupported_component_names
        if (warn_missing and server_args.component_residency is None)
        or server_args.is_explicit_layerwise_offload_component(component_name)
    ]
    if explicit_unsupported_component_names:
        raise ComponentResidencyError(
            "Components selected for layerwise-offload do not support it: "
            f"{sorted(explicit_unsupported_component_names)}"
        )
    if unsupported_component_names:
        for component_name in unsupported_component_names:
            server_args.record_component_layerwise_capability(
                component_name, supported=False
            )
        selected_pipeline_component_names = [
            component_name
            for component_name in selected_pipeline_component_names
            if component_name not in unsupported_component_names
        ]
        logger.warning(
            "Auto layerwise selection skipped unsupported components; their "
            "existing placement remains active: %s",
            sorted(unsupported_component_names),
        )

    def _default_num_inference_steps() -> int:
        from sglang.multimodal_gen.registry import get_pipeline_config_classes

        pipeline_class_name = server_args.pipeline_class_name
        if not pipeline_class_name:
            return 1
        config_classes = get_pipeline_config_classes(pipeline_class_name)
        if config_classes is None:
            return 1
        return max(1, int(config_classes[1]().num_inference_steps))

    default_steps = _default_num_inference_steps()

    def _h2d_bytes_a_pin_would_save(name: str) -> int:
        """What pinning this component is worth, in bytes moved per request.

        A DiT under layerwise offload re-streams its layers on every denoise
        step; everything else transfers once. Ranking on the product rather
        than on "is it the DiT" matters for few-step models, where a large
        one-shot text encoder can move more bytes per request than a small DiT
        stepped four times.
        """
        module = modules[name]
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            return 0
        return pin_benefit_bytes(
            weight_bytes=module_weight_bytes(module),
            uses_per_request=(
                default_steps if module.layerwise_offload_dit_group_enabled else 1
            ),
        )

    # Offer the budget in descending order of what a pin saves, so the bytes
    # that would move most often claim it first. sorted() is stable, so equal
    # rankings keep their original order.
    selected_pipeline_component_names = sorted(
        selected_pipeline_component_names,
        key=_h2d_bytes_a_pin_would_save,
        reverse=True,
    )
    pin_budget = HostPinBudget()
    logger.info("Layerwise offload: %s", describe_host_memory())

    for component_name in selected_pipeline_component_names:
        module = modules[component_name]
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            continue
        server_args.record_component_layerwise_capability(
            component_name, supported=True
        )
        module_id = id(module)
        if module_id in configured_module_ids:
            # avoid duplicated configures on a same module
            continue

        configured_module_ids.add(module_id)

        if not is_layerwise_offloaded_module(module):
            module.configure_layerwise_offload(
                server_args, pin_budget=pin_budget, component_name=component_name
            )
        if not is_layerwise_offloaded_module(module):
            raise ComponentResidencyError(
                f"Component {component_name!r} did not enable layerwise offload"
            )
        configured_component_names.append(component_name)

    if configured_component_names:
        # Report where the weights ended up, not just which components opted
        # in. The loader's per-component line runs before this, so it can only
        # ever describe the pre-offload placement.
        from sglang.multimodal_gen.runtime.loader.utils import (
            format_component_residency,
        )

        logger.info(
            "Enabled layerwise offload for pipeline components: %s",
            ", ".join(
                f"{name} ({format_component_residency(modules[name])})"
                for name in configured_component_names
            ),
        )
    elif warn_missing:
        logger.debug("No selected pipeline component enabled layerwise offload")
    return configured_component_names
