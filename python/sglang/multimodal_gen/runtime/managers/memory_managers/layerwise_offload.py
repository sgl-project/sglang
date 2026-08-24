import bisect
import queue
import re
import threading
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.utils import MappedRegions
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_RESIDENCY_GROUPS,
    LAYERWISE_OFFLOAD,
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
    describe_host_memory,
    host_copies_would_not_fit,
    host_memory_available_bytes,
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

# Parking a component's non-layer weights frees device memory at the cost of two
# transfers per use and a host copy that competes with the page cache. It is
# worth that only when what it frees is a meaningful share of the headroom
# actually available; on a card with room it is pure loss. Below this share of
# free device memory, the component stays where it is.
PARK_SIGNIFICANCE = 0.1


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


class MappedLayerCourier:
    """Ships a mapped layer's weights to the device off the compute thread.

    A copy whose source is a checkpoint mapping is synchronous however it is
    requested -- the driver stages unpinned memory through its own buffer -- and
    the prefetch hooks run on the compute thread, so every such copy stalls the
    step. This worker thread reads the mapped bytes into a pinned slot (a plain
    memcpy when the page cache holds them) and issues the device copy from
    there on its own stream, where it is genuinely asynchronous. The compute
    thread's large tensor copies release the GIL, so reading layer i+1 really
    does overlap computing layer i.

    The thread only prepares device tensors and records an event; parameters
    are rebound on the compute thread at collect time, so module state is never
    touched concurrently.
    """

    _NUM_SLOTS = 2

    def __init__(
        self,
        *,
        mapped_cpu_weights: Dict[int, Dict[str, torch.Tensor]],
        weight_metadata: Dict[int, Dict[str, Dict[str, Any]]],
        device: torch.device,
        pin_slots: bool,
    ) -> None:
        self._mapped_cpu_weights = mapped_cpu_weights
        self._weight_metadata = weight_metadata
        self._device = device
        slot_bytes = max(
            (
                sum(t.numel() * t.element_size() for t in weights.values())
                for weights in mapped_cpu_weights.values()
                if weights
            ),
            default=0,
        )
        if slot_bytes <= 0:
            raise ValueError("no mapped weights to ship")
        self._slots = [
            torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=pin_slots)
            for _ in range(self._NUM_SLOTS)
        ]
        self._slot_events: List[Optional[Any]] = [None] * self._NUM_SLOTS
        self._stream = torch.get_device_module().Stream()
        self._tasks: queue.Queue[Optional[int]] = queue.Queue()
        self._results: Dict[int, Any] = {}
        self._ready = threading.Condition()
        self._pending: Set[int] = set()
        self._broken = False
        self._thread = threading.Thread(
            target=self._run, name="mapped-layer-courier", daemon=True
        )
        self._thread.start()

    def submit(self, layer_idx: int) -> bool:
        """Queue a layer. False when the courier is out of service."""
        if self._broken:
            return False
        with self._ready:
            if layer_idx in self._pending or layer_idx in self._results:
                return True
            self._pending.add(layer_idx)
        self._tasks.put(layer_idx)
        return True

    def pending(self, layer_idx: int) -> bool:
        with self._ready:
            return layer_idx in self._pending or layer_idx in self._results

    def collect(self, layer_idx: int):
        """Block until the layer is shipped; (event, {name: gpu_tensor})."""
        with self._ready:
            while layer_idx not in self._results:
                if self._broken and layer_idx not in self._results:
                    raise RuntimeError("mapped-layer courier stopped")
                self._ready.wait(timeout=1.0)
            outcome = self._results.pop(layer_idx)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self._tasks.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        slot_turn = 0
        while True:
            layer_idx = self._tasks.get()
            if layer_idx is None:
                return
            try:
                outcome = self._ship(layer_idx, slot_turn)
                slot_turn = (slot_turn + 1) % self._NUM_SLOTS
            except BaseException as exc:  # published, never swallowed
                outcome = exc
                self._broken = True
            with self._ready:
                self._pending.discard(layer_idx)
                self._results[layer_idx] = outcome
                self._ready.notify_all()
            if self._broken:
                return

    def _ship(self, layer_idx: int, slot_turn: int):
        slot = self._slots[slot_turn]
        previous = self._slot_events[slot_turn]
        if previous is not None:
            # the previous transfer through this slot must land before reuse
            previous.synchronize()
        tensors: Dict[str, torch.Tensor] = {}
        offset = 0
        with torch.inference_mode(False), torch.no_grad():
            staged = []
            for name, cpu_tensor in self._mapped_cpu_weights[layer_idx].items():
                width = cpu_tensor.element_size()
                if offset % width:
                    offset += width - (offset % width)
                start = offset // width
                window = slot.view(cpu_tensor.dtype)[
                    start : start + cpu_tensor.numel()
                ].view(cpu_tensor.shape)
                window.copy_(cpu_tensor)
                offset += cpu_tensor.numel() * width
                staged.append((name, window))
            event = torch.get_device_module().Event()
            with torch.get_device_module().stream(self._stream):
                for name, window in staged:
                    meta = self._weight_metadata[layer_idx][name]
                    gpu_tensor = torch.empty(
                        meta["shape"], dtype=meta["dtype"], device=self._device
                    )
                    gpu_tensor.copy_(window, non_blocking=True)
                    tensors[name] = gpu_tensor
                event.record(self._stream)
        self._slot_events[slot_turn] = event
        return event, tensors


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
        pin_budget: HostPinBudget | None = None,
        pin_component_name: str = "layerwise offload",
    ) -> None:
        self.model = model
        self.layers_attr_str = layers_attr_str
        self.num_layers = num_layers
        self._synchronous_mps = current_platform.is_mps()
        # mps shares physical memory with the CPU and has no pinned host memory
        # or CUDA-style copy streams
        self.pin_cpu_memory = bool(pin_cpu_memory and not self._synchronous_mps)
        # asked per layer rather than for the whole component; see
        # _plan_pinned_layers
        # A missing budget is not a licence to ignore host memory: without one
        # every layer looked affordable and the copies-do-not-fit check below
        # was never reached. A private budget reads the same host limit.
        self._pin_budget = pin_budget if pin_budget is not None else HostPinBudget()
        self._pin_component_name = pin_component_name
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
        # layer_idx -> {name: tensor still viewing the checkpoint file}
        # Weights left on their mapping rather than copied into host memory, so
        # the page cache decides what stays resident. Used when the copies would
        # not fit; see _plan_layer_hosting.
        self._mapped_cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._mapped_bytes = 0
        # mps keeps the original CPU tensor for each layer instead of building a
        # second flattened host copy
        self._mps_cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape}}
        # stores the offset and numel of each weight from a same layer, of same dtype
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()
        # mapped layers handed to the courier and not yet collected
        self._mapped_courier: Optional[MappedLayerCourier] = None
        self._courier_inflight: Set[int] = set()
        # layer_idx -> torch.get_device_module().Event for fine-grained sync, to make sure the weight is resident in pre-hook
        self._prefetch_events: Dict[int, torch.get_device_module().Event] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        self._offload_placeholders: Dict[torch.dtype, torch.Tensor] = {}
        self._has_dtensor_weights = False
        # A snapshot of this process's mappings, taken now because the weights
        # have just been loaded and their mappings exist.
        self._mapped_regions = MappedRegions()
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

    def _layer_byte_totals(
        self, layer_groups: Dict
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Per layer: (all weight bytes, the subset that are checkpoint views)."""
        totals: Dict[int, int] = {}
        mapped: Dict[int, int] = {}
        for layer_idx, dtype_to_params in layer_groups.items():
            total = 0
            from_mapping = 0
            for weights in dtype_to_params.values():
                for _, weight in weights:
                    tensor = self._to_local_tensor(weight)
                    nbytes = tensor.untyped_storage().nbytes()
                    total += nbytes
                    if self._mapped_regions.holds(tensor):
                        from_mapping += nbytes
            totals[layer_idx] = total
            mapped[layer_idx] = from_mapping
        return totals, mapped

    def _plan_layer_hosting(self, layer_groups: Dict) -> Dict[int, str]:
        """Where each layer's weights live on the host: pinned, pageable or mapped.

        Pinning is what lets the copy stream run ahead of compute; a pageable
        or mapped source transfers synchronously however it is requested.

        The budget used to be asked for the whole component at once, so a DiT
        larger than the whole spendable budget pinned nothing at all. Asking
        per layer spends what there is.

        A layer that misses the budget falls back the way it always did, to a
        pageable copy, and only stays on its mapping when those copies do not
        fit either. The order matters: a pageable copy transfers synchronously,
        since the driver stages it through its own pinned buffer, so it buys
        none of the overlap -- but it is guaranteed resident, where a mapping can
        be dropped and re-read from disk.

        Which layers get pinned matters only through how often each is read.
        A streamed layer is transferred once per denoise step; a resident one is
        transferred once per stage, so a pin on it is worth about 1/steps of the
        same pin on a streamed layer. Streamed layers therefore get the budget
        first, in streamed order, which is also deterministic. Unpinning a
        resident layer costs one possibly-faulting arming copy per request and
        buys a whole layer's worth of per-step overlap.
        """
        totals, mapped = self._layer_byte_totals(layer_groups)
        pinned_bytes = 0
        hosting: Dict[int, str] = {}
        pin_order: List[int] = []
        spendable = self._pin_budget.spendable_bytes if self._pin_budget else 0
        streamed = [idx for idx in self._streamed_order if idx in totals]
        resident = [idx for idx in sorted(totals) if idx not in set(streamed)]
        for layer_idx in streamed + resident:
            layer_bytes = totals[layer_idx]
            if self.pin_cpu_memory and pinned_bytes + layer_bytes <= spendable:
                hosting[layer_idx] = "pinned"
                pinned_bytes += layer_bytes
                pin_order.append(layer_idx)
            else:
                hosting[layer_idx] = "pageable"

        def anonymous_new_bytes() -> int:
            # What this plan adds, net, to anonymous memory. A store buffer
            # that replaces an anonymous original -- the non-view share, such
            # as a fused qkv -- is a wash: the original is freed when its
            # parameter is rebound. The net cost of hosting a layer off its
            # mapping is therefore only the checkpoint-view share it copies in,
            # and a layer left on the mapping adds nothing.
            return sum(
                mapped[idx] for idx, where in hosting.items() if where != "mapped"
            )

        unpinned = [idx for idx, where in hosting.items() if where != "pinned"]
        # The pins are booked but not yet allocated, so what the plan adds has
        # to be weighed as one sum against the live reading. Asking about any
        # one tier alone counts the same free bytes twice, and the error only
        # ever says "fits".
        if unpinned and host_copies_would_not_fit(anonymous_new_bytes()):
            for layer_idx in unpinned:
                if mapped[layer_idx]:
                    hosting[layer_idx] = "mapped"
            # If the pins alone still do not fit, pins are what there is to
            # give back. The tail of the pin order holds the least valuable
            # ones, so they go first.
            while pin_order and host_copies_would_not_fit(anonymous_new_bytes()):
                layer_idx = pin_order.pop()
                hosting[layer_idx] = "mapped" if mapped[layer_idx] else "pageable"
                pinned_bytes -= totals[layer_idx]
            if host_copies_would_not_fit(anonymous_new_bytes()):
                logger.warning(
                    "Layerwise offload: %s adds %.2f GiB of anonymous host "
                    "memory that no mapping can absorb, and %.2f GiB is "
                    "available. Expect the host to be the limit.",
                    self._pin_component_name,
                    anonymous_new_bytes() / 1024**3,
                    host_memory_available_bytes() / 1024**3,
                )
        if pinned_bytes and self._pin_budget is not None:
            self._pin_budget.request(
                component_name=self._pin_component_name, weight_bytes=pinned_bytes
            )

        if unpinned:
            counts = {where: 0 for where in ("pinned", "pageable", "mapped")}
            for where in hosting.values():
                counts[where] += 1
            logger.info(
                "Layerwise offload: %s pins %d of %d layers (%.2f GiB of %.2f GiB "
                "spendable). Of the rest, %d are copied into pageable host memory "
                "and %d stay on the checkpoint mapping. Pinning every layer would "
                "need %.2f GiB.",
                self._pin_component_name,
                counts["pinned"],
                len(totals),
                pinned_bytes / 1024**3,
                spendable / 1024**3,
                counts["pageable"],
                counts["mapped"],
                sum(totals.values()) / 1024**3,
            )
        return hosting

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

        layer_hosting = self._plan_layer_hosting(layer_groups)

        # 2. concat and offload (in pinned memory)
        for layer_idx, dtype_to_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._strided_cpu_weights[layer_idx] = {}
            self._mapped_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}

            hosting = layer_hosting.get(layer_idx, "pinned")
            pin_this_layer = hosting == "pinned"

            for dtype, weights in dtype_to_params.items():
                contiguous_weights: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
                for name, weight in weights:
                    local_weight = self._to_local_tensor(weight)
                    if hosting == "mapped" and self._mapped_regions.holds(local_weight):
                        # Already a view into the checkpoint. Copying it would
                        # add a second copy of bytes the page cache holds
                        # anyway, and that copy is what does not fit.
                        # `_to_local_tensor` hands back the parameter itself
                        # for anything that is not a DTensor, so storing it
                        # directly stores the parameter -- and `weight.data`
                        # below swaps that same object's storage for a (1,)
                        # placeholder, leaving the placeholder in the store.
                        # Keep an independent tensor over the mapped storage.
                        self._mapped_cpu_weights[layer_idx][
                            name
                        ] = local_weight.detach().view_as(local_weight)
                        self._weight_metadata[layer_idx][name] = {
                            "dtype": local_weight.dtype,
                            "shape": tuple(local_weight.shape),
                            "stride": local_weight.stride(),
                            "preserve_strides": False,
                            "mapped": True,
                        }
                        self._mapped_bytes += local_weight.untyped_storage().nbytes()
                        weight.data = self._get_shared_empty_tensor_for_target(
                            weight, local_weight.dtype
                        )
                        continue
                    if local_weight.is_contiguous():
                        contiguous_weights.append((name, weight, local_weight))
                        continue

                    # Preserve non-contiguous layouts such as the transposed FP8
                    # weight views expected by CUTLASS kernels.
                    cpu_tensor = torch.empty_strided(
                        size=local_weight.shape,
                        stride=local_weight.stride(),
                        dtype=dtype,
                        pin_memory=pin_this_layer,
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
                    total_numel, dtype=dtype, pin_memory=pin_this_layer
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
        if layer_idx in self._courier_inflight:
            if non_blocking:
                return
            self._collect_mapped_layer(layer_idx)
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
        if layer_idx not in self._consolidated_cpu_weights and not (
            self._mapped_cpu_weights.get(layer_idx)
        ):
            return
        if self.copy_stream is not None:
            self.copy_stream.wait_stream(torch.get_device_module().current_stream())
            stream_context = torch.get_device_module().stream(self.copy_stream)
        else:
            # the device has no CUDA-like stream or pinned-memory support
            non_blocking = False
            stream_context = nullcontext()

        # A mapped source is synchronous on this thread however the copy is
        # requested, so hand those weights to the courier and let it overlap
        # this layer's transfer with the previous layer's compute. Blocking
        # callers keep the direct path: they need the weights now.
        ship_mapped = False
        if non_blocking and self._mapped_cpu_weights.get(layer_idx):
            courier = self._ensure_mapped_courier()
            if courier is not None and courier.submit(layer_idx):
                self._courier_inflight.add(layer_idx)
                ship_mapped = True

        # create gpu buffer and load from CPU buffer
        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        with (
            torch.inference_mode(False),
            torch.no_grad(),
            stream_context,
        ):
            for dtype, cpu_buffer in self._consolidated_cpu_weights.get(
                layer_idx, {}
            ).items():
                gpu_buffer = torch.empty(
                    cpu_buffer.shape, dtype=dtype, device=self.device
                )
                gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                gpu_buffers[dtype] = gpu_buffer

            # restore model's weights by their metadata using the same copy stream
            # so the recorded event covers both flat-buffer and stride-preserving copies.
            for name, meta in self._weight_metadata[layer_idx].items():
                target = self.get_target_with_name(name)
                if meta.get("mapped", False):
                    if ship_mapped:
                        # the courier stages and ships these; bound at collect
                        continue
                    # Straight from the mapping. Not pinned, so this copy runs
                    # on the compute thread rather than ahead of it, and a page
                    # the kernel has reclaimed is faulted back in here.
                    cpu_tensor = self._mapped_cpu_weights[layer_idx][name]
                    gpu_tensor = torch.empty(
                        meta["shape"], dtype=meta["dtype"], device=self.device
                    )
                    gpu_tensor.copy_(cpu_tensor, non_blocking=False)
                    target.data = self._wrap_for_target(target, gpu_tensor)
                    continue

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

        if not ship_mapped:
            self._gpu_layers.add(layer_idx)

    def _ensure_mapped_courier(self) -> Optional[MappedLayerCourier]:
        """The courier, built on first use; None where it cannot help."""
        if self._mapped_courier is not None:
            return self._mapped_courier
        if envs.SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER:
            return None
        if self.copy_stream is None or self._synchronous_mps:
            return None
        if not self._mapped_bytes:
            return None
        try:
            self._mapped_courier = MappedLayerCourier(
                mapped_cpu_weights=self._mapped_cpu_weights,
                weight_metadata=self._weight_metadata,
                device=self.device,
                pin_slots=current_platform.is_cuda(),
            )
            logger.info(
                "Layerwise offload: %s ships mapped layers through a courier "
                "thread with %d pinned slots, so their device copies overlap "
                "compute instead of stalling it.",
                self.layers_attr_str,
                MappedLayerCourier._NUM_SLOTS,
            )
        except (RuntimeError, MemoryError, ValueError) as exc:
            logger.info(
                "Layerwise offload: no courier for mapped layers (%s); they "
                "keep the synchronous copy.",
                exc,
            )
            self._mapped_courier = None
            self._mapped_bytes = self._mapped_bytes  # unchanged; direct path
        return self._mapped_courier

    def _collect_mapped_layer(self, layer_idx: int) -> None:
        """Bind a shipped layer's tensors on the compute thread."""
        courier = self._mapped_courier
        try:
            event, tensors = courier.collect(layer_idx)
        except BaseException as exc:
            # The courier is out of service: fall back to the direct
            # synchronous path for this and every later layer.
            logger.warning(
                "Layerwise offload: courier failed for layer %d (%s); mapped "
                "layers return to the synchronous copy.",
                layer_idx,
                exc,
            )
            self._mapped_courier = None
            self._courier_inflight.discard(layer_idx)
            self.prefetch_layer(layer_idx, non_blocking=False)
            return
        with torch.inference_mode(False), torch.no_grad():
            for name, gpu_tensor in tensors.items():
                target = self.get_target_with_name(name)
                target.data = self._wrap_for_target(target, gpu_tensor)
        torch.get_device_module().current_stream().wait_event(event)
        self._courier_inflight.discard(layer_idx)
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

        # A layer still in flight holds device tensors inside the courier;
        # collecting binds and accounts for them so the release below sees them.
        for layer_idx in list(self._courier_inflight):
            self._collect_mapped_layer(layer_idx)

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
            if meta.get("mapped", False):
                # The store is the checkpoint file. Inference does not mutate
                # weights, so there is nothing to write back; writing would
                # copy-on-write the mapping into the anonymous memory this
                # exists to avoid.
                continue
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
                if meta.get("mapped", False):
                    yield name, self._mapped_cpu_weights[layer_idx][name]
                    continue

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

    # Whether to park non-layer parameters on the host between uses. Costs a
    # transfer per request and is worth it only when device memory is the
    # binding constraint, so it follows --performance-mode memory.
    park_non_layer_weights_between_uses: bool = False

    def _managed_layer_parameter_names(self) -> set:
        """Parameter names some layerwise manager already streams."""
        return {
            name
            for manager in self.layerwise_offload_managers
            for names in manager._weight_metadata.values()
            for name in names
        }

    def park_non_layer_weights(self) -> None:
        """Move the parameters no manager streams back to the host.

        A layerwise component holds its non-layer parameters on the device for
        the whole request. That is right while it is the component being used
        and pure cost afterwards. Measured on H3 at 864x480 / 124 frames: the
        DiT keeps 2.09 GB and the text encoder 1.40 GB through a VAE decode
        that touches neither, and the decode is exactly where the budget runs
        out -- with the VAE's blocks held resident it needs 11.86 GiB against a
        12 GiB card, and fails for want of 20 MiB.

        Buffers are left where they are. Layerwise offload keeps them resident
        on purpose, because a shared buffer such as a RoPE cache is referenced
        by many layers.
        """
        if not self.park_non_layer_weights_between_uses:
            return
        if current_platform.is_mps():
            # MPS parks its own non-layer weights, scoped to subphases
            return
        managed = self._managed_layer_parameter_names()
        resident = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name not in managed and parameter.device.type != "cpu"
        ]
        holds = sum(p.numel() * p.element_size() for _, p in resident)
        if holds <= self._device_headroom_bytes() * PARK_SIGNIFICANCE:
            # There is room. Give back any host copies rather than hold them.
            self._parked_non_layer_weights.clear()
            return

        parked = self._parked_non_layer_weights
        with torch.inference_mode(False), torch.no_grad():
            for name, parameter in resident:
                if name not in parked:
                    parked[name] = parameter.detach().to("cpu", copy=True)
                parameter.data = self._park_placeholder(parameter)

    def _device_headroom_bytes(self) -> int:
        """What an allocation could get without the allocator growing its pool.

        `get_available_gpu_memory` reports driver-level free memory, which
        excludes blocks the caching allocator has already reserved and not
        handed out. On a warm process that undercounts the real headroom badly,
        so the allocator's own unused reserve is added back.
        """
        free = int(
            current_platform.get_available_gpu_memory(empty_cache=False) * (1 << 30)
        )
        device_module = torch.get_device_module()
        unused_reserve = (
            device_module.memory_reserved() - device_module.memory_allocated()
        )
        return free + max(0, unused_reserve)

    def _park_placeholder(self, parameter: torch.Tensor) -> torch.Tensor:
        """One shared stand-in per (device, dtype), not one per parked weight."""
        key = (parameter.device, parameter.dtype)
        placeholder = self._park_placeholders.get(key)
        if placeholder is None:
            placeholder = torch.empty(
                (1,), dtype=parameter.dtype, device=parameter.device
            )
            self._park_placeholders[key] = placeholder
        return placeholder

    def restore_non_layer_weights(self) -> None:
        """Bring parked parameters back before this component is used again."""
        parked = self._parked_non_layer_weights
        if not parked:
            return
        device = current_platform.get_local_torch_device()
        parameters = dict(self.named_parameters())
        with torch.inference_mode(False), torch.no_grad():
            for name, host_tensor in parked.items():
                parameter = parameters.get(name)
                if parameter is None:
                    continue
                # The parked copy is pageable, so this transfer stages through
                # the driver's own pinned buffer and is synchronous whatever is
                # asked for. Pinning it instead would make the copy async, at
                # the price of host memory the kernel can never reclaim -- the
                # wrong trade on the hosts this path exists for.
                parameter.data = host_tensor.to(device)

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

    @property
    def _parked_non_layer_weights(self) -> dict:
        store = self.__dict__.get("_parked_non_layer_weight_store")
        if store is None:
            store = {}
            self.__dict__["_parked_non_layer_weight_store"] = store
        return store

    @property
    def _park_placeholders(self) -> dict:
        store = self.__dict__.get("_park_placeholder_store")
        if store is None:
            store = {}
            self.__dict__["_park_placeholder_store"] = store
        return store

    def configure_layerwise_offload(
        self,
        server_args: ServerArgs,
        *,
        pin_budget: HostPinBudget | None = None,
        component_name: str | None = None,
    ):
        self.park_non_layer_weights_between_uses = (
            server_args.performance_mode == "memory"
        )
        self.layerwise_offload_managers = []
        named_modules = dict(self.named_modules())
        layer_specs = []
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

            layer_specs.append((layer_name, num_layers, prefetch_size, resident_layers))

        if not layer_specs:
            logger.debug(
                "No layerwise-offloadable ModuleList found for %s. Candidates: %s",
                self.__class__.__name__,
                self.layer_names,
            )
            return

        component_label = (
            f"{component_name} ({self.__class__.__name__})"
            if component_name is not None
            else self.__class__.__name__
        )
        logger.info(
            "Configuring layerwise offload for %s: %s",
            component_label,
            ", ".join(
                f"{layer_name} ({num_layers} layers)"
                for layer_name, num_layers, _, _ in layer_specs
            ),
        )
        started_at = perf_counter()

        for layer_name, num_layers, prefetch_size, resident_layers in layer_specs:
            # Pinning these weights is what lets the copy stream run ahead of
            # compute, but pinned pages are the ones the kernel cannot reclaim,
            # so they are handed out only while the budget lasts. The budget goes
            # to the manager rather than being spent here, because it is asked
            # per layer: a component too large to pin whole can still pin part of
            # itself. See _plan_layer_hosting.
            pin_component_name = f"{component_name or type(self).__name__}.{layer_name}"

            manager = LayerwiseOffloadManager(
                model=self,
                layers_attr_str=layer_name,
                num_layers=num_layers,
                enabled=True,
                pin_cpu_memory=server_args.pin_cpu_memory,
                pin_budget=pin_budget,
                pin_component_name=pin_component_name,
                prefetch_size=prefetch_size,
                resident_layers=resident_layers,
                initialize=False,
                residency_policy=residency_policy,
            )
            self.layerwise_offload_managers.append(manager)

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

        managers = self.layerwise_offload_managers
        prefetch_sizes = ", ".join(
            str(value)
            for value in sorted({manager.prefetch_size for manager in managers})
        )
        policies = ", ".join(sorted({manager.residency_policy for manager in managers}))
        total_layers = sum(manager.num_layers for manager in managers)
        resident_layers = sum(manager.resident_layers for manager in managers)
        logger.info(
            "Layerwise offload ready for %s in %.2fs: groups=%d, layers=%d, "
            "prefetch/group=%s, resident=%d/%d, policy=%s",
            component_label,
            perf_counter() - started_at,
            len(managers),
            total_layers,
            prefetch_sizes,
            resident_layers,
            total_layers,
            policies,
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
        from sglang.multimodal_gen.registry import (
            get_model_info,
            get_pipeline_config_classes,
        )

        sampling_cls = None
        pipeline_class_name = server_args.pipeline_class_name
        if pipeline_class_name:
            config_classes = get_pipeline_config_classes(pipeline_class_name)
            if config_classes is not None:
                sampling_cls = config_classes[1]
        else:
            # The override is normally unset. Resolve the pipeline the way
            # build_pipeline does -- a cache hit by now -- because falling
            # back to 1 here turns the benefit ranking into a bare-bytes
            # ranking, and a once-per-request encoder can then outrank the
            # stepped DiT for the pin budget.
            model_path = getattr(server_args, "model_path", None)
            if model_path:
                model_info = get_model_info(
                    model_path,
                    backend=getattr(server_args, "backend", None),
                    model_id=getattr(server_args, "model_id", None),
                )
                if model_info is not None:
                    sampling_cls = model_info.sampling_param_cls
        if sampling_cls is None:
            return 1
        steps = getattr(sampling_cls(), "num_inference_steps", None)
        if not steps:
            return 1
        return max(1, int(steps))

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
    logger.info("Layerwise offload host memory: %s", describe_host_memory())

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
            "Layerwise offload summary: %s",
            ", ".join(
                f"{name} ({format_component_residency(modules[name])})"
                for name in configured_component_names
            ),
        )
    elif warn_missing:
        logger.debug("No selected pipeline component enabled layerwise offload")
    return configured_component_names
