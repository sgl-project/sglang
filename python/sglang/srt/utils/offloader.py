import logging
import os
from abc import ABC
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from torch.func import functional_call

from sglang.srt.distributed.naive_distributed import (
    NaiveDistributed,
    get_naive_distributed,
    set_naive_distributed,
)
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, is_pin_memory_available
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.host_shared_memory import (
    HostSharedMemoryManager,
    get_host_shared_memory_manager,
    set_host_shared_memory_manager,
)
from sglang.srt.utils.offloader_v2_mem_pool import ParamInfo, StaticBufferPool

logger = logging.getLogger(__name__)

_SubmoduleAccessor = Callable[[torch.nn.Module], torch.nn.Module]
_WhitelistParamNamesCreator = Callable[[torch.nn.Module], List[str]]


class BaseOffloader(ABC):
    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return list(all_modules_generator)

    def post_init(self):
        pass

    @property
    def forbid_copy_engine_usage(self):
        return False


class NoopOffloader(BaseOffloader):
    pass


# For simplicity use singleton, but can surely support multi instance
_instance: Optional[BaseOffloader] = NoopOffloader()


def get_offloader():
    assert _instance is not None
    return _instance


def set_offloader(instance: BaseOffloader):
    global _instance
    _instance = instance


def create_offloader_from_server_args(server_args: ServerArgs, dp_rank: int):
    if server_args.cpu_offload_gb > 0:
        return OffloaderV1(
            cpu_offload_max_bytes=int(server_args.cpu_offload_gb * 1024**3)
        )
    if server_args.offload_group_size > 0:
        assert (
            server_args.cpu_offload_gb == 0
        ), "V2 offload does not support cpu_offload_gb yet"
        return OffloaderV2(
            group_size=server_args.offload_group_size,
            num_in_group=server_args.offload_num_in_group,
            prefetch_step=server_args.offload_prefetch_step,
            mode=server_args.offload_mode,
            dp_rank=dp_rank,
            dp_size=server_args.dp_size,
        )
    return NoopOffloader()


class OffloaderV1(BaseOffloader):
    def __init__(self, cpu_offload_max_bytes: int):
        self._cpu_offload_bytes = 0
        self._cpu_offload_max_bytes = cpu_offload_max_bytes

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return [self.maybe_offload_to_cpu(module) for module in all_modules_generator]

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
            return module

        pin_memory = is_pin_memory_available()
        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for p in module.parameters():
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            p.data = cpu_data
            self._cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device, it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }
                output = functional_call(module, device_state, args=args, kwargs=kwargs)
                module.forward = forward
                return output

            module.forward = forward

        return module


class OffloaderV2(BaseOffloader):
    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        mode: str,
        dp_rank: int,
        dp_size: int,
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.mode = mode

        run_id = os.environ["SGLANG_RUN_ID"]

        # Temporarily init inside Offloader, can move if other modules also need this
        if self.mode in {"sharded_gpu", "shm_cpu"}:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size

            assert (
                get_tensor_model_parallel_world_size() == 1
            ), "not yet support tp_size!=1"
            set_naive_distributed(
                NaiveDistributed(
                    rank=dp_rank,
                    world_size=dp_size,
                    rendezvous=f"/tmp/{run_id}",
                )
            )
        if self.mode in {"shm_cpu"}:
            set_host_shared_memory_manager(
                HostSharedMemoryManager(
                    base_name=run_id,
                )
            )

        self.offloaders: List[_ModuleOffloader] = []
        self._buffer_pool: Optional[StaticBufferPool] = None
        self._alt_stream: Optional[torch.cuda.Stream] = None

    @property
    def forbid_copy_engine_usage(self):
        # Default path monopolizes the copy engine for prefetch H2D.
        return True

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        assert not self.offloaders, "wrap_modules should only be called once"
        assert (
            submodule_accessor is not None
        ), "submodule_accessor is required, this is likely due to missing offloader_kwargs for the model"
        assert (
            whitelist_param_names_creator is not None
        ), "whitelist_param_names_creator is required, this is likely due to missing offloader_kwargs for the model"

        self._alt_stream = torch.cuda.Stream()

        # Assign offloader a slot, then immediately offloads layer.
        all_modules: List[torch.nn.Module] = []
        offload_submodules: List[torch.nn.Module] = []
        for module_index, module in enumerate(all_modules_generator):
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                submodule = submodule_accessor(module)
                whitelist_param_names = whitelist_param_names_creator(submodule)
                logger.info(
                    f"[offloader] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names} memory_allocated={torch.cuda.memory_allocated()}"
                )
                offload_submodules.append(submodule)
                self.offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        alt_stream=self._alt_stream,
                        whitelist_param_names=whitelist_param_names,
                    )
                )

        # Slot assignment is `idx % prefetch_step` and the prefetch hook
        # schedules `(idx + prefetch_step) % N` as the next target. When N is
        # not a multiple of `prefetch_step`, the wrap-around picks a layer
        # whose slot collides with one still being read by a later layer in
        # the same forward, silently corrupting that layer's weights.
        assert not self.offloaders or len(self.offloaders) % self.prefetch_step == 0, (
            f"OffloaderV2 requires the number of layers offloaded "
            f"({len(self.offloaders)}) to be divisible by prefetch_step "
            f"({self.prefetch_step}); otherwise the slot collision may corrupt "
            f"proceeding layer's weight. Adjust your offload configuration accordingly."
        )

        for index, submodule in enumerate(offload_submodules):
            _hook_module_forward(index=index, module=submodule, offloader=self)

        return all_modules

    def post_init(self):
        if not self.offloaders:
            return

        for offloader in self.offloaders:
            offloader.prepare()

        # process_weights_after_loading may have some weights re-pinned / re-quantized;
        # refresh master copies before computing pool keys.
        for offloader in self.offloaders:
            offloader.load_cpu_data()

        param_infos: List[ParamInfo] = []
        device: Optional[torch.device] = None
        for offloader in self.offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device
        assert device is not None

        # Create a shared buffer pool and assign each offloader a slot.
        self._buffer_pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )

        # One slot per prefetch step, so prefetches can safely be started and run forward.
        for idx, offloader in enumerate(self.offloaders):
            offloader.assign_buffer_slot(self._buffer_pool, idx % self.prefetch_step)

        # Kick the first prefetch_step prefetches so layer 0..prefetch_step-1
        # don't stall on H2D for the first forward.
        for i in range(min(self.prefetch_step, len(self.offloaders))):
            self.offloaders[i].start_onload()

    def wait_for_layer(self, layer_idx: int) -> None:
        if not self.offloaders:
            return
        offloader = self.offloaders[layer_idx]
        if torch.cuda.is_current_stream_capturing():
            # When it is capturing, only wait on prefetches recorded inside this same capture.
            if not offloader.prefetch_in_capture:
                return
            torch.cuda.current_stream().wait_event(offloader.copy_done_event)
            offloader.prefetch_in_capture = False
        else:
            if offloader.event_valid_for_eager:
                torch.cuda.current_stream().wait_event(offloader.copy_done_event)
            else:
                # Event recorded inside a previous capture; join alt_stream.
                torch.cuda.current_stream().wait_stream(offloader.alt_stream)

    def wait_for_prev_onload(self) -> None:
        """Wait for all previous in-flight load."""
        if self._alt_stream is None:
            return
        torch.cuda.current_stream().wait_stream(self._alt_stream)

    def start_prefetch(self, layer_idx: int) -> None:
        if not self.offloaders:
            return
        self.offloaders[layer_idx].start_onload()

    def join_after_forward(self) -> None:
        """Join alt_stream after model forward completes.

        Joins the alt_stream before CUDA graph capture ends.
        This ensures all prefetches started during the forward are complete.

        Support both full-graph and piecewise mode.
        """
        if not self.offloaders:
            return
        for offloader in self.offloaders:
            if offloader.prefetch_in_capture:
                torch.cuda.current_stream().wait_event(offloader.copy_done_event)
                offloader.prefetch_in_capture = False


@register_custom_op(
    op_name="offloader_v2_wait_prefetch",
    mutates_args=["input_tensor"],
    fake_impl=lambda input_tensor, layer_idx: None,
)
def offloader_v2_wait_prefetch(input_tensor: torch.Tensor, layer_idx: int) -> None:
    offloader = get_offloader()
    if isinstance(offloader, OffloaderV2):
        offloader.wait_for_layer(layer_idx)


@register_custom_op(
    op_name="offloader_v2_start_prefetch",
    mutates_args=["output_tensor"],
    fake_impl=lambda output_tensor, layer_idx: None,
)
def offloader_v2_start_prefetch(output_tensor: torch.Tensor, layer_idx: int) -> None:
    offloader = get_offloader()
    if isinstance(offloader, OffloaderV2):
        offloader.start_prefetch(layer_idx)


def _first_tensor_arg(args, kwargs) -> Optional[torch.Tensor]:
    for v in args:
        if isinstance(v, torch.Tensor):
            return v
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _first_tensor_out(output) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


# ---------------------------------------------------------------------------
# Forward hook
# ---------------------------------------------------------------------------


def _hook_module_forward(
    index: int,
    module: torch.nn.Module,
    offloader: OffloaderV2,
) -> None:
    """Gate prefetch sync points through opaque custom ops.

    ``mutates_args`` on the input/output tensors keeps dynamo from reordering
    the wait/start across the layer's compute, while the ops themselves hide
    the underlying ``torch.cuda.Stream`` / ``Event`` from dynamo tracing.
    """
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        try:
            in_t = _first_tensor_arg(args, kwargs)
            if in_t is not None:
                torch.ops.sglang.offloader_v2_wait_prefetch(in_t, index)
            output = original_forward(*args, **kwargs)
            num = len(offloader.offloaders)
            next_idx = (index + offloader.prefetch_step) % num
            out_t = _first_tensor_out(output)
            if out_t is not None:
                torch.ops.sglang.offloader_v2_start_prefetch(out_t, next_idx)
            return output
        finally:
            module.forward = forward

    module.forward = forward


class _ModuleOffloader:
    def __init__(
        self,
        mode: str,
        module: torch.nn.Module,
        alt_stream: torch.cuda.Stream,
        whitelist_param_names: List[str],
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream

        assert self.device != torch.device(
            "cpu"
        ), "module parameters must start on the device; offloader handles CPU placement"

        # Event when copy is complete.
        self.copy_done_event = torch.cuda.Event()
        # Event when forward compute is complete.
        self.comp_done_event = torch.cuda.Event()
        # True when the most recent record() happened outside a graph capture
        # (events recorded during capture are unusable in eager wait).
        self.event_valid_for_eager = False
        # True when the most recent prefetch was started inside a capture.
        self.prefetch_in_capture = False

        param_dict = dict(self.module.named_parameters())
        assert all(
            name in param_dict for name in whitelist_param_names
        ), f"{whitelist_param_names=} {list(param_dict.keys())=}"

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for offloader in self._param_offloaders.values():
            offloader.post_init()

    def start_onload(self):
        if torch.cuda.is_current_stream_capturing():
            self._device_tensors = self._create_device_tensors()
            self._load_event = None
            return
        self.alt_stream.wait_stream(torch.cuda.current_stream())

    def prepare(self) -> None:
        for offloader in self._param_offloaders.values():
            offloader.prepare()

    def load_cpu_data(self) -> None:
        for offloader in self._param_offloaders.values():
            offloader.load_cpu_data()
        # Drop any params that disappeared during process_weights_after_loading.
        deleted = [n for n, o in self._param_offloaders.items() if o._param_deleted]
        for name in deleted:
            del self._param_offloaders[name]

    def get_param_infos(self) -> List[ParamInfo]:
        return [
            ParamInfo(
                name=name,
                shape=tuple(offloader.master_shape),
                stride=tuple(offloader.master_stride),
                dtype=offloader.master_dtype,
            )
            for name, offloader in self._param_offloaders.items()
        ]

    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int) -> None:
        for name, offloader in self._param_offloaders.items():
            buffer = pool.get(
                name=name,
                shape=tuple(offloader.master_shape),
                stride=tuple(offloader.master_stride),
                dtype=offloader.master_dtype,
                slot_idx=slot_idx,
            )
            offloader.assign_device_buffer(buffer)

    # def start_onload(self) -> None:
    #     """Start parameter fetch task.

    #     IMPORTANT: Onload stream must wait for current forward compute to complete.
    #     And ensure the streams are fully synced. This allows CUDA Graph capturing.
    #     """
    #     self.prefetch_in_capture = torch.cuda.is_current_stream_capturing()

    #     # alt_stream waits for forward compute to complete.
    #     # This orchestration allows alt_stream copy to be captured.
    #     torch.cuda.current_stream().record_event(self.comp_done_event)
    #     self.alt_stream.wait_event(self.comp_done_event)

    #     # Trigger parameter fetch.
    #     with torch.cuda.stream(self.alt_stream):
    #         for offloader in self._param_offloaders.values():
    #             offloader.load_device_tensor()

    def offload(self):
        self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self):
        assert self._device_tensors is not None
        if torch.cuda.is_current_stream_capturing():
            if self._load_event is not None:
                self._device_tensors = self._create_device_tensors()
                self._load_event = None
            return self._device_tensors
        if self._load_event is not None:
            self._load_event.wait()
        return self._device_tensors

    def _create_device_tensors(self):
        return {k: v.create_device_tensor() for k, v in self._param_offloaders.items()}
        # # Record copy completion event.
        # self.copy_done_event.record(self.alt_stream)
        # # Event is only valid for eager wait_event if recorded outside capture.
        # # Events recorded during capture become invalid after capture ends.
        # self.event_valid_for_eager = not self.prefetch_in_capture


class _BaseParamOffloader(ABC):
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        return {
            "meta": _MetaParamOffloader,
            "cpu": _CpuParamOffloader,
            "shm_cpu": _ShmCpuParamOffloader,
            "sharded_gpu": _ShardedGpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, module: torch.nn.Module, param_name: str):
        self._module = module
        self._param_name = param_name
        self._device_tensor: Optional[torch.Tensor] = None
        self._param_deleted = False

    @property
    def _param(self) -> torch.nn.Parameter:
        parent, leaf = _resolve_param(self._module, self._param_name)
        return getattr(parent, leaf)

    # -- protocol --

    @property
    def master_shape(self) -> torch.Size:
        raise NotImplementedError

    @property
    def master_stride(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def master_dtype(self) -> torch.dtype:
        raise NotImplementedError

    def post_init(self):
        pass

    def prepare(self) -> None:
        """One-time setup before pool allocation. Default: no-op."""
        pass

    def load_cpu_data(self) -> None:
        """Refresh master copy from current ``param.data``. Default: no-op."""
        pass

    def assign_device_buffer(self, device_buffer: torch.Tensor) -> None:
        """Bind ``param.data`` to the device buffer and seed it."""
        self._device_tensor = device_buffer
        parent, leaf = _resolve_param(self._module, self._param_name)
        old_param = getattr(parent, leaf)
        if old_param.device == device_buffer.device:
            old_param.data = device_buffer
        else:
            setattr(parent, leaf, _build_param(old_param, device_buffer))
        # Seed the buffer once so the very first forward reads valid data.
        self.load_device_tensor()

    def load_device_tensor(self) -> None:
        """Async refill the device buffer from the master copy.

        Called on the prefetch copy stream.
        """
        raise NotImplementedError


class _MetaParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        p = self._param
        self._shape = p.shape
        self._stride = p.stride()
        self._dtype = p.dtype
        _move_param_to_meta(module, param_name)

    @property
    def master_shape(self):
        return self._shape

    @property
    def master_stride(self):
        return self._stride

    @property
    def master_dtype(self):
        return self._dtype

    def load_device_tensor(self) -> None:
        # No real master; the buffer keeps whatever uninitialized data
        # ``torch.empty_strided`` produced. Meta mode is debug only.
        pass


class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        p = self._param
        self._cpu_data: torch.Tensor = _create_cpu_data(
            p.data, pin_memory=is_pin_memory_available()
        )
        # Replace the device storage immediately to free its memory.
        p.data = self._cpu_data

    @property
    def master_shape(self):
        return self._cpu_data.shape

    @property
    def master_stride(self):
        return self._cpu_data.stride()

    @property
    def master_dtype(self):
        return self._cpu_data.dtype

    def load_cpu_data(self) -> None:
        try:
            param = self._param
        except AttributeError:
            self._param_deleted = True
            return

        data = param.data
        if data.device.type == "cpu":
            if is_pin_memory_available() and not data.is_pinned():
                pinned = _create_cpu_data(data, pin_memory=True)
                self._cpu_data = pinned
                param.data = pinned
            else:
                self._cpu_data = data
        else:
            # Quantization may have placed param.data back on device.
            if (
                self._cpu_data.shape != data.shape
                or self._cpu_data.dtype != data.dtype
                or self._cpu_data.stride() != data.stride()
            ):
                self._cpu_data = _create_cpu_data(
                    data, pin_memory=is_pin_memory_available()
                )
            self._cpu_data.copy_(data)

    def load_device_tensor(self) -> None:
        assert self._device_tensor is not None
        self._device_tensor.copy_(self._cpu_data, non_blocking=True)


class _ShmCpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()

        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        self._shm_cpu_data: torch.Tensor = get_host_shared_memory_manager().malloc(
            shape=self._param.shape, dtype=self._param.dtype
        )

        if self._rank == 0:
            self._shm_cpu_data.copy_(self._param.data.to("cpu"))
        _move_param_to_meta(self._module, self._param_name)
        get_naive_distributed().barrier()

    @property
    def master_shape(self):
        return self._shm_cpu_data.shape

    @property
    def master_stride(self):
        return self._shm_cpu_data.stride()

    @property
    def master_dtype(self):
        return self._shm_cpu_data.dtype

    def load_device_tensor(self) -> None:
        assert self._device_tensor is not None
        self._device_tensor.copy_(self._shm_cpu_data, non_blocking=True)


def update_param(param, new_tensor):
    """Update a parameter."""
    if param.device == new_tensor.device:
        param.data = new_tensor
    else:
        assert param.device == torch.device(
            "cpu"
        ), f"{param.device=} {new_tensor.device=}"
        param.data = _create_cpu_data(new_tensor, pin_memory=True)


def _move_param_to_cpu(param, pin_memory: bool):
    param.data = _create_cpu_data(param.data, pin_memory=pin_memory)


def _create_cpu_data(data, pin_memory: bool):
    cpu_data = _empty_strided_like(data, device="cpu", pin_memory=pin_memory)
    cpu_data.copy_(data)
    return cpu_data


def _resolve_param(module: torch.nn.Module, dotted: str):
    """Walk a dotted name like 'mlp.gate_up_proj.weight' to (parent, leaf)."""
    parts = dotted.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _build_param(
    old_param: torch.nn.Parameter, new_data: torch.Tensor
) -> torch.nn.Parameter:
    """Construct a Parameter wrapping ``new_data`` matching ``old_param``'s
    class. Supports ``ModelWeightParameter`` and plain ``nn.Parameter``;
    other subclasses raise so we notice unsupported quant types.
    """
    old_param_type = type(old_param)
    if old_param_type == ModelWeightParameter:
        # manually checked how `w13_weight` and `w2_weight` are constructed
        new_param = ModelWeightParameter(
            data=new_data,
            **{
                k: getattr(old_param, k)
                for k in ["input_dim", "output_dim", "weight_loader"]
            },
        )
    elif old_param_type == torch.nn.Parameter:
        new_param = torch.nn.Parameter(
            data=new_data,
            requires_grad=False,
        )
        if hasattr(old_param, "weight_loader"):
            new_param.weight_loader = old_param.weight_loader
        else:
            new_param.weight_loader = lambda *args, **kwargs: None
    else:
        raise ValueError(f"Unknown {old_param_type=} {old_param=}")

    # Carry over any extra attributes.
    new_param.__dict__.update(old_param.__dict__)
    if not hasattr(new_param, "weight_loader"):
        new_param.weight_loader = lambda *args, **kwargs: None

    return new_param


def _move_param_to_meta(module, param_name):
    """Replace a Parameter with a meta placeholder of the same class."""
    parent, leaf = _resolve_param(module, param_name)
    old_param = getattr(parent, leaf)
    new_data = old_param.data.to("meta")
    setattr(parent, leaf, _build_param(old_param, new_data))


def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# sharded_gpu: master split across DP rank GPUs; refill via NVLink/IPC.
# ----------------------------------------- ShardedGpu ------------------------------------------------------


# TODO unify with ShmCpu mode
class _ShardedGpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()
        self._world_size = get_naive_distributed().get_world_size()

        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        # Capture full layout before we throw the local copy away.
        self._full_shape: torch.Size = self._param.shape
        self._full_stride: Tuple[int, ...] = self._param.stride()
        self._full_dtype: torch.dtype = self._param.dtype

        if self._rank == 0:
            _move_param_to_cpu(self._param, pin_memory=True)
        else:
            _move_param_to_meta(self._module, self._param_name)

        self.sharded_param_handles: Optional[List[torch.Tensor]] = None

    @property
    def master_shape(self):
        return self._full_shape

    @property
    def master_stride(self):
        return self._full_stride

    @property
    def master_dtype(self):
        return self._full_dtype

    def prepare(self) -> None:
        # rank 0 has the full pinned-CPU copy at this point; scatter it.
        scatter_src = None
        scatter_list: Optional[List[torch.Tensor]] = None
        if self._rank == 0:
            assert self._param.data.is_contiguous()
            scatter_src = self._param.data.to("cuda")
            scatter_list = _even_chunk(scatter_src, self._world_size)
        shard_shape = (self._full_shape[0] // self._world_size,) + tuple(
            self._full_shape[1:]
        )
        sharded_param = torch.empty(shard_shape, dtype=self._full_dtype, device="cuda")
        self.sharded_param_handles = _create_shared_buffer_tensors(
            local_tensor=sharded_param
        )
        get_naive_distributed().scatter(sharded_param, scatter_list)
        _move_param_to_meta(self._module, self._param_name)

    def load_device_tensor(self) -> None:
        assert (
            self._device_tensor is not None and self.sharded_param_handles is not None
        )
        output_chunks = self._device_tensor.chunk(self._world_size)
        for index in range(self._world_size):
            src_rank = (self._rank + index) % self._world_size
            output_chunks[src_rank].copy_(self.sharded_param_handles[src_rank])


def _even_chunk(x: torch.Tensor, chunks: int) -> List[torch.Tensor]:
    assert x.shape[0] % chunks == 0, f"{x.shape=} {chunks=}"
    return list(x.chunk(chunks))


def _create_shared_buffer_tensors(local_tensor: torch.Tensor) -> List[torch.Tensor]:
    self_rank = get_naive_distributed().get_rank()
    world_size = get_naive_distributed().get_world_size()

    object_list = get_naive_distributed().all_gather_object(
        dict(
            dup_serialized_local_tensor=[
                (
                    None
                    if interesting_rank == self_rank
                    else MultiprocessingSerializer.serialize(local_tensor)
                )
                for interesting_rank in range(world_size)
            ]
        )
    )

    output_tensors: List[torch.Tensor] = []
    for output_rank in range(world_size):
        remote_serialized_tensor = object_list[output_rank][
            "dup_serialized_local_tensor"
        ][self_rank]
        if output_rank == self_rank:
            assert remote_serialized_tensor is None
            output_tensors.append(local_tensor)
        else:
            output_tensors.append(
                MultiprocessingSerializer.deserialize(remote_serialized_tensor)
            )

    return output_tensors


# ALL_MODEL_PARAMS is a helper dict to offload all weights from the model.
# Users can set offloader_kwargs to ALL_MODEL_PARAMS to enable full-offload.
ALL_MODEL_PARAMS: Dict[str, Any] = dict(
    submodule_accessor=lambda layer: layer,
    whitelist_param_names_creator=lambda module: [
        name
        for name, _ in module.named_parameters(recurse=True)
        if name.endswith(".weight")
    ],
)
