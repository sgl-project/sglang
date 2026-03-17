import logging
import os
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Tuple

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
from sglang.srt.utils.common import direct_register_custom_op
from sglang.srt.utils.host_shared_memory import (
    HostSharedMemoryManager,
    get_host_shared_memory_manager,
    set_host_shared_memory_manager,
)

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

    def sync_prev_onload(self):
        pass

    def join_after_forward(self):
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
            offload_param_names=server_args.offload_param_names,
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


@dataclass
class ParamInfo:
    """Metadata about an offloaded parameter."""

    name: str
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype

    @property
    def key(self) -> Tuple[str, Tuple[int, ...], Tuple[int, ...], torch.dtype]:
        # Include `name` so parameters with same shape/stride/dtype but different
        # meaning do not share buffers.
        return (self.name, self.shape, self.stride, self.dtype)

    @property
    def num_bytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * torch.finfo(self.dtype).bits // 8


class StaticBufferPool:
    """Pre-allocated GPU buffer pool for offloaded parameters.

    Buffers are allocated per unique (name, shape, stride, dtype) and are
    reused across layers via a slot mechanism.
    """

    def __init__(
        self,
        param_infos: List[ParamInfo],
        slot_capacity: int,
        device: torch.device,
    ):
        self.slot_capacity = slot_capacity
        self.total_bytes = 0
        self._device = device

        # Allocate one set of buffers per unique param signature
        unique_params: Dict[
            Tuple[str, Tuple[int, ...], Tuple[int, ...], torch.dtype], ParamInfo
        ] = {}
        for info in param_infos:
            if info.key not in unique_params:
                unique_params[info.key] = info

        self._buffers: Dict[
            Tuple[str, Tuple[int, ...], Tuple[int, ...], torch.dtype],
            List[torch.Tensor],
        ] = {}
        for key, info in unique_params.items():
            slot_tensors: List[torch.Tensor] = []
            for _ in range(slot_capacity):
                buf = torch.empty_strided(
                    size=info.shape,
                    stride=info.stride,
                    dtype=info.dtype,
                    device=device,
                )
                slot_tensors.append(buf)
                self.total_bytes += info.num_bytes
            self._buffers[key] = slot_tensors

        logger.debug(
            "[StaticBufferPool] Allocated %d unique (name, shape, stride, dtype), %d slots each, total %.4f GB",
            len(unique_params),
            slot_capacity,
            self.total_bytes / 1e9,
        )

    def get_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        return self._buffers[(name, shape, stride, dtype)][
            slot_idx % self.slot_capacity
        ]


# Register custom ops for torch.compile / CUDA graph support


def _wait_prefetch_impl(input_tensor: torch.Tensor, layer_idx: int) -> None:
    # Only wait in eager mode, not during CUDA graph capture
    if not torch.cuda.is_current_stream_capturing():
        get_offloader()._wait_for_layer(layer_idx)


def _wait_prefetch_fake(input_tensor: torch.Tensor, layer_idx: int) -> None:
    # No-op during torch.compile tracing
    return


def _start_prefetch_impl(output_tensor: torch.Tensor, layer_idx: int) -> None:
    # Only prefetch in eager mode, not during CUDA graph capture
    if not torch.cuda.is_current_stream_capturing():
        get_offloader()._start_prefetch(layer_idx)


def _start_prefetch_fake(output_tensor: torch.Tensor, layer_idx: int) -> None:
    # No-op during torch.compile tracing
    return


def _register_prefetch_offloader_ops() -> None:
    direct_register_custom_op(
        op_name="wait_prefetch",
        op_func=_wait_prefetch_impl,
        mutates_args=["input_tensor"],
        fake_impl=_wait_prefetch_fake,
    )

    direct_register_custom_op(
        op_name="start_prefetch",
        op_func=_start_prefetch_impl,
        mutates_args=["output_tensor"],
        fake_impl=_start_prefetch_fake,
    )


_register_prefetch_offloader_ops()


class OffloaderV2(BaseOffloader):
    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        mode: str,
        dp_rank: int,
        dp_size: int,
        offload_param_names: Optional[List[str]] = None,
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.mode = mode
        self.offload_param_names = set(offload_param_names or [])

        self.copy_stream = torch.cuda.Stream()
        self.module_offloaders: List[_ModuleOffloader] = []
        self.buffer_pool: Optional[StaticBufferPool] = None
        self.total_offloaded_bytes = 0

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

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        assert len(self.module_offloaders) == 0, "should only call wrap_modules once"

        submodule_accessor = submodule_accessor or (lambda m: m)
        whitelist_param_names_creator = (
            whitelist_param_names_creator or self._default_whitelist_param_names
        )

        all_modules = []
        offload_submodules = []

        for module_index, module in enumerate(all_modules_generator):
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                submodule = submodule_accessor(module)
                whitelist_param_names = whitelist_param_names_creator(submodule)
                if not whitelist_param_names:
                    continue

                logger.info(
                    f"[offloader] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names} "
                    f"memory_allocated={torch.cuda.memory_allocated()}"
                )

                offload_submodules.append(submodule)
                self.module_offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        copy_stream=self.copy_stream,
                        whitelist_param_names=whitelist_param_names,
                        layer_idx=len(self.module_offloaders),
                    )
                )

        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index,
                module=module,
                prefetch_step=self.prefetch_step,
                num_offloaders=len(self.module_offloaders),
            )

        return all_modules

    def _default_whitelist_param_names(self, module: torch.nn.Module) -> List[str]:
        all_params = [name for name, _ in module.named_parameters()]
        if not self.offload_param_names:
            return all_params
        return [
            name
            for name in all_params
            if any(f".{p}." in f".{name}." for p in self.offload_param_names)
        ]

    def post_init(self):
        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()

        param_infos: List[ParamInfo] = []
        device: Optional[torch.device] = None
        for offloader in self.module_offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device

        if device is None:
            # No modules to offload
            return

        self.buffer_pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )

        for idx, offloader in enumerate(self.module_offloaders):
            offloader.assign_buffer_slot(self.buffer_pool, idx % self.prefetch_step)

        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        logger.info(
            f"[offloader] Initialized {len(self.module_offloaders)} modules. "
            f"Total GPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static buffer pool: {self.buffer_pool.total_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode})"
        )

        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            self.module_offloaders[i].start_onload_to_static()

    def _start_prefetch(self, layer_idx: int):
        self.module_offloaders[layer_idx].start_onload_to_static()

    def _wait_for_layer(self, layer_idx: int):
        offloader = self.module_offloaders[layer_idx]

        if offloader._event_valid_for_eager:
            torch.cuda.current_stream().wait_event(offloader._copy_done_event)
        else:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def sync_prev_onload(self):
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    @property
    def forbid_copy_engine_usage(self):
        return self.mode == "cpu"


def _hook_module_forward_for_offloader(
    index: int, module: torch.nn.Module, prefetch_step: int, num_offloaders: int
):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward

        # Call custom ops which handle capture/eager mode internally
        input_tensor = args[0] if args else kwargs.get("hidden_states")
        torch.ops.sglang.wait_prefetch(input_tensor, index)

        output = original_forward(*args, **kwargs)

        next_index = (index + prefetch_step) % num_offloaders
        output_tensor = output[0] if isinstance(output, tuple) else output
        torch.ops.sglang.start_prefetch(output_tensor, next_index)

        module.forward = forward
        return output

    module.forward = forward


class _ModuleOffloader(ABC):
    def __init__(
        self,
        mode: str,
        module: torch.nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: List[str],
        layer_idx: int,
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self.layer_idx = layer_idx
        self.offloaded_bytes = 0

        # Event to signal when H2D copy to static buffer is complete.
        self._copy_done_event = torch.cuda.Event()

        # Track whether `_copy_done_event` can be used with wait_event in eager mode.
        self._event_valid_for_eager = False

        # Track whether this layer's next prefetch was started during CUDA graph capture.
        self._prefetch_in_capture = False

        assert self.device != torch.device("cpu"), (
            "Module parameters should not already be on CPU "
            "(offloader handles CPU placement)"
        )

        self._buffer_pool: Optional[StaticBufferPool] = None
        self._buffer_slot_idx: int = 0

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"Whitelist params {whitelist_param_names} not found in module params "
            f"{list(param_dict.keys())}"
        )

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for param_offloader in self._param_offloaders.values():
            param_offloader.post_init()
            self.offloaded_bytes += getattr(param_offloader, "offloaded_bytes", 0)

    def sync_cpu_storage(self):
        for param_offloader in self._param_offloaders.values():
            param_offloader.sync_cpu_storage()

    def get_param_infos(self) -> List[ParamInfo]:
        infos: List[ParamInfo] = []
        for name, offloader in self._param_offloaders.items():
            cpu_storage = getattr(offloader, "_cpu_storage", None)
            assert cpu_storage is not None, "CPU storage not initialized"
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                )
            )
        return infos

    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int):
        self._buffer_pool = pool
        self._buffer_slot_idx = slot_idx

        for name, offloader in self._param_offloaders.items():
            cpu_storage = getattr(offloader, "_cpu_storage", None)
            assert cpu_storage is not None, "CPU storage not initialized"
            buffer = pool.get_buffer(
                name=name,
                shape=tuple(cpu_storage.shape),
                stride=tuple(cpu_storage.stride()),
                dtype=cpu_storage.dtype,
                slot_idx=slot_idx,
            )
            offloader.assign_static_buffer(buffer)

    def start_onload_to_static(self):
        assert self._buffer_pool is not None, "Buffer pool is not assigned"

        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()

        # Fork: record event on compute stream, have copy stream wait on it.
        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(self.copy_stream):
            for offloader in self._param_offloaders.values():
                cpu_storage = getattr(offloader, "_cpu_storage", None)
                gpu_buffer = getattr(offloader, "_gpu_buffer", None)
                assert cpu_storage is not None, "CPU storage not initialized"
                assert gpu_buffer is not None, "GPU buffer not assigned"
                assert not is_pin_memory_available() or cpu_storage.is_pinned(), (
                    f"CPU storage for {offloader._param_name} is not pinned! "
                    "non_blocking=True H2D copy from non-pinned memory "
                    "causes stream synchronization that breaks event-based fork synchronization."
                )
                gpu_buffer.copy_(cpu_storage, non_blocking=True)

        self._copy_done_event.record(self.copy_stream)
        self._event_valid_for_eager = not torch.cuda.is_current_stream_capturing()


class _BaseParamOffloader(ABC):
    """Base class for parameter offloading strategies."""

    _cpu_storage: Optional[torch.Tensor]
    _gpu_buffer: Optional[torch.Tensor]

    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        if mode == "cpu":
            return _CpuParamOffloader(**kwargs)
        elif mode == "meta":
            return _MetaParamOffloader(**kwargs)
        elif mode == "shm_cpu":
            return _ShmCpuParamOffloader(**kwargs)
        elif mode == "sharded_gpu":
            return _ShardedGpuParamOffloader(**kwargs)
        else:
            raise ValueError(f"Unknown offload mode: {mode}")

    def __init__(self, module, param_name):
        self._module = module
        self._param_name = param_name
        self.offloaded_bytes = 0
        self._cpu_storage = None
        self._gpu_buffer = None

    @property
    def _param(self):
        obj = self._module
        for part in self._param_name.split("."):
            obj = getattr(obj, part)
        return obj

    def post_init(self):
        return

    def sync_cpu_storage(self) -> None:
        raise NotImplementedError

    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        raise NotImplementedError


class _CpuParamOffloader(_BaseParamOffloader):
    """Offload parameter to pinned CPU memory and keep GPU static buffer."""

    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._offload_to_cpu_internal()

    def _offload_to_cpu_internal(self) -> None:
        param = self._param
        pin_memory = is_pin_memory_available()

        self._cpu_storage = torch.empty_strided(
            size=param.data.size(),
            stride=param.data.stride(),
            dtype=param.data.dtype,
            layout=param.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_storage.copy_(param.data)

        self.offloaded_bytes = (
            self._cpu_storage.numel() * self._cpu_storage.element_size()
        )

        # Point parameter to CPU storage to free GPU memory
        param.data = self._cpu_storage

    def _update_cpu_storage_from_param(self) -> None:
        param = self._param
        if param.data.device.type == "cpu":
            if is_pin_memory_available() and not param.data.is_pinned():
                pinned = torch.empty_strided(
                    size=param.data.size(),
                    stride=param.data.stride(),
                    dtype=param.data.dtype,
                    layout=param.data.layout,
                    device="cpu",
                    pin_memory=True,
                )
                pinned.copy_(param.data)
                self._cpu_storage = pinned
            else:
                self._cpu_storage = param.data
        else:
            assert self._cpu_storage is not None
            self._cpu_storage.copy_(param.data)

    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        assert (
            self._cpu_storage is not None
        ), "_offload_to_cpu_internal() must be called before assign_static_buffer()"

        # Ensure CPU storage has the latest weights
        self._update_cpu_storage_from_param()

        self._gpu_buffer = gpu_buffer
        param = self._param
        param.data = gpu_buffer

    def sync_cpu_storage(self) -> None:
        self._update_cpu_storage_from_param()


class _MetaParamOffloader(_BaseParamOffloader):
    """Usually used for debugging."""

    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_meta(module, param_name)

    def sync_cpu_storage(self) -> None:
        pass

    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        # Parameter is already on meta; nothing to do.
        pass


class _ShmCpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()
        self._world_size = get_naive_distributed().get_world_size()

        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        self.shm_cpu_data = get_host_shared_memory_manager().malloc(
            shape=self._param.shape, dtype=self._param.dtype
        )

        if self._rank == 0:
            self.shm_cpu_data.copy_(self._param.data.to("cpu"))
            self._param.data = self.shm_cpu_data
        else:
            _move_param_to_meta(self._module, self._param_name)
        get_naive_distributed().barrier()

    def post_init(self):
        if self._rank == 0:
            assert (
                self.shm_cpu_data.data_ptr() == self._param.data.data_ptr()
            ), f"{self.shm_cpu_data.data_ptr()=} {self._param.data.data_ptr()=} {self.shm_cpu_data=} {self._param.data=}"

        _move_param_to_meta(self._module, self._param_name)

    def sync_cpu_storage(self) -> None:
        # No-op; CPU storage is in shared memory.
        pass

    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        # Same as _CpuParamOffloader but use shared memory as source
        self._gpu_buffer = gpu_buffer
        self._param.data = gpu_buffer


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

        if self._rank == 0:
            _move_param_to_cpu(self._param, pin_memory=True)
        else:
            _move_param_to_meta(self._module, self._param_name)

        self.sharded_param_handles = None

    def post_init(self):
        # check again since it may be changed
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        scatter_src = self._param.data

        logger.info(
            f"[offloader] post_init {scatter_src.nbytes=} {scatter_src.dtype=} {scatter_src.shape=} {torch.cuda.memory_allocated()=}"
        )

        if self._rank == 0:
            scatter_src = scatter_src.to("cuda")
        scatter_list = _even_chunk(scatter_src, self._world_size)

        sharded_param = torch.empty(
            scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda"
        )
        self.sharded_param_handles = _create_shared_buffer_tensors(
            local_tensor=sharded_param
        )

        get_naive_distributed().scatter(
            sharded_param, scatter_list if self._rank == 0 else None
        )

        _move_param_to_meta(self._module, self._param_name)

    def create_device_tensor(self):
        output = _empty_strided_like(self._param, device="cuda")
        output_chunks = output.chunk(self._world_size)

        for index in range(self._world_size):
            src_rank = (self._rank + index) % self._world_size
            src_buf = self.sharded_param_handles[src_rank]
            output_chunks[src_rank].copy_(src_buf)

        return output


def _get_param_by_name(module: torch.nn.Module, name: str):
    """Support nested parameter names like 'self_attn.qkv_proj.weight'."""
    obj = module
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _set_param_by_name(
    module: torch.nn.Module, name: str, new_param: torch.nn.Parameter
):
    """Set nested parameter given a dotted name like 'self_attn.qkv_proj.weight'."""
    parts = name.split(".")
    obj = module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_param)


def _update_param_data(param, new_tensor):
    """Update parameter while keeping properties needed by Offloader (e.g. pinned host memory)."""
    if param.device == new_tensor.device:
        param.data = new_tensor
    else:
        assert param.device == torch.device(
            "cpu"
        ), f"{param.device=} {new_tensor.device=}"
        param.data = _create_cpu_data(new_tensor, pin_memory=True)


def _even_chunk(x: torch.Tensor, chunks: int):
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

    output_tensors = []
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


def _move_param_to_cpu(param, pin_memory: bool):
    param.data = _create_cpu_data(param.data, pin_memory=pin_memory)


def _create_cpu_data(data, pin_memory: bool):
    cpu_data = _empty_strided_like(
        data,
        device="cpu",
        pin_memory=pin_memory,
    )
    cpu_data.copy_(data)
    return cpu_data


def _move_param_to_meta(module, param_name):
    old_param = getattr(module, param_name)
    old_param_type = type(old_param)

    new_data = old_param.data.to("meta")

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
    else:
        raise ValueError(f"Unknown {old_param_type=} {old_param=}")

    setattr(module, param_name, new_param)


def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )
