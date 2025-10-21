import logging
import os
from abc import ABC
from typing import Callable, Generator, List, Optional

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

        self.offloaders = []

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        assert len(self.offloaders) == 0, "should only call wrap_modules once"

        alt_stream = torch.cuda.Stream()

        all_modules = []
        offload_submodules = []
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
                        alt_stream=alt_stream,
                        whitelist_param_names=whitelist_param_names,
                    )
                )

        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index,
                module=module,
                offloaders=self.offloaders,
                prefetch_step=self.prefetch_step,
            )

        return all_modules

    def post_init(self):
        for offloader in self.offloaders:
            offloader.post_init()

        for i in range(self.prefetch_step):
            self.offloaders[i].start_onload()

    @property
    def forbid_copy_engine_usage(self):
        return self.mode == "cpu"


def _hook_module_forward_for_offloader(index, module, offloaders, prefetch_step):
    def _on_forward_end():
        offloaders[(index + prefetch_step) % len(offloaders)].start_onload()
        offloaders[index].offload()

    _hook_module_forward_raw(
        module,
        on_forward_end=_on_forward_end,
        get_parameter_and_buffer_dicts=lambda: offloaders[
            index
        ].wait_and_get_device_tensors(),
    )


def _hook_module_forward_raw(module, on_forward_end, get_parameter_and_buffer_dicts):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        output = functional_call(
            module, get_parameter_and_buffer_dicts(), args=args, kwargs=kwargs
        )
        on_forward_end()
        module.forward = forward
        return output

    module.forward = forward


class _ModuleOffloader(ABC):
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
        ), "not handled device=cpu case yet (should skip this tensor)"

        self._device_tensors = None
        self._load_event = None

        param_dict = dict(self.module.named_parameters())
        assert all(
            name in param_dict for name in whitelist_param_names
        ), f"{whitelist_param_names=} {list(param_dict.keys())=}"

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for name, param_offloader in self._param_offloaders.items():
            param_offloader.post_init()

    def start_onload(self):
        self.alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.alt_stream):
            self._device_tensors = self._create_device_tensors()
            self._load_event = torch.cuda.Event()
            self._load_event.record()

    def offload(self):
        self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self):
        assert self._device_tensors is not None
        self._load_event.wait()
        return self._device_tensors

    def _create_device_tensors(self):
        return {k: v.create_device_tensor() for k, v in self._param_offloaders.items()}


class _BaseParamOffloader(ABC):
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        return {
            "meta": _MetaParamOffloader,
            "cpu": _CpuParamOffloader,
            "shm_cpu": _ShmCpuParamOffloader,
            "sharded_gpu": _ShardedGpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, module, param_name):
        self._module = module
        self._param_name = param_name

    @property
    def _param(self):
        return getattr(self._module, self._param_name)

    def post_init(self):
        pass

    def create_device_tensor(self):
        raise NotImplementedError


class _MetaParamOffloader(_BaseParamOffloader):
    """Usually used for debugging."""

    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_meta(module, param_name)

    def create_device_tensor(self):
        return torch.empty_like(self._param.data, device="cuda")


class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_cpu(self._param, pin_memory=True)

    def create_device_tensor(self):
        return self._param.to("cuda", non_blocking=True)


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

    def create_device_tensor(self):
        return self.shm_cpu_data.to("cuda", non_blocking=True)


def update_param(param, new_tensor):
    """Update parameter while keeping properties needed by Offloader (e.g. pinned host memory)."""

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
