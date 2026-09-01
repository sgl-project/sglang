import importlib
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_server_args

logger = logging.getLogger(__name__)

_SUPPORTED_COLLECTIVES = {"allreduce", "allgather", "reducescatter"}
_DEFAULT_SUPPORTED_DTYPES = (torch.float, torch.float16, torch.bfloat16)
_DEFAULT_GPU_BUFFER_SIZE = 1 << 27
_TUNING_MESSAGE_SIZES = tuple(1 << exponent for exponent in range(9, 24))


@dataclass(frozen=True)
class _MessageSizeRange:
    minimum: int
    maximum: int

    def __post_init__(self):
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError(f"Invalid message size range: {self}")

    def contains(self, size: int) -> bool:
        return self.minimum <= size <= self.maximum


@dataclass(frozen=True, kw_only=True)
class _AlgorithmConfig(ABC):
    implementation: ClassVar[str]
    name: str
    collective: str
    world_sizes: tuple[int, ...]
    ipc_domain_counts: tuple[int, ...]
    threads_per_block: tuple[int, ...]
    message_size_range: _MessageSizeRange
    reduce_op: str
    supported_dtypes: tuple[torch.dtype, ...] = _DEFAULT_SUPPORTED_DTYPES
    in_place: bool = True
    requires_nvls: bool = False
    algorithm: Any = None

    def __post_init__(self):
        if self.collective not in _SUPPORTED_COLLECTIVES:
            raise ValueError(f"Unsupported collective: {self.collective}")
        if self.reduce_op not in {"SUM", "NOP"}:
            raise ValueError(f"Unsupported reduction operation: {self.reduce_op}")
        if not self.world_sizes or not self.ipc_domain_counts:
            raise ValueError("Algorithm topology constraints cannot be empty")
        if not self.threads_per_block:
            raise ValueError("Algorithm topology constraints cannot be empty")
        if not self.supported_dtypes:
            raise ValueError("Algorithm supported dtypes cannot be empty")

    def supports_topology(self, world_size: int, ipc_domain_count: int) -> bool:
        return (
            world_size in self.world_sizes
            and ipc_domain_count in self.ipc_domain_counts
        )

    def requirements_satisfied(
        self,
        world_size: int,
        ipc_domain_count: int,
        nvls_supported: bool,
        symmetric_memory: bool,
    ) -> bool:
        return self.supports_topology(
            world_size, ipc_domain_count
        ) and self.support_nvls(nvls_supported, symmetric_memory)

    def support_nvls(self, nvls_supported: bool, symmetric_memory: bool) -> bool:
        return not self.requires_nvls or (nvls_supported and symmetric_memory)

    def supports_message_size(self, message_size: int) -> bool:
        return self.message_size_range.contains(message_size)

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in self.supported_dtypes

    def resolve_reduce_op(self, reduce_ops):
        return getattr(reduce_ops, self.reduce_op)

    def input_buffer_size(self, message_size: int, world_size: int) -> int:
        if self.collective == "allgather":
            if message_size % world_size != 0:
                raise ValueError(
                    f"All-gather output size {message_size} must be divisible "
                    f"by world size {world_size}"
                )
            return message_size // world_size
        return message_size

    def output_buffer_size(self, message_size: int, world_size: int) -> int:
        if self.collective == "reducescatter":
            if message_size % world_size != 0:
                raise ValueError(
                    f"Reduce-scatter input size {message_size} must be divisible "
                    f"by world size {world_size}"
                )
            return message_size // world_size
        return message_size

    @abstractmethod
    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        raise NotImplementedError

    def bind(self, algorithm) -> "_AlgorithmConfig":
        return replace(self, algorithm=algorithm)

    def select(self, nblocks: int, threads_per_block: int) -> "_AlgorithmConfig":
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        if (nblocks, threads_per_block) not in self.tuning_launches():
            raise ValueError(f"Invalid launch selection for algorithm {self.name}")
        return self

    def selected_launch(self) -> tuple[int, int]:
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        return self.tuning_launches()[0]

    def reset(self):
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        self.algorithm.reset()

    @staticmethod
    def compose_rsag(
        reduce_scatter: "_AlgorithmConfig",
        allgather: "_AlgorithmConfig",
        *,
        rank: int,
        world_size: int,
        ipc_domain_count: int,
        reduce_ops: Any,
    ) -> "_CompositeAlgorithmConfig":
        if (
            reduce_scatter.collective != "reducescatter"
            or allgather.collective != "allgather"
        ):
            raise ValueError("RSAG requires reduce-scatter and all-gather algorithms")
        if reduce_scatter.algorithm is None or allgather.algorithm is None:
            raise RuntimeError("RSAG composition requires bound algorithms")
        message_size_range = _MessageSizeRange(
            minimum=max(
                reduce_scatter.message_size_range.minimum,
                allgather.message_size_range.minimum,
            ),
            maximum=min(
                reduce_scatter.message_size_range.maximum,
                allgather.message_size_range.maximum,
            ),
        )
        algorithm = _TwoKernelAllReduce(
            name=(
                f"allreduce_rsag_{reduce_scatter.algorithm.name}_"
                f"{allgather.algorithm.name}"
            ),
            reduce_scatter=reduce_scatter.algorithm,
            allgather=allgather.algorithm,
            allgather_op=allgather.resolve_reduce_op(reduce_ops),
            rank=rank,
            world_size=world_size,
        )
        return _CompositeAlgorithmConfig(
            name=algorithm.name,
            collective="allreduce",
            world_sizes=(world_size,),
            ipc_domain_counts=(ipc_domain_count,),
            threads_per_block=(0,),
            message_size_range=message_size_range,
            reduce_op="SUM",
            supported_dtypes=tuple(
                dtype
                for dtype in reduce_scatter.supported_dtypes
                if dtype in allgather.supported_dtypes
            ),
            algorithm=algorithm,
        )


@dataclass(frozen=True, kw_only=True)
class _DslAlgorithmConfig(_AlgorithmConfig):
    implementation: ClassVar[str] = "dsl"
    algo_spec: Any
    algorithm_kwargs: tuple[dict[str, Any], ...] = ()
    name_variant: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.threads_per_block:
            raise ValueError("DSL compile thread candidates cannot be empty")
        if any(threads <= 0 for threads in self.threads_per_block):
            raise ValueError("DSL compile thread candidates must be positive")
        if self.algorithm is None:
            if self.algo_spec.world_size != 0 or self.algo_spec.nranks_per_node != 0:
                raise ValueError("DSL AlgoSpec templates require zero topology values")
            if self.algo_spec.name != self.name:
                raise ValueError("DSL AlgoSpec and algorithm names must match")
        elif self.algo_spec.world_size <= 0 or self.algo_spec.nranks_per_node <= 0:
            raise ValueError("Bound DSL algorithms require a materialized AlgoSpec")
        if self.algo_spec.collective.name != self.collective:
            raise ValueError("DSL AlgoSpec and algorithm collectives must match")
        if self.algo_spec.in_place != self.in_place:
            raise ValueError("DSL AlgoSpec and algorithm buffer modes must match")

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        return ((0, 0),)

    def bind(self, algorithm, algo_spec) -> "_AlgorithmConfig":
        return replace(self, algorithm=algorithm, algo_spec=algo_spec)

    def dsl_name(
        self,
        ipc_domain_count: int,
        threads_per_block: int,
        algorithm_kwargs: dict[str, Any],
    ) -> str:
        variant_parts = [self.name_variant] if self.name_variant else []
        variant_parts.extend(
            f"{key}_{value}" for key, value in sorted(algorithm_kwargs.items())
        )
        variant = f"{'_'.join(variant_parts)}_" if variant_parts else ""
        return f"{self.name}_{ipc_domain_count}node_{variant}" f"{threads_per_block}TPB"


@dataclass(frozen=True, kw_only=True)
class _NativeAlgorithmConfig(_AlgorithmConfig):
    implementation: ClassVar[str] = "native"
    nblocks: tuple[int, ...]
    selected_launch_parameters: Optional[tuple[int, int]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.nblocks:
            raise ValueError("Native launch candidates cannot be empty")
        if self.selected_launch_parameters is not None:
            if self.algorithm is None:
                raise ValueError("Only bound native algorithms can select a launch")
            if self.selected_launch_parameters not in self.tuning_launches():
                raise ValueError("Invalid native launch selection")

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (nblocks, threads_per_block)
            for nblocks in self.nblocks
            for threads_per_block in self.threads_per_block
        )

    def select(self, nblocks: int, threads_per_block: int) -> "_AlgorithmConfig":
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        launch = (nblocks, threads_per_block)
        if launch not in self.tuning_launches():
            raise ValueError(f"Invalid launch selection for algorithm {self.name}")
        return replace(self, selected_launch_parameters=launch)

    def selected_launch(self) -> tuple[int, int]:
        if self.selected_launch_parameters is None:
            raise RuntimeError(f"Algorithm {self.name} has not been tuned")
        return self.selected_launch_parameters


@dataclass(frozen=True, kw_only=True)
class _CompositeAlgorithmConfig(_AlgorithmConfig):
    implementation: ClassVar[str] = "composite"

    def __post_init__(self):
        super().__post_init__()
        if self.algorithm is None:
            raise ValueError("Composite algorithms must be bound when constructed")

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        return ((0, 0),)


_DEFAULT_THREADS_PER_BLOCK = (256, 512, 768, 1024)
_DEFAULT_THREAD_BLOCK_GROUP_SIZES = (1, 2, 4, 8)
_SUPPORTED_WORLD_SIZES = (4, 8, 16, 32, 64)
_NATIVE_IPC_DOMAIN_COUNTS = (1,)
_MULTI_NODE_IPC_DOMAIN_COUNTS = (2, 4, 8)


_NATIVE_ALGORITHM_CONFIGS = (
    _NativeAlgorithmConfig(
        name="default_allreduce_nvls_packet",
        collective="allreduce",
        world_sizes=_SUPPORTED_WORLD_SIZES,
        ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
        message_size_range=_MessageSizeRange(0, 512 << 10),
        nblocks=(4, 8, 12, 16),
        threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
        reduce_op="SUM",
        requires_nvls=True,
    ),
    _NativeAlgorithmConfig(
        name="default_allreduce_packet",
        collective="allreduce",
        world_sizes=_SUPPORTED_WORLD_SIZES,
        ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
        message_size_range=_MessageSizeRange(0, 2 << 20),
        nblocks=(14, 21, 28, 42, 56),
        threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
        reduce_op="SUM",
    ),
    _NativeAlgorithmConfig(
        name="default_allreduce_rsag_zero_copy",
        collective="allreduce",
        world_sizes=_SUPPORTED_WORLD_SIZES,
        ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
        message_size_range=_MessageSizeRange(512 << 10, 4 << 30),
        nblocks=(32, 48, 64, 128),
        threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
        reduce_op="SUM",
    ),
    _NativeAlgorithmConfig(
        name="default_allreduce_nvls_zero_copy",
        collective="allreduce",
        world_sizes=_SUPPORTED_WORLD_SIZES,
        ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
        message_size_range=_MessageSizeRange(512 << 10, 4 << 30),
        nblocks=(4, 8, 12, 16, 32),
        threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
        reduce_op="SUM",
        requires_nvls=True,
    ),
)


def _create_algorithm_configs(language) -> tuple[_AlgorithmConfig, ...]:
    default_spec = language.AlgoSpec(
        name="allreduce_multi_nodes",
        collective=language.collectives.AllReduce(0, 1, True),
        nranks_per_node=0,
        world_size=0,
        in_place=True,
        instances=1,
        protocol="LL",
        instr_fusion=True,
        auto_sync=False,
        replication_policy=language.ReplicationPolicy.interleaved,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        buffer_alignment=16,
    )
    allgather_spec = replace(
        default_spec,
        name="allgather_multi_nodes",
        collective=language.collectives.AllGather(0, 1, False),
        in_place=False,
    )
    reduce_scatter_spec = replace(
        default_spec,
        name="reducescatter_multi_nodes",
        collective=language.collectives.ReduceScatter(0, 1, True),
        instr_fusion=False,
    )
    dsl_configs = (
        _DslAlgorithmConfig(
            name="allreduce_multi_nodes",
            collective="allreduce",
            world_sizes=_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_MULTI_NODE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 8 << 20),
            reduce_op="SUM",
            algo_spec=default_spec,
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            algorithm_kwargs=tuple(
                {"thread_block_group_size": thread_block_group_size}
                for thread_block_group_size in _DEFAULT_THREAD_BLOCK_GROUP_SIZES
            ),
        ),
        _DslAlgorithmConfig(
            name="allgather_multi_nodes",
            collective="allgather",
            world_sizes=_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_MULTI_NODE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 8 << 20),
            reduce_op="NOP",
            in_place=False,
            algo_spec=allgather_spec,
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
        ),
        _DslAlgorithmConfig(
            name="reducescatter_multi_nodes",
            collective="reducescatter",
            world_sizes=_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_MULTI_NODE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 8 << 20),
            reduce_op="SUM",
            algo_spec=reduce_scatter_spec,
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            algorithm_kwargs=tuple(
                {"thread_block_group_size": thread_block_group_size}
                for thread_block_group_size in _DEFAULT_THREAD_BLOCK_GROUP_SIZES
            ),
            name_variant="directowner_v4_inplace_unfused",
        ),
    )
    return (*dsl_configs, *_NATIVE_ALGORITHM_CONFIGS)


@dataclass(frozen=True)
class _TwoKernelAllReduce:
    name: str
    reduce_scatter: Any
    allgather: Any
    allgather_op: Any
    rank: int
    world_size: int

    def execute(
        self,
        *,
        comm,
        executor,
        input_buffer,
        output_buffer,
        input_size,
        output_size,
        dtype,
        op,
        stream,
        nblocks,
        nthreads_per_block,
        symmetric_memory,
    ):
        if input_buffer != output_buffer:
            raise ValueError("RSAG AllReduce requires in-place buffers")
        if input_size % (16 * self.world_size) != 0:
            raise ValueError(
                f"RSAG AllReduce input size {input_size} must be divisible "
                f"by {16 * self.world_size}"
            )
        shard_size = input_size // self.world_size

        result = self.reduce_scatter.execute(
            comm=comm,
            executor=executor,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            input_size=input_size,
            output_size=shard_size,
            dtype=dtype,
            op=op,
            stream=stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads_per_block,
            symmetric_memory=symmetric_memory,
        )
        if result != 0:
            return result
        return self.allgather.execute(
            comm=comm,
            executor=executor,
            input_buffer=output_buffer + self.rank * shard_size,
            output_buffer=output_buffer,
            input_size=shard_size,
            output_size=output_size,
            dtype=dtype,
            op=self.allgather_op,
            stream=stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads_per_block,
            symmetric_memory=symmetric_memory,
        )

    def reset(self):
        self.reduce_scatter.reset()
        self.allgather.reset()


class PyMscclppCommunicator:
    def _is_symm_mem_enabled(self) -> bool:
        try:
            return get_server_args().enable_symm_mem
        except ValueError:
            return False

    def _is_weak_contiguous(self, inp: torch.Tensor):
        return inp.is_contiguous() or (
            inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
            == inp.numel() * inp.element_size()
        )

    def _get_tuned_config(self, collective: str, message_size: int):
        if message_size <= 0:
            return None
        configs = self._best_configs[collective]
        target_size = 1 << (message_size - 1).bit_length()
        config = configs.get(target_size)
        if config is not None and config.supports_message_size(message_size):
            return config

        if not configs:
            return None
        minimum_tuned_size = min(configs)
        minimum_config = configs[minimum_tuned_size]
        if target_size < minimum_tuned_size and minimum_config.supports_message_size(
            message_size
        ):
            return minimum_config
        return None

    def _get_allreduce_tuned_config(self, message_size):
        return self._get_tuned_config("allreduce", message_size)

    def _get_allgather_tuned_config(self, message_size):
        return self._get_tuned_config("allgather", message_size)

    def _algorithm_configs(self, ipc_domain_count: int):
        return [
            config
            for config in self._registered_algorithm_configs
            if config.requirements_satisfied(
                self.world_size,
                ipc_domain_count,
                nvls_supported=self.mscclpp.is_nvls_supported(),
                symmetric_memory=self.symm_mem_enabled,
            )
        ]

    def _compile_dsl_algorithm(
        self,
        config: _DslAlgorithmConfig,
        ipc_domain_count: int,
    ) -> list[_DslAlgorithmConfig]:
        algorithm_builder = getattr(self.def_algo, config.name, None)
        if algorithm_builder is None:
            raise RuntimeError(
                "The installed MSCCL++ package does not provide "
                f"default_algos.{config.name}"
            )
        template_collective = config.algo_spec.collective
        collective = type(template_collective)(
            self.world_size,
            template_collective.chunk_factor,
            config.in_place,
        )
        algorithms = []
        algorithm_kwargs_variants = config.algorithm_kwargs or ({},)
        for algorithm_kwargs in algorithm_kwargs_variants:
            for threads_per_block in config.threads_per_block:
                spec = replace(
                    config.algo_spec,
                    name=config.dsl_name(
                        ipc_domain_count,
                        threads_per_block,
                        algorithm_kwargs,
                    ),
                    collective=collective,
                    nranks_per_node=self.nranks_per_ipc_domain,
                    world_size=self.world_size,
                    num_threads_per_block=threads_per_block,
                )
                algorithm = self.mscclpp.compile(
                    algorithm_builder,
                    spec,
                    self.rank,
                    **algorithm_kwargs,
                )
                algorithms.append(config.bind(algorithm, spec))
        return algorithms

    def _create_dsl_algorithms(
        self,
        configs: list[_DslAlgorithmConfig],
        ipc_domain_count: int,
    ) -> list[_DslAlgorithmConfig]:
        algorithms = []
        for config in configs:
            algorithms.extend(self._compile_dsl_algorithm(config, ipc_domain_count))
        return algorithms

    def _create_native_algorithms(
        self, configs: list[_NativeAlgorithmConfig]
    ) -> list[_NativeAlgorithmConfig]:
        dlpack = self.mscclpp.RawGpuBuffer(_DEFAULT_GPU_BUFFER_SIZE).to_dlpack(
            data_type=str(torch.float16)
        )
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        algos = self.mscclpp_ext.AlgorithmCollectionBuilder().build_default_algorithms(
            scratch_buffer=self.scratch_buffer.data_ptr(),
            scratch_buffer_size=self.scratch_buffer.nbytes,
            rank=self.rank,
        )
        configs_by_name = {config.name: config for config in configs}
        algorithms = []

        for algo in algos:
            config = configs_by_name.get(algo.name)
            if config is None:
                continue
            message_range = config.message_size_range
            algo.set_message_size_range(message_range.minimum, message_range.maximum)
            algorithms.append(config.bind(algo))

        return algorithms

    def _create_algorithm_candidates(
        self,
        configs: list[_AlgorithmConfig],
        ipc_domain_count: int,
    ) -> list[_AlgorithmConfig]:
        native_configs = [
            config for config in configs if isinstance(config, _NativeAlgorithmConfig)
        ]
        dsl_configs = [
            config for config in configs if isinstance(config, _DslAlgorithmConfig)
        ]
        algorithms = []
        if native_configs:
            algorithms.extend(self._create_native_algorithms(native_configs))
        if dsl_configs:
            algorithms.extend(
                self._create_dsl_algorithms(dsl_configs, ipc_domain_count)
            )
        return algorithms

    def _create_algorithms(self):
        ipc_domain_count = self.world_size // self.nranks_per_ipc_domain
        algorithms = self._create_algorithm_candidates(
            self._algorithm_configs(ipc_domain_count),
            ipc_domain_count,
        )
        algorithms_by_collective = {
            collective: [
                algorithm
                for algorithm in algorithms
                if algorithm.collective == collective
            ]
            for collective in ("allreduce", "allgather", "reducescatter")
        }
        allreduce_algorithms = algorithms_by_collective["allreduce"]
        allgather_algorithms = algorithms_by_collective["allgather"]
        reduce_scatter_algorithms = algorithms_by_collective["reducescatter"]
        need_rsag = (
            "allreduce" in self.collectives
            and bool(allgather_algorithms)
            and bool(reduce_scatter_algorithms)
        )
        if "allgather" in self.collectives or need_rsag:
            if allgather_algorithms:
                self._tune(allgather_algorithms)

        if "allreduce" in self.collectives and allreduce_algorithms:
            self._tune(allreduce_algorithms)
            if need_rsag:
                self._tune(reduce_scatter_algorithms)
                self._merge_rsag_configs()

    def _merge_rsag_configs(self):
        allreduce_configs = self._best_configs["allreduce"]
        allgather_configs = self._best_configs["allgather"]
        reduce_scatter_configs = self._best_configs["reducescatter"]
        allreduce_times = self._best_config_times["allreduce"]
        allgather_times = self._best_config_times["allgather"]
        reduce_scatter_times = self._best_config_times["reducescatter"]
        for message_size, reduce_scatter_config in sorted(
            reduce_scatter_configs.items()
        ):
            allreduce_config = allreduce_configs.get(message_size)
            allgather_config = allgather_configs.get(message_size)
            if allreduce_config is None or allgather_config is None:
                continue

            composite = _AlgorithmConfig.compose_rsag(
                reduce_scatter_config,
                allgather_config,
                rank=self.rank,
                world_size=self.world_size,
                ipc_domain_count=(self.world_size // self.nranks_per_ipc_domain),
                reduce_ops=self.mscclpp.ReduceOp,
            )
            try:
                composite_time = (
                    reduce_scatter_times[message_size] + allgather_times[message_size]
                )
                allreduce_time = allreduce_times[message_size]
            except KeyError as exc:
                raise RuntimeError(
                    "RSAG merge requires timings for all tuned algorithms"
                ) from exc
            selected = composite_time < allreduce_time
            if selected:
                allreduce_configs[message_size] = composite
                allreduce_times[message_size] = composite_time
            if self.rank == 0:
                logger.info(
                    "MSCCL++ allreduce RSAG merge: input_bytes=%d "
                    "reduce_scatter=%s allgather=%s estimated_time_ms=%.4f "
                    "selected=%s",
                    message_size,
                    reduce_scatter_config.algorithm.name,
                    allgather_config.algorithm.name,
                    composite_time,
                    selected,
                )

    def _get_time(
        self,
        algorithm_config,
        input_tensor,
        output_tensor,
        message_size,
        nblocks,
        nthreads,
        n_warmup,
        n_graph_launches,
        n_ops_per_graph,
    ):
        def run(stream=None):
            return self._run_algorithm(
                algorithm_config,
                input_tensor,
                output_tensor,
                message_size,
                nblocks,
                nthreads,
                stream=stream,
            )

        result = run()
        if result != 0:
            raise RuntimeError(
                f"MSCCL++ {algorithm_config.collective} tuning failed "
                f"with error code {result}"
            )

        for _ in range(n_warmup):
            run()

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            run(capture_stream)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(n_ops_per_graph):
                run(capture_stream)

        capture_stream.synchronize()
        dist.barrier(group=self.group)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(capture_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(n_graph_launches):
                graph.replay()
        end_event.record(capture_stream)
        end_event.synchronize()
        torch.cuda.current_stream().wait_stream(capture_stream)

        elapsed = start_event.elapsed_time(end_event) / (
            n_graph_launches * n_ops_per_graph
        )
        max_elapsed = torch.tensor([elapsed], dtype=torch.float64)
        dist.all_reduce(max_elapsed, op=ReduceOp.MAX, group=self.group)
        return max_elapsed.item()

    def _allocate_tuning_tensor(self, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
        element_size = torch.empty((), dtype=dtype).element_size()
        if nbytes <= 0 or nbytes % element_size != 0:
            raise ValueError(
                f"Tuning buffer size {nbytes} is invalid for dtype {dtype}"
            )
        with torch.cuda.device(self.device):
            dlpack = self.mscclpp.RawGpuBuffer(nbytes).to_dlpack(data_type=str(dtype))
            return torch.utils.dlpack.from_dlpack(dlpack)

    def _tune(
        self,
        algorithms: list[_AlgorithmConfig],
        n_warmup: int = 5,
        n_graph_launches: int = 20,
        n_ops_per_graph: int = 20,
    ):
        dtype = torch.bfloat16
        element_size = torch.empty((), dtype=dtype).element_size()
        tuning_cases = []
        for message_size in _TUNING_MESSAGE_SIZES:
            candidates = [
                candidate
                for candidate in algorithms
                if candidate.supports_message_size(message_size)
            ]
            if not candidates:
                continue
            input_size = max(
                candidate.input_buffer_size(message_size, self.world_size)
                for candidate in candidates
            )
            output_size = max(
                candidate.output_buffer_size(message_size, self.world_size)
                for candidate in candidates
            )
            tuning_cases.append((message_size, candidates, input_size, output_size))

        if tuning_cases:
            input_buffer = self._allocate_tuning_tensor(
                max(case[2] for case in tuning_cases), dtype
            )
            output_buffer = self._allocate_tuning_tensor(
                max(case[3] for case in tuning_cases), dtype
            )

        for message_size, candidates, input_size, output_size in tuning_cases:
            input_tensor = input_buffer[: input_size // element_size]
            output_tensor = output_buffer[: output_size // element_size]

            best_time = float("inf")
            best_config = None
            for candidate in candidates:
                candidate_output = input_tensor if candidate.in_place else output_tensor
                for nblocks, nthreads in candidate.tuning_launches():
                    if self.rank == 0:
                        threads_per_block = (
                            candidate.algo_spec.num_threads_per_block
                            if isinstance(candidate, _DslAlgorithmConfig)
                            else nthreads
                        )
                        logger.debug(
                            "MSCCL++ %s tuning candidate: message_bytes=%d "
                            "algorithm=%s nblocks=%d nthreads=%d",
                            candidate.collective,
                            message_size,
                            candidate.algorithm.name,
                            nblocks,
                            threads_per_block,
                        )
                    avg_time = self._get_time(
                        candidate,
                        input_tensor,
                        candidate_output,
                        message_size,
                        nblocks,
                        nthreads,
                        n_warmup,
                        n_graph_launches,
                        n_ops_per_graph,
                    )
                    config = (candidate, nblocks, nthreads)
                    if avg_time < best_time:
                        best_time = avg_time
                        best_config = config

            if best_config is not None:
                self._best_configs[candidate.collective][message_size] = best_config[
                    0
                ].select(
                    best_config[1],
                    best_config[2],
                )
                self._best_config_times[candidate.collective][message_size] = best_time
                if self.rank == 0:
                    threads_per_block = (
                        best_config[0].algo_spec.num_threads_per_block
                        if isinstance(best_config[0], _DslAlgorithmConfig)
                        else best_config[2]
                    )
                    logger.info(
                        "MSCCL++ %s tuning: message_bytes=%d "
                        "algorithm=%s nblocks=%d nthreads=%d time_ms=%.4f",
                        candidate.collective,
                        message_size,
                        best_config[0].algorithm.name,
                        best_config[1],
                        threads_per_block,
                        best_time,
                    )
            del input_tensor, output_tensor
        torch.cuda.synchronize()
        for candidate in algorithms:
            candidate.reset()

    def _run_algorithm(
        self,
        config: _AlgorithmConfig,
        input_tensor,
        output_tensor,
        message_size,
        nblocks,
        nthreads,
        *,
        stream=None,
    ):
        if stream is None:
            stream = torch.cuda.current_stream()

        if config.algorithm is None:
            raise RuntimeError(f"Algorithm {config.name} has not been bound")
        input_size = config.input_buffer_size(message_size, self.world_size)
        output_size = config.output_buffer_size(message_size, self.world_size)
        return config.algorithm.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=input_tensor.data_ptr(),
            output_buffer=output_tensor.data_ptr(),
            input_size=input_size,
            output_size=output_size,
            dtype=self.dtype_to_mscclpp_dtype(input_tensor.dtype),
            op=config.resolve_reduce_op(self.mscclpp.ReduceOp),
            stream=stream.cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=self.symm_mem_enabled,
        )

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        group_name: str = "anonymous",
        collectives: Optional[set[str]] = None,
    ) -> None:
        """Args:
            group: A non-NCCL process group used to bootstrap MSCCL++ and
                synchronize tuning results.
            device: The CUDA device used by this rank.
            group_name: a human-readable process-group name for logging.
            collectives: Public collectives to compile and tune. Reduce-scatter
                may be tuned internally to form an all-reduce candidate.

        Ranks must be consecutive, and each rank must be bound to its own
        device. Supported multi-node layouts contain 2, 4, or 8 IPC domains.
        """
        self.disabled = True
        self.available = False
        self._best_configs = {
            collective: {} for collective in ("allreduce", "allgather", "reducescatter")
        }
        self._best_config_times = {
            collective: {} for collective in ("allreduce", "allgather", "reducescatter")
        }
        self.scratch_buffer = None
        self.collectives = frozenset(
            ("allreduce", "allgather") if collectives is None else collectives
        )
        unsupported_collectives = self.collectives - {"allreduce", "allgather"}
        if unsupported_collectives:
            raise ValueError(
                f"Unsupported MSCCL++ collectives: {unsupported_collectives}"
            )

        try:
            self.mscclpp = importlib.import_module("mscclpp")
            self.mscclpp_ext = importlib.import_module("mscclpp.ext")
            self.def_algo = importlib.import_module("mscclpp.default_algos")
        except ImportError as exc:
            logger.warning(
                "PyMscclpp is unavailable because its dependencies failed to import: %s",
                exc,
            )
            self.mscclpp = None
            return

        self._registered_algorithm_configs = _create_algorithm_configs(
            self.mscclpp.language
        )

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if self.world_size == 1:
            # No need to initialize mscclpp for single GPU case.
            return

        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_mscclpp=True explicitly.",
                self.world_size,
                str(_SUPPORTED_WORLD_SIZES),
            )
            return

        self.ranks = torch.distributed.get_process_group_ranks(group)
        # for now mscclpp with stride in the communicator is not tested
        if not (abs(self.ranks[-1] - self.ranks[0]) == self.world_size - 1):
            logger.warning(
                "PyMscclpp is disabled due to an unsupported group %s."
                "Please ensure all ranks in the group are consecutive."
                "To silence this warning, specify disable_mscclpp=True explicitly.",
                str(self.ranks),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        self.comm = self.mscclpp.CommGroup(
            torch_group=self.group, rank=self.rank, size=self.world_size
        )
        nranks_per_ipc_domain = getattr(self.comm, "nranks_per_ipc_domain", None)
        self.nranks_per_ipc_domain = (
            self.comm.nranks_per_node
            if nranks_per_ipc_domain is None
            else nranks_per_ipc_domain
        )
        self.executor = self.mscclpp.Executor(self.comm.communicator)
        self.symm_mem_enabled = self._is_symm_mem_enabled()
        try:
            self._create_algorithms()
            self.executor.reset()
        except Exception:
            logger.exception("Failed to initialize MSCCL++ collectives")
            raise
        self.available = bool(
            self._best_configs["allreduce"] or self._best_configs["allgather"]
        )
        if not self.available:
            logger.warning("PyMscclpp did not produce any usable tuned configurations.")
        else:
            logger.info(
                "Created MSCCL++ communicator: group=%s world_size=%d device=%s "
                "allreduce_configs=%d allgather_configs=%d",
                group_name,
                self.world_size,
                self.device,
                len(self._best_configs["allreduce"]),
                len(self._best_configs["allgather"]),
            )

    def destroy(self):
        self._best_configs = None
        self._best_config_times = None
        self.executor = None
        self.scratch_buffer = None
        self.comm = None

    def should_mscclpp_allreduce(
        self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> bool:
        if self.disabled or self.world_size not in _SUPPORTED_WORLD_SIZES:
            return False
        if not self._is_weak_contiguous(inp):
            return False
        if op is not ReduceOp.SUM:
            return False
        config = self._get_allreduce_tuned_config(inp.numel() * inp.element_size())
        if config is None or not config.supports_dtype(inp.dtype):
            return False
        # mscclpp must not be used during any piecewise CUDA graph phase
        # (compile, capture, or replay) as it changes the allreduce dispatch
        # path and triggers recompilation.
        if (
            is_in_tc_piecewise_cuda_graph()
            or is_in_torch_compile_warmup()
            or get_pcg_capture_stream() is not None
        ):
            return False
        return True

    def should_mscclpp_allgather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> bool:
        if self.disabled or self.world_size not in _SUPPORTED_WORLD_SIZES:
            return False
        if output_tensor.dtype != input_tensor.dtype:
            return False
        if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
            return False
        if output_tensor.numel() != input_tensor.numel() * self.world_size:
            return False
        if output_tensor.device != input_tensor.device:
            return False
        input_nbytes = input_tensor.numel() * input_tensor.element_size()
        if input_nbytes % 16 != 0:
            return False
        output_nbytes = output_tensor.numel() * output_tensor.element_size()
        config = self._get_allgather_tuned_config(output_nbytes)
        if config is None or not config.supports_dtype(input_tensor.dtype):
            return False
        if (
            is_in_tc_piecewise_cuda_graph()
            or is_in_torch_compile_warmup()
            or get_pcg_capture_stream() is not None
        ):
            return False
        return True

    def dtype_to_mscclpp_dtype(self, dtype: torch.dtype):
        if dtype == torch.float16:
            return self.mscclpp.DataType.float16
        elif dtype == torch.float32:
            return self.mscclpp.DataType.float32
        elif dtype == torch.int32:
            return self.mscclpp.DataType.int32
        elif dtype == torch.bfloat16:
            return self.mscclpp.DataType.bfloat16
        else:
            raise ValueError(f"Unknown data type: {dtype}")

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: torch.cuda.Stream = None,
    ):
        if op != torch.distributed.ReduceOp.SUM:
            raise ValueError("MSCCL++ AllReduce only supports SUM")
        nbytes = tensor.numel() * tensor.element_size()
        config = self._get_allreduce_tuned_config(nbytes)
        if config is None:
            raise RuntimeError(
                f"No tuned MSCCL++ AllReduce configuration for {nbytes} bytes"
            )
        nblocks, threads_per_block = config.selected_launch()
        result = self._run_algorithm(
            config,
            tensor,
            tensor,
            nbytes,
            nblocks,
            threads_per_block,
            stream=stream,
        )
        if result != 0:
            raise RuntimeError(f"MSCCL++ AllReduce failed with error code {result}")
        return tensor

    def all_gather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream = None,
    ):
        input_nbytes = input_tensor.numel() * input_tensor.element_size()
        output_nbytes = output_tensor.numel() * output_tensor.element_size()
        if output_nbytes != input_nbytes * self.world_size:
            raise ValueError(
                f"MSCCL++ AllGather output has {output_nbytes} bytes; expected "
                f"{input_nbytes * self.world_size} bytes"
            )
        config = self._get_allgather_tuned_config(output_nbytes)
        if config is None:
            raise RuntimeError(
                "No tuned MSCCL++ AllGather configuration for "
                f"{output_nbytes} output bytes"
            )
        nblocks, threads_per_block = config.selected_launch()
        result = self._run_algorithm(
            config,
            input_tensor,
            output_tensor,
            output_nbytes,
            nblocks,
            threads_per_block,
            stream=stream,
        )
        if result != 0:
            raise RuntimeError(f"MSCCL++ AllGather failed with error code {result}")
        return output_tensor

    @contextmanager
    def change_state(
        self,
        enable: Optional[bool] = None,
    ):
        if enable is None or self.available is False:
            enable = self.available

        old_disable = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disable
