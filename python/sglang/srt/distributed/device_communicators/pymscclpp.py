import importlib
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

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


@dataclass(frozen=True)
class _TuningCandidate:
    algorithm: Any
    nblocks: tuple[int, ...]
    nthreads: tuple[int, ...]
    min_message_size: int
    max_message_size: int


@dataclass(frozen=True)
class _TwoKernelAllReduce:
    name: str
    reduce_scatter: Any
    allgather: Any
    rank: int
    world_size: int

    @property
    def message_size_range(self) -> tuple[int, int]:
        reduce_scatter_range = self.reduce_scatter.message_size_range
        allgather_range = self.allgather.message_size_range
        return (
            max(reduce_scatter_range[0], allgather_range[0]),
            min(reduce_scatter_range[1], allgather_range[1]),
        )

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
            op=op,
            stream=stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads_per_block,
            symmetric_memory=symmetric_memory,
        )

    def reset(self):
        self.reduce_scatter.reset()
        self.allgather.reset()


class PyMscclppCommunicator:
    _SUPPORTED_WORLD_SIZES = [4, 8, 16, 32, 64]
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]
    _ALLGATHER_MIN_TOTAL_BYTES = 1 << 10
    _ALLGATHER_MAX_TOTAL_BYTES = 8 << 20

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

    def _get_allreduce_tuned_config(self, size):
        if size <= 512:
            target_size = 512
        elif size > 256 * 1024 * 1024:
            target_size = 256 * 1024 * 1024
        else:
            target_size = 1 << (size - 1).bit_length()
        return self.allreduce_best_configs.get(target_size)

    def _get_allgather_tuned_config(self, size):
        total_size = size * self.world_size
        if not (
            self._ALLGATHER_MIN_TOTAL_BYTES
            <= total_size
            <= self._ALLGATHER_MAX_TOTAL_BYTES
        ):
            return None
        target_size = 1 << (size - 1).bit_length()
        return self.allgather_best_configs.get(target_size)

    def _compile_dsl_candidate(
        self,
        *,
        name,
        collective,
        algorithm_builder,
        num_threads_per_block,
        plan_min_message_size,
        plan_max_message_size,
        tuning_min_message_size=None,
        tuning_max_message_size=None,
        instr_fusion=True,
        compile_kwargs=None,
    ):
        spec = self.mscclpp.language.AlgoSpec(
            name=name,
            collective=collective,
            nranks_per_node=self.nranks_per_ipc_domain,
            world_size=self.world_size,
            in_place=collective.inplace,
            instances=1,
            protocol="LL",
            auto_sync=False,
            instr_fusion=instr_fusion,
            num_threads_per_block=num_threads_per_block,
            reuse_resources=True,
            use_double_scratch_buffer=True,
            min_message_size=plan_min_message_size,
            max_message_size=plan_max_message_size,
            tags={"default": 1},
        )
        algorithm = self.mscclpp.compile(
            algorithm_builder,
            spec,
            self.rank,
            **(compile_kwargs or {}),
        )
        message_range = algorithm.message_size_range
        return _TuningCandidate(
            algorithm=algorithm,
            nblocks=(0,),
            nthreads=(0,),
            min_message_size=(
                message_range[0]
                if tuning_min_message_size is None
                else tuning_min_message_size
            ),
            max_message_size=(
                message_range[1]
                if tuning_max_message_size is None
                else tuning_max_message_size
            ),
        )

    def _create_dsl_allreduce_algorithms(self):
        algorithms = []
        n_ipc_domains = self.world_size // self.nranks_per_ipc_domain
        if n_ipc_domains not in (2, 4, 8):
            return algorithms

        for tbg in (1, 2, 4, 8):
            for num_threads_per_block in (256, 512, 768, 1024):
                algorithms.append(
                    self._compile_dsl_candidate(
                        name=(
                            f"allreduce_{n_ipc_domains}node_"
                            f"{tbg}TBG_{num_threads_per_block}TPB"
                        ),
                        collective=self.mscclpp.language.collectives.AllReduce(
                            self.world_size,
                            1,
                            True,
                        ),
                        algorithm_builder=self.def_algo.allreduce_multi_nodes,
                        num_threads_per_block=num_threads_per_block,
                        plan_min_message_size=tbg * (1 << 10),
                        plan_max_message_size=8 << 20,
                        compile_kwargs={"thread_block_group_size": tbg},
                    )
                )
        return algorithms

    def _create_dsl_allgather_algorithms(self):
        if not hasattr(self.def_algo, "allgather_multi_nodes"):
            raise RuntimeError(
                "The installed MSCCL++ package does not provide "
                "default_algos.allgather_multi_nodes"
            )

        algorithms = []
        n_ipc_domains = self.world_size // self.nranks_per_ipc_domain
        if n_ipc_domains not in (2, 4, 8):
            return algorithms

        input_min = self._ALLGATHER_MIN_TOTAL_BYTES // self.world_size
        input_max = self._ALLGATHER_MAX_TOTAL_BYTES // self.world_size
        for num_threads_per_block in self.allgather_tuning_threads:
            algorithms.append(
                self._compile_dsl_candidate(
                    name=(
                        f"allgather_{n_ipc_domains}node_1TBG_"
                        f"{num_threads_per_block}TPB"
                    ),
                    collective=self.mscclpp.language.collectives.AllGather(
                        self.world_size,
                        1,
                        False,
                    ),
                    algorithm_builder=self.def_algo.allgather_multi_nodes,
                    num_threads_per_block=num_threads_per_block,
                    plan_min_message_size=self._ALLGATHER_MIN_TOTAL_BYTES,
                    plan_max_message_size=self._ALLGATHER_MAX_TOTAL_BYTES,
                    tuning_min_message_size=input_min,
                    tuning_max_message_size=input_max,
                )
            )
        return algorithms

    def _create_dsl_reducescatter_algorithms(self):
        if not hasattr(self.def_algo, "reducescatter_multi_nodes"):
            return []

        algorithms = []
        n_ipc_domains = self.world_size // self.nranks_per_ipc_domain
        tbg_min_message_size = {
            1: 1 << 10,
            2: 1 << 20,
            4: 2 << 20,
            8: 8 << 20,
        }
        for tbg in (1, 2, 4, 8):
            for num_threads_per_block in (256, 512, 768, 1024):
                algorithms.append(
                    self._compile_dsl_candidate(
                        name=(
                            f"reducescatter_{n_ipc_domains}node_"
                            "directowner_v4_inplace_unfused_"
                            f"{tbg}TBG_{num_threads_per_block}TPB"
                        ),
                        collective=self.mscclpp.language.collectives.ReduceScatter(
                            self.world_size,
                            1,
                            True,
                        ),
                        algorithm_builder=self.def_algo.reducescatter_multi_nodes,
                        num_threads_per_block=num_threads_per_block,
                        plan_min_message_size=1 << 10,
                        plan_max_message_size=8 << 20,
                        tuning_min_message_size=tbg_min_message_size[tbg],
                        instr_fusion=False,
                        compile_kwargs={"thread_block_group_size": tbg},
                    )
                )
        return algorithms

    def _create_native_allreduce_algorithms(self):
        native_algorithms_config = []
        force_disable_nvls = os.getenv("MSCCLPP_FORCE_DISABLE_NVLS") == "1"
        dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(
            data_type=str(torch.float16)
        )
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        self.flag_buffer = torch.ones(128, dtype=torch.uint32, device="cuda")
        algos = self.mscclpp_ext.AlgorithmCollectionBuilder().build_default_algorithms(
            scratch_buffer=self.scratch_buffer.data_ptr(),
            scratch_buffer_size=self.scratch_buffer.nbytes,
            rank=self.rank,
        )

        for algo in algos:
            if force_disable_nvls and "nvls" in algo.name:
                continue
            if algo.name == "default_allreduce_nvls_packet":
                algo.set_message_size_range(0, 512 << 10)
                native_algorithms_config.append(
                    _TuningCandidate(
                        algorithm=algo,
                        nblocks=(4, 8, 12, 16),
                        nthreads=(256, 512, 768, 1024),
                        min_message_size=0,
                        max_message_size=512 << 10,
                    )
                )
            if algo.name == "default_allreduce_packet":
                algo.set_message_size_range(0, 2 << 20)
                native_algorithms_config.append(
                    _TuningCandidate(
                        algorithm=algo,
                        nblocks=(14, 21, 28, 42, 56),
                        nthreads=(256, 512, 768, 1024),
                        min_message_size=0,
                        max_message_size=2 << 20,
                    )
                )
            if algo.name == "default_allreduce_rsag_zero_copy":
                algo.set_message_size_range(512 << 10, 4 << 30)
                native_algorithms_config.append(
                    _TuningCandidate(
                        algorithm=algo,
                        nblocks=(32, 48, 64, 128),
                        nthreads=(256, 512, 768, 1024),
                        min_message_size=512 << 10,
                        max_message_size=4 << 30,
                    )
                )
            if (
                self.symm_mem_enabled
                and algo.name == "default_allreduce_nvls_zero_copy"
            ):
                algo.set_message_size_range(512 << 10, 4 << 30)
                native_algorithms_config.append(
                    _TuningCandidate(
                        algorithm=algo,
                        nblocks=(4, 8, 12, 16, 32),
                        nthreads=(256, 512, 768, 1024),
                        min_message_size=512 << 10,
                        max_message_size=4 << 30,
                    )
                )

        return native_algorithms_config

    def _tune_collective(self, collective, algorithms):
        if not algorithms:
            raise RuntimeError(f"No MSCCL++ {collective} algorithms were compiled")
        self._tune(
            collective,
            algorithms,
            n_warmup=5,
            n_graph_launches=20,
            n_ops_per_graph=20,
        )
        best_configs = {
            "allreduce": self.allreduce_best_configs,
            "allgather": self.allgather_best_configs,
            "reducescatter": self.reducescatter_best_configs,
        }[collective]
        if not best_configs:
            raise RuntimeError(
                f"MSCCL++ {collective} produced no usable tuned configurations"
            )

    def _create_algorithms(self):
        n_ipc_domains = self.world_size // self.nranks_per_ipc_domain
        if n_ipc_domains == 1:
            if "allreduce" in self.collectives:
                self._tune_collective(
                    "allreduce",
                    self._create_native_allreduce_algorithms(),
                )
            return

        if n_ipc_domains not in (2, 4, 8):
            return

        need_rsag = "allreduce" in self.collectives and hasattr(
            self.def_algo,
            "reducescatter_multi_nodes",
        )
        if "allgather" in self.collectives or need_rsag:
            self._tune_collective(
                "allgather",
                self._create_dsl_allgather_algorithms(),
            )

        if "allreduce" in self.collectives:
            self._tune_collective(
                "allreduce",
                self._create_dsl_allreduce_algorithms(),
            )
            if need_rsag:
                self._tune_collective(
                    "reducescatter",
                    self._create_dsl_reducescatter_algorithms(),
                )
                self._merge_rsag_configs(
                    n_warmup=5,
                    n_graph_launches=20,
                    n_ops_per_graph=20,
                )

    def _merge_rsag_configs(
        self,
        *,
        n_warmup,
        n_graph_launches,
        n_ops_per_graph,
    ):
        n_ipc_domains = self.world_size // self.nranks_per_ipc_domain
        element_size = torch.empty((), dtype=torch.bfloat16).element_size()
        size = 1 << 10
        while size <= 8 << 20:
            reduce_scatter_config = self.reducescatter_best_configs.get(size)
            allgather_config = self.allgather_best_configs.get(size // self.world_size)
            if reduce_scatter_config is None or allgather_config is None:
                size <<= 1
                continue

            reduce_scatter = reduce_scatter_config[0]
            allgather = allgather_config[0]
            if size not in self.allreduce_best_configs:
                size <<= 1
                continue
            reduce_scatter_prefix = (
                f"reducescatter_{n_ipc_domains}node_" "directowner_v4_inplace_unfused_"
            )
            allgather_prefix = f"allgather_{n_ipc_domains}node_1TBG_"
            algorithm = _TwoKernelAllReduce(
                name=(
                    f"allreduce_rsag_{n_ipc_domains}node_"
                    f"RS{reduce_scatter.name.removeprefix(reduce_scatter_prefix)}_"
                    f"AG{allgather.name.removeprefix(allgather_prefix)}"
                ),
                reduce_scatter=reduce_scatter,
                allgather=allgather,
                rank=self.rank,
                world_size=self.world_size,
            )
            tensor = torch.empty(
                size // element_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            elapsed = self._get_time(
                "allreduce",
                algorithm,
                tensor,
                tensor,
                size,
                0,
                0,
                n_warmup,
                n_graph_launches,
                n_ops_per_graph,
                symmetric_memory=False,
            )
            legacy_time = self.allreduce_best_config_times.get(
                size,
                float("inf"),
            )
            if elapsed < legacy_time:
                self.allreduce_best_configs[size] = (algorithm, 0, 0)
                self.allreduce_best_config_times[size] = elapsed
            if self.rank == 0:
                logger.info(
                    "MSCCL++ allreduce RSAG merge: input_bytes=%d "
                    "reduce_scatter=%s allgather=%s time_ms=%.4f "
                    "selected=%s",
                    size,
                    reduce_scatter.name,
                    allgather.name,
                    elapsed,
                    elapsed < legacy_time,
                )
            del tensor
            size <<= 1

    def _get_time(
        self,
        collective,
        algo,
        input_tensor,
        output_tensor,
        input_size,
        nblocks,
        nthreads,
        n_warmup,
        n_graph_launches,
        n_ops_per_graph,
        symmetric_memory=False,
    ):
        def run(stream=None):
            return self._run_algo(
                collective,
                algo,
                input_tensor,
                nblocks,
                nthreads,
                output_tensor=output_tensor,
                input_size=input_size,
                stream=stream,
                symmetric_memory=symmetric_memory,
            )

        result = run()
        if result != 0:
            raise RuntimeError(
                f"MSCCL++ {collective} tuning failed with error code {result}"
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

    def _tune(
        self,
        collective: str,
        algos_config: list[_TuningCandidate],
        *,
        n_warmup: int = 5,
        n_graph_launches: int = 20,
        n_ops_per_graph: int = 1,
    ):
        if collective not in {"allreduce", "allgather", "reducescatter"}:
            raise ValueError(f"Unsupported tuning collective: {collective}")

        if collective == "allreduce":
            size = 1 << 9
            max_size = 1 << 23
            dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(
                data_type=str(torch.float16)
            )
            allreduce_tensor = torch.utils.dlpack.from_dlpack(dlpack)
        elif collective == "allgather":
            size = self._ALLGATHER_MIN_TOTAL_BYTES // self.world_size
            max_size = self._ALLGATHER_MAX_TOTAL_BYTES // self.world_size
            allreduce_tensor = None
        else:
            size = self._ALLGATHER_MIN_TOTAL_BYTES
            max_size = self._ALLGATHER_MAX_TOTAL_BYTES
            allreduce_tensor = None

        while size <= max_size:
            warmup = n_warmup
            graph_launches = n_graph_launches
            if collective == "allgather":
                elements = (
                    size
                    // torch.empty(
                        (),
                        dtype=torch.bfloat16,
                    ).element_size()
                )
                input_tensor = torch.empty(
                    elements,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output_tensor = torch.empty(
                    elements * self.world_size,
                    dtype=input_tensor.dtype,
                    device=self.device,
                )
            elif collective == "reducescatter":
                elements = (
                    size
                    // torch.empty(
                        (),
                        dtype=torch.bfloat16,
                    ).element_size()
                )
                input_tensor = torch.empty(
                    elements,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                output_tensor = input_tensor[: elements // self.world_size]
            else:
                input_tensor = allreduce_tensor
                output_tensor = allreduce_tensor

            best_time = float("inf")
            best_config = None
            for candidate in algos_config:
                if not (
                    candidate.min_message_size <= size <= candidate.max_message_size
                ):
                    continue
                for nblocks in candidate.nblocks:
                    for nthreads in candidate.nthreads:
                        if self.rank == 0:
                            logger.info(
                                "MSCCL++ %s tuning candidate: input_bytes=%d "
                                "algorithm=%s nblocks=%d nthreads=%d",
                                collective,
                                size,
                                candidate.algorithm.name,
                                nblocks,
                                nthreads,
                            )
                        avg_time = self._get_time(
                            collective,
                            candidate.algorithm,
                            input_tensor,
                            output_tensor,
                            size,
                            nblocks,
                            nthreads,
                            warmup,
                            graph_launches,
                            n_ops_per_graph,
                            symmetric_memory=collective == "allreduce",
                        )
                        config = (
                            candidate.algorithm,
                            nblocks,
                            nthreads,
                        )
                        if avg_time < best_time:
                            best_time = avg_time
                            best_config = config

            if best_config is not None:
                if collective == "allreduce":
                    self.allreduce_best_configs[size] = best_config
                    self.allreduce_best_config_times[size] = best_time
                elif collective == "allgather":
                    self.allgather_best_configs[size] = best_config
                    self.allgather_best_config_times[size] = best_time
                else:
                    self.reducescatter_best_configs[size] = best_config
                    self.reducescatter_best_config_times[size] = best_time
                if self.rank == 0:
                    logger.info(
                        "MSCCL++ %s tuning: input_bytes=%d "
                        "algorithm=%s nblocks=%d nthreads=%d time_ms=%.4f",
                        collective,
                        size,
                        best_config[0].name,
                        best_config[1],
                        best_config[2],
                        best_time,
                    )

            if collective in {"allgather", "reducescatter"}:
                del input_tensor, output_tensor
            size <<= 1

        torch.cuda.synchronize()
        if collective in {"allreduce", "reducescatter"}:
            for candidate in algos_config:
                candidate.algorithm.reset()

    def _run_algo(
        self,
        collective,
        algo,
        input_tensor,
        nblocks,
        nthreads,
        *,
        output_tensor=None,
        input_size=None,
        stream=None,
        symmetric_memory=False,
    ):
        if collective not in {"allreduce", "allgather", "reducescatter"}:
            raise ValueError(f"Unsupported MSCCL++ collective: {collective}")
        if stream is None:
            stream = torch.cuda.current_stream()
        if input_size is None:
            input_size = input_tensor.nbytes
        if output_tensor is None:
            output_tensor = input_tensor

        if collective == "allreduce":
            output_size = input_size
            reduce_op = self.mscclpp.ReduceOp.SUM
        elif collective == "allgather":
            output_size = output_tensor.nbytes
            reduce_op = self.mscclpp.ReduceOp.NOP
        else:
            output_size = output_tensor.nbytes
            reduce_op = self.mscclpp.ReduceOp.SUM

        return algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=input_tensor.data_ptr(),
            output_buffer=output_tensor.data_ptr(),
            input_size=input_size,
            output_size=output_size,
            dtype=self.dtype_to_mscclpp_dtype(input_tensor.dtype),
            op=reduce_op,
            stream=stream.cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=symmetric_memory,
        )

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        group_name: str = "anonymous",
        collectives: Optional[set[str]] = None,
        allgather_tuning_threads: tuple[int, ...] = (256, 512, 768, 1024),
    ) -> None:
        """Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
            group_name: a human-readable process-group name for logging.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self.available = False
        self.allreduce_best_configs = {}
        self.allreduce_best_config_times = {}
        self.allgather_best_configs = {}
        self.allgather_best_config_times = {}
        self.reducescatter_best_configs = {}
        self.reducescatter_best_config_times = {}
        self._logged_allgather_sizes = set()
        self.scratch_buffer = None
        self.flag_buffer = None
        self.collectives = frozenset(
            ("allreduce", "allgather") if collectives is None else collectives
        )
        unsupported_collectives = self.collectives - {"allreduce", "allgather"}
        if unsupported_collectives:
            raise ValueError(
                f"Unsupported MSCCL++ collectives: {unsupported_collectives}"
            )
        if not allgather_tuning_threads or any(
            threads <= 0 or threads > 1024 for threads in allgather_tuning_threads
        ):
            raise ValueError(
                "allgather_tuning_threads must contain values in [1, 1024]"
            )
        self.allgather_tuning_threads = tuple(allgather_tuning_threads)

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

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            # No need to initialize mscclpp for single GPU case.
            return

        if world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_mscclpp=True explicitly.",
                world_size,
                str(PyMscclppCommunicator._SUPPORTED_WORLD_SIZES),
            )
            return

        self.ranks = torch.distributed.get_process_group_ranks(group)
        # for now mscclpp with stride in the communicator is not tested
        if not (abs(self.ranks[-1] - self.ranks[0]) == world_size - 1):
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
            torch_group=self.group, rank=rank, size=world_size
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
            self.allreduce_best_configs or self.allgather_best_configs
        )
        if not self.available:
            logger.warning("PyMscclpp did not produce any usable tuned configurations.")
        else:
            logger.info(
                "Created MSCCL++ communicator: group=%s world_size=%d device=%s "
                "allreduce_configs=%d allgather_configs=%d",
                group_name,
                world_size,
                self.device,
                len(self.allreduce_best_configs),
                len(self.allgather_best_configs),
            )

    def destroy(self):
        self.allreduce_best_configs = None
        self.allreduce_best_config_times = None
        self.allgather_best_configs = None
        self.allgather_best_config_times = None
        self.reducescatter_best_configs = None
        self.reducescatter_best_config_times = None
        self.executor = None
        self.scratch_buffer = None
        self.flag_buffer = None
        self.comm = None

    def should_mscclpp_allreduce(
        self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> bool:
        if (
            self.disabled
            or self.world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES
        ):
            return False
        if inp.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if not self._is_weak_contiguous(inp):
            return False
        if op is not ReduceOp.SUM:
            return False
        if self._get_allreduce_tuned_config(inp.numel() * inp.element_size()) is None:
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
        if (
            self.disabled
            or self.world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES
        ):
            return False
        if input_tensor.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if output_tensor.dtype != input_tensor.dtype:
            return False
        if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
            return False
        if output_tensor.numel() != input_tensor.numel() * self.world_size:
            return False
        if output_tensor.device != input_tensor.device:
            return False
        nbytes = input_tensor.numel() * input_tensor.element_size()
        if nbytes % 16 != 0:
            return False
        config = self._get_allgather_tuned_config(nbytes)
        if config is None:
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
        assert op == torch.distributed.ReduceOp.SUM
        nbytes = tensor.numel() * tensor.element_size()
        algo, nblocks, nthreads = self._get_allreduce_tuned_config(nbytes)
        result = self._run_algo(
            "allreduce",
            algo,
            tensor,
            nblocks,
            nthreads,
            input_size=nbytes,
            symmetric_memory=self.symm_mem_enabled,
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
        nbytes = input_tensor.numel() * input_tensor.element_size()
        config = self._get_allgather_tuned_config(nbytes)
        if config is None:
            raise RuntimeError(
                f"No tuned MSCCL++ AllGather configuration for {nbytes} bytes"
            )
        algo, nblocks, nthreads = config
        if self.rank == 0 and nbytes not in self._logged_allgather_sizes:
            self._logged_allgather_sizes.add(nbytes)
            logger.info(
                "Dispatching MSCCL++ AllGather: input_bytes=%d "
                "total_bytes=%d algorithm=%s",
                nbytes,
                nbytes * self.world_size,
                algo.name,
            )
        result = self._run_algo(
            "allgather",
            algo,
            input_tensor,
            nblocks,
            nthreads,
            output_tensor=output_tensor,
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
            # guess a default value when not specified
            # DO: Decided if raise an exception here or not
            enable = self.available

        old_disable = self.disabled
        self.disabled = not enable

        yield

        self.disabled = old_disable
