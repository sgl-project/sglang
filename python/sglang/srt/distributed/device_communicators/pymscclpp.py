import importlib
import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_server_args

from .base import AllReduceMode, BaseCommunicator

logger = logging.getLogger(__name__)


class PyMscclppCommunicator(BaseCommunicator):
    name = "pymscclpp"
    _SUPPORTED_WORLD_SIZES = [8, 16, 32]
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]

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

    def _get_tuned_config(self, size):
        if size <= 512:
            target_size = 512
        elif size > 256 * 1024 * 1024:
            target_size = 256 * 1024 * 1024
        else:
            target_size = 1 << (size - 1).bit_length()
        return self.best_configs.get(target_size)

    def _create_dsl_algorithms(self):
        dsl_algos_config = []
        n_nodes = self.world_size // self.nranks_per_node
        if n_nodes == 2 or n_nodes == 4:
            for tbg in [1, 2, 4, 8]:
                for num_threads_per_block in [256, 512, 768, 1024]:
                    spec = self.mscclpp.language.AlgoSpec(
                        name=f"allreduce_{n_nodes}node_{tbg}TBG_{num_threads_per_block}TPB",
                        collective=self.mscclpp.language.collectives.AllReduce(
                            self.world_size, 1, True
                        ),
                        nranks_per_node=self.nranks_per_node,
                        world_size=self.world_size,
                        in_place=True,
                        instances=1,
                        protocol="LL",
                        auto_sync=False,
                        num_threads_per_block=num_threads_per_block,
                        reuse_resources=True,
                        use_double_scratch_buffer=True,
                        min_message_size=tbg * (1 << 10),
                        max_message_size=8 << 20,
                        tags={"default": 1},
                    )
                    algo = self.mscclpp.compile(
                        self.def_algo.allreduce_multi_nodes,
                        spec,
                        self.rank,
                        thread_block_group_size=tbg,
                    )
                    dsl_algos_config.append((algo, [0], [0]))
        return dsl_algos_config

    def _create_native_algorithms(self):
        navitve_algorithms_config = []
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
            if algo.name == "default_allreduce_nvls_packet":
                algo.set_message_size_range(0, 512 << 10)
                navitve_algorithms_config.append(
                    (algo, [4, 8, 12, 16], [256, 512, 768, 1024])
                )
            if algo.name == "default_allreduce_packet":
                algo.set_message_size_range(0, 2 << 20)
                navitve_algorithms_config.append(
                    (algo, [14, 21, 28, 42, 56], [256, 512, 768, 1024])
                )
            if algo.name == "default_allreduce_rsag_zero_copy":
                algo.set_message_size_range(512 << 10, 4 << 30)
                navitve_algorithms_config.append(
                    (algo, [32, 48, 64, 128], [256, 512, 768, 1024])
                )
            if (
                self.symm_mem_enabled
                and algo.name == "default_allreduce_nvls_zero_copy"
            ):
                algo.set_message_size_range(512 << 10, 4 << 30)
                navitve_algorithms_config.append(
                    (algo, [4, 8, 12, 16, 32], [256, 512, 768, 1024])
                )

        return navitve_algorithms_config

    def _create_algorithms(self):
        if self.world_size == 8:
            self.algos_config = self._create_native_algorithms()
            self._tune(5, 10, 20, self.algos_config)
        elif self.world_size == 16 or self.world_size == 32:
            self.dsl_algos_config = self._create_dsl_algorithms()
            self._tune(5, 10, 20, self.dsl_algos_config)

    def _get_time(
        self,
        algo,
        tune_tensor,
        size,
        nb,
        nt,
        n_warmup,
        n_graph_launches,
        n_ops_per_graph,
    ):
        # Check if the algorithm can run with the given configuration
        if self._run_algo(algo, tune_tensor, size, nb, nt, True) != 0:
            return float("inf")

        # Warmup iterations to stabilize performance
        for _ in range(n_warmup):
            self._run_algo(algo, tune_tensor, size, nb, nt, True)

        # Warmup on capture stream
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            self._run_algo(algo, tune_tensor, size, nb, nt, True)
        capture_stream.synchronize()

        # Capture the algorithm execution in a CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            for _ in range(n_ops_per_graph):
                self._run_algo(algo, tune_tensor, size, nb, nt, True)

        # Measure the execution time of the captured graph
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(capture_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(n_graph_launches):
                g.replay()
        end_event.record(capture_stream)
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event)

        # Synchronize timing results across all ranks to ensure consistent algorithm selection
        # replicate n times such due to algo limitations
        time_tensor = torch.full(
            (self.world_size,), elapsed, dtype=torch.float64, device="cuda"
        ).to(dtype=torch.float32)
        torch.cuda.current_stream().wait_stream(capture_stream)
        if self.rank == 0:
            avg_time = time_tensor[self.rank].item() / self.world_size
            tensor = torch.tensor([avg_time])
        else:
            tensor = torch.empty(1)
        dist.broadcast(tensor, src=0, group=self.group)
        avg_time = tensor.item()

        return avg_time

    def _tune(self, n_warmup, n_graph_launches, n_ops_per_graph, algos_config):
        sizes = [1 << i for i in range(9, 24)]
        dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(
            data_type=str(torch.float16)
        )
        tune_tensor = torch.utils.dlpack.from_dlpack(dlpack)

        for size in sizes:
            best_time = float("inf")
            best_config = None
            for i in range(len(algos_config)):
                algo, candidates_nblocks, candidates_nthreads = algos_config[i]
                if (
                    size >= algo.message_size_range[0]
                    and size <= algo.message_size_range[1]
                ):
                    for nb in candidates_nblocks:
                        for nt in candidates_nthreads:
                            avg_time = self._get_time(
                                algo,
                                tune_tensor,
                                size,
                                nb,
                                nt,
                                n_warmup,
                                n_graph_launches,
                                n_ops_per_graph,
                            )
                            if avg_time < best_time:
                                best_time = avg_time
                                best_config = (algo, nb, nt)
            if best_config:
                self.best_configs[size] = best_config

        torch.cuda.synchronize()
        for algo, _, _ in algos_config:
            algo.reset()

    def _run_algo(self, algo, tensor, size, nblocks, nthreads, sym_mem_enabled=False):
        return algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=size,
            output_size=size,
            dtype=self.dtype_to_mscclpp_dtype(tensor.dtype),
            op=self.mscclpp.ReduceOp.SUM,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=sym_mem_enabled,
        )

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
    ) -> None:
        """Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """

        try:
            self.mscclpp = importlib.import_module("mscclpp")
            self.mscclpp_ext = importlib.import_module("mscclpp.ext")
            self.def_algo = importlib.import_module("mscclpp.default_algos")
        except ImportError as e:
            # e.g. mscclpp is not installed or in a non-cuda environment
            raise RuntimeError(
                "PyMscclpp is disabled because the mscclpp library is not found. "
                "To silence this warning, specify disable_mscclpp=True explicitly."
            ) from e

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "PyMscclppCommunicator should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        assert world_size > 1
        # PyMscclpp is enabled only in cuda graph
        super().__init__(world_size=world_size, disabled=True)

        if world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "PyMscclpp is disabled due to an unsupported world size: "
                f"{world_size}. Supported world sizes: "
                f"{PyMscclppCommunicator._SUPPORTED_WORLD_SIZES}. To silence "
                "this warning, specify disable_mscclpp=True explicitly."
            )

        self.ranks = torch.distributed.get_process_group_ranks(group)
        self.nranks_per_node = torch.cuda.device_count()
        # for now mscclpp with stride in the communicator is not tested
        if not (abs(self.ranks[-1] - self.ranks[0]) == world_size - 1):
            raise ValueError(
                f"PyMscclpp is disabled due to an unsupported group {self.ranks}. "
                "Please ensure all ranks in the group are consecutive. "
                "To silence this warning, specify disable_mscclpp=True explicitly."
            )

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        self.rank = rank
        self.world_size = world_size
        self.comm = self.mscclpp.CommGroup(
            torch_group=self.group, rank=rank, size=world_size
        )
        self.executor = self.mscclpp.Executor(self.comm.communicator)
        self.symm_mem_enabled = self._is_symm_mem_enabled()
        self.best_configs = {}
        self._create_algorithms()

    def destroy(self):
        self.algos_config = None
        self.best_configs = None
        self.executor = None
        self.scratch_buffer = None
        self.flag_buffer = None
        self.comm = None

    def graph_capture_context(self):
        return self.change_state(enable=True)

    def should_use_custom_op(self) -> bool:
        return True

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        if self.disabled:
            return None
        if input_.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return None
        if not self._is_weak_contiguous(input_):
            return None
        if self._get_tuned_config(input_.numel() * input_.element_size()) is None:
            return None
        # mscclpp must not be used during any piecewise CUDA graph phase
        # (compile, capture, or replay) as it changes the allreduce dispatch
        # path and triggers recompilation.
        if (
            is_in_tc_piecewise_cuda_graph()
            or is_in_torch_compile_warmup()
            or get_pcg_capture_stream() is not None
        ):
            return None
        # the mscclpp kernel reduces into the input buffer
        return AllReduceMode.INPLACE

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

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        self.assert_inplace("all_reduce", inplace)
        nbytes = input_.numel() * input_.element_size()
        algo, nblocks, nthreads = self._get_tuned_config(nbytes)
        self._run_algo(algo, input_, nbytes, nblocks, nthreads, self.symm_mem_enabled)
        return input_
