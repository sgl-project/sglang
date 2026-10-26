"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._decode_runner = None
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._device_id = self._device_module.current_device()
        self._tp_group = get_parallel().tp_group
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )
        # Reuse one device-bound worker for graph input updates.
        self._update_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="npu-graph-update",
            initializer=self._device_module.set_device,
            initargs=(self._device_id,),
        )
        # Event of the last replay; the next rebind waits on it so the device is
        # done reading the host-side seq_lens array before we rewrite it.
        self._rebind_fence = None

    @property
    def cache_loc_dtype(self):
        return torch.int32

    def bind_decode_runner(self, runner):
        self._decode_runner = runner

    def _decode_update_attr_name(self):
        runner = self._decode_runner
        architectures = runner.model_runner.model_config.hf_config.architectures or []
        use_v2 = any(
            arch
            in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "Step3p5ForCausalLM")
            for arch in architectures
        ) or (
            envs.SGLANG_NPU_USE_FIAS_V2_BSND.get()
            and runner.model_runner.spec_algorithm.is_dspark()
            and runner.capture_forward_mode.is_target_verify()
        )
        return "actual_seq_kvlen" if use_v2 else "actual_seq_lengths_kv"

    def _decode_update_lengths(self, forward_batch):
        runner = self._decode_runner
        if forward_batch.forward_mode.is_target_verify():
            if runner.model_runner.spec_algorithm.is_dflash():
                attn_backend = runner._replay_attn_backend()
                metadata = getattr(attn_backend, "forward_metadata", None)
                metadata_lengths = getattr(metadata, "seq_lens_cpu_list", None)
            else:
                metadata_lengths = None
            if metadata_lengths is not None:
                seq_lens = list(metadata_lengths)
            else:
                if forward_batch.seq_lens_cpu is not None:
                    seq_lens_cpu = (
                        forward_batch.seq_lens_cpu[: runner.raw_bs]
                        + runner.captured_req_width
                    )
                else:
                    seq_lens_cpu = (
                        forward_batch.seq_lens.cpu() + runner.captured_req_width
                    )
                seq_lens = seq_lens_cpu.tolist() + [0] * (runner.bs - runner.raw_bs)
        else:
            if forward_batch.seq_lens_cpu is not None:
                seq_lens = forward_batch.seq_lens_cpu[: runner.raw_bs].tolist() + [
                    0
                ] * (runner.bs - runner.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    runner.bs - runner.raw_bs
                )
        return seq_lens

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        runner = self._decode_runner
        if runner is not None:
            from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4
            from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
                EAGLEDraftExtendCudaGraphRunner,
            )

            hf_config = runner.model_runner.model_config.hf_config
            if not (is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)):
                if isinstance(runner, EAGLEDraftExtendCudaGraphRunner):
                    cpu_update_input = self._eagle_draft_extend_update_lengths(
                        runner, static_forward_batch
                    )
                else:
                    cpu_update_input = [
                        {
                            self._decode_update_attr_name(): self._decode_update_lengths(
                                static_forward_batch
                            )
                        }
                    ]
                return self.replay_with_input_update(
                    shape_key, seq_lens=None, cpu_update_input=cpu_update_input
                )
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    @staticmethod
    def _eagle_draft_extend_update_lengths(runner, forward_batch):
        seq_lens = forward_batch.seq_lens_cpu.tolist() + [0] * (
            runner.bs - runner.raw_bs
        )
        return [{"actual_seq_lengths_kv": seq_lens}]

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph, then replay.

        NPUGraph.update must complete before replay can consume the updated
        KV lengths. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]

        if self._rebind_fence is not None:
            self._rebind_fence.synchronize()

        update_future = self._update_executor.submit(
            graph.update, cpu_update_input=cpu_update_input
        )
        graph.replay()
        update_future.result()

        fence = self._device_module.Event()
        fence.record()
        self._rebind_fence = fence
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._update_executor.shutdown(wait=True, cancel_futures=True)
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
        self._decode_runner = None
