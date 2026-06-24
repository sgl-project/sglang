# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TcPiecewiseCudaGraphBackend — torch.compile-driven piecewise CUDA graph.

FX-splits the model forward at attention layers; per-shape compiled
callables internally capture sub-graphs via
compilation/cuda_piecewise_backend. torch.compile owns the per-shape
cache so this backend has no _graphs table — only a single
_compiled_fn reused for every shape.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import tqdm

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.compile_phase import (
    enable_torch_compile_warmup,
    set_pcg_capture_stream,
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    from sglang.srt.server_args import ServerArgs


_VALID_COMPILERS = ("eager", "inductor")


def _toggle_multi_platform_ops(
    model: torch.nn.Module, *, reverse: bool, num_tokens: int
) -> None:
    """Recursively flip MultiPlatformOp submodules into / out of
    torch.compile mode."""
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _toggle_multi_platform_ops(sub, reverse=reverse, num_tokens=num_tokens)


class TcPiecewiseCudaGraphBackend(BaseCudaGraphBackend):
    """torch.compile-driven piecewise capture; attention metadata
    recomputed at replay outside the compiled callable's sub-graphs.
    """

    def __init__(self, cuda_graph_runner: BaseCudaGraphRunner) -> None:
        model_runner = cuda_graph_runner.model_runner
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._compile_config: CompilationConfig = self.build_compilation_config(
            model_runner.server_args
        )
        self._language_model: torch.nn.Module = getattr(
            model_runner.model, "language_model", model_runner.model
        )
        self._run_compile_pass(cuda_graph_runner)
        # model_runner.model.forward is the wrapper that builds LogitsProcessorOutput.
        # The compiled trampoline is dispatched internally by it.
        self._compiled_fn: Callable = model_runner.model.forward

    @staticmethod
    def build_compilation_config(server_args: ServerArgs) -> CompilationConfig:
        """Construct a CompilationConfig from ServerArgs and
        register the MoE A2A split-op when DeepEP / Mooncake is in use."""
        prefill = server_args.cuda_graph_config.prefill
        num_tokens = prefill.bs
        compiler = prefill.tc_compiler
        assert num_tokens is not None, "cuda_graph_config[prefill].bs is not set"
        assert compiler in _VALID_COMPILERS, (
            f"By now, only {_VALID_COMPILERS} are supported for the "
            "tc_piecewise prefill compiler."
        )

        config = CompilationConfig(
            num_tokens,
            compiler,
            server_args.enable_torch_compile_debug_mode,
        )

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            config.add_split_op("sglang.moe_forward_piecewise_cuda_graph_impl")

        return config

    @staticmethod
    def install_compile(
        language_model: Any,
        *,
        compile_config: CompilationConfig,
        graph_pool: Any,
        fullgraph: bool = True,
        dynamic_arg_dims: Optional[Any] = None,
    ) -> None:
        """Wrap language_model.model.forward with torch.compile."""
        install_torch_compiled(
            language_model,
            fullgraph=fullgraph,
            dynamic_arg_dims=dynamic_arg_dims,
            compile_config=compile_config,
            graph_pool=graph_pool,
        )

    def _run_compile_pass(self, cuda_graph_runner: BaseCudaGraphRunner) -> None:
        """JIT-activate kernels at the smallest shape, install
        torch.compile, then run one forward per shape inside
        enable_torch_compile_warmup to drive FX / inductor through
        every shape without capturing cuda graphs yet."""
        language_model = self._language_model
        compiler = self._compile_config.compiler
        with enable_tc_piecewise_cuda_graph():
            try:
                if compiler != "eager":
                    _toggle_multi_platform_ops(
                        language_model.model, reverse=False, num_tokens=16
                    )

                cuda_graph_runner._run_dummy_forward(
                    num_tokens=cuda_graph_runner.capture_num_tokens[0]
                )

                if self._pool is None:
                    self._pool = self._device_module.graph_pool_handle()
                set_graph_pool_id(self._pool)

                self.install_compile(
                    language_model.model,
                    compile_config=self._compile_config,
                    graph_pool=self._pool,
                )

                with enable_torch_compile_warmup():
                    if is_hip():
                        # AMD: single Dynamo trace is sufficient; the capture
                        # phase does per-shape JIT kernel warmup before each
                        # CUDA graph recording.  The N-iteration loop is
                        # redundant and extremely slow on ROCm (~30 min).
                        cuda_graph_runner._run_dummy_forward(
                            num_tokens=cuda_graph_runner.capture_num_tokens[-1]
                        )
                    else:
                        compile_range = (
                            tqdm.tqdm(
                                list(reversed(cuda_graph_runner.capture_num_tokens))
                            )
                            if get_tensor_model_parallel_rank() == 0
                            else reversed(cuda_graph_runner.capture_num_tokens)
                        )
                        for num_tokens in compile_range:
                            if get_tensor_model_parallel_rank() == 0:
                                compile_range.set_description(
                                    f"Compiling num tokens ({num_tokens=})"
                                )
                            cuda_graph_runner._run_dummy_forward(num_tokens=num_tokens)
            finally:
                _toggle_multi_platform_ops(
                    language_model.model, reverse=True, num_tokens=16
                )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        self._capture_stream = stream
        try:
            with self.replay_session():
                with set_pcg_capture_stream(stream):
                    yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # Call 1 warms FX state; call 2 captures the cuda graph inside capture_session.
        # See cuda_piecewise_backend.py for the FX backend that drives the capture.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        # torch.compile manages its per-shape cache internally.
        # _run_compile_pass warms every shape in capture_num_tokens at __init__.
        return True

    @contextmanager
    def replay_session(self):
        with enable_tc_piecewise_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        return self._compiled_fn(
            static_forward_batch.input_ids,
            static_forward_batch.positions,
            static_forward_batch,
            **kwargs,
        )

    def cleanup(self) -> None:
        self._compiled_fn = None
        self._compile_config = None
        self._language_model = None
        self._pool = None
