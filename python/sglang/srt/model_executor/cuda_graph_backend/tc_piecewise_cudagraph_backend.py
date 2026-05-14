"""TcPiecewiseCudaGraphBackend — torch.compile-driven piecewise CUDA graph.

Uses ``CompilationConfig``, the FX/inductor pipeline from
``sglang.srt.compilation``, and the warmup-compile flag from
``compilation/compile_phase``. Produces piecewise graphs by FX-splitting
the model forward at attention layers; per-shape compiled callables
each internally capture sub-graphs via
``compilation/cuda_piecewise_backend``.

Unlike Full / Breakable, this backend doesn't keep a per-shape
``_graphs`` table — torch.compile owns the per-shape compiled
callable cache internally. The backend's only state is ``_compiled_fn``
(the wrapped model.forward, the same callable for every shape).
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
from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


_VALID_COMPILERS = ("eager", "inductor")


def _toggle_multi_platform_ops(
    model: torch.nn.Module, *, reverse: bool, num_tokens: int
) -> None:
    """Recursively flip MultiPlatformOp submodules into / out of torch.compile mode.

    Mirrors the legacy ``_to_torch`` walk; lighter than the full
    ``patch_model`` because tc_piecewise uses ``install_torch_compiled`` for
    actual compilation rather than calling ``torch.compile`` directly.
    """
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
    recomputed at replay (outside the compiled callable's sub-graphs).
    """

    def __init__(self) -> None:
        self._compile_config: Optional[CompilationConfig] = None
        self._language_model: Optional[torch.nn.Module] = None
        self._compiled_fn: Optional[Callable] = None
        self._device_module = None
        self._tp_group = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._pool = None

    # -----------------------------------------------------------------
    # Static helper retained from the previous version (used by the
    # legacy PCG runner during the migration window).
    # -----------------------------------------------------------------
    @staticmethod
    def build_compilation_config(server_args: "ServerArgs") -> CompilationConfig:
        """Construct the ``CompilationConfig`` from ``ServerArgs``.

        Validates the ``--piecewise-cuda-graph-compiler`` choice, builds
        the config, and registers the MoE A2A split-op when DeepEP /
        Mooncake is in use.
        """
        assert (
            server_args.piecewise_cuda_graph_tokens is not None
        ), "piecewise_cuda_graph_tokens is not set"
        assert server_args.piecewise_cuda_graph_compiler in _VALID_COMPILERS, (
            f"By now, only {_VALID_COMPILERS} are supported for piecewise "
            "cuda graph compiler."
        )

        config = CompilationConfig(
            server_args.piecewise_cuda_graph_tokens,
            server_args.piecewise_cuda_graph_compiler,
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
        """Wrap ``language_model`` with ``torch.compile`` via
        ``install_torch_compiled``. Side effect: model's forward is
        replaced with the compiled trampoline.
        """
        install_torch_compiled(
            language_model,
            fullgraph=fullgraph,
            dynamic_arg_dims=dynamic_arg_dims,
            compile_config=compile_config,
            graph_pool=graph_pool,
        )

    # -----------------------------------------------------------------
    # BaseCudaGraphBackend interface
    # -----------------------------------------------------------------
    def prepare(self, runner) -> None:
        """One-time setup: build CompilationConfig, JIT-activate kernels,
        install torch.compile wrapping on the language model, then run
        the compile-loop pass so torch.compile finishes FX/inductor
        compilation for every shape. Per-shape *cuda graph capture* is
        not done here — that lives in ``capture_one`` (matches Full /
        Breakable: prepare = setup, capture_one = warmup + record).

        Phase breakdown for tc_piecewise, paired with what Full / Breakable do:

          1. JIT-kernel-activate forward (1 call at smallest shape):
             warms shared CUDA kernels before torch.compile sees the
             model. tc_piecewise-only.
          2. install_compile: wraps ``language_model.model.forward``
             with ``torch.compile``. tc_piecewise-only.
          3. Compile-loop pass: 1 forward per shape inside
             ``enable_torch_compile_warmup`` — the FX backend short-
             circuits the cuda-graph branch (see
             ``cuda_piecewise_backend:143``) so this only triggers
             FX/inductor compilation. tc_piecewise-only.
          4. Per-shape warmup + record: handled by ``capture_one``;
             matches Full / Breakable's 2x warmup + 1x capture pattern.

        Requires the runner to expose:
          - ``runner.model_runner.server_args``
          - ``runner.model_runner.model`` (.language_model.model in
            multimodal case, or .model otherwise)
          - ``runner.device_module``
          - ``runner.capture_num_tokens``
          - ``runner._run_dummy_forward(num_tokens)`` callable
        """
        self._compile_config = self.build_compilation_config(
            runner.model_runner.server_args
        )
        self._device_module = runner.device_module
        self._tp_group = runner.model_runner.tp_group

        language_model = getattr(
            runner.model_runner.model, "language_model", runner.model_runner.model
        )
        self._language_model = language_model

        compiler = self._compile_config.compiler
        with enable_tc_piecewise_cuda_graph():
            try:
                if compiler != "eager":
                    _toggle_multi_platform_ops(
                        language_model.model, reverse=False, num_tokens=16
                    )

                # Step 1: JIT-activate kernels at the smallest shape.
                runner._run_dummy_forward(num_tokens=runner.capture_num_tokens[0])

                if self._pool is None:
                    self._pool = self._device_module.graph_pool_handle()
                set_graph_pool_id(self._pool)

                # Step 2: wrap model.forward with torch.compile.
                self.install_compile(
                    language_model.model,
                    compile_config=self._compile_config,
                    graph_pool=self._pool,
                )

                # Step 3: trigger FX/inductor compilation for every shape.
                # The FX backend skips cuda-graph capture while the
                # warmup flag is set.
                with enable_torch_compile_warmup():
                    compile_range = (
                        tqdm.tqdm(list(reversed(runner.capture_num_tokens)))
                        if get_tensor_model_parallel_rank() == 0
                        else reversed(runner.capture_num_tokens)
                    )
                    for num_tokens in compile_range:
                        if get_tensor_model_parallel_rank() == 0:
                            compile_range.set_description(
                                f"Compiling num tokens ({num_tokens=})"
                            )
                        runner._run_dummy_forward(num_tokens=num_tokens)
            finally:
                _toggle_multi_platform_ops(
                    language_model.model, reverse=True, num_tokens=16
                )

        # Replay invokes the outer model_runner.model.forward — the
        # wrapper that builds LogitsProcessorOutput. The torch.compile
        # we installed on language_model.model is dispatched internally
        # by that wrapper.
        self._compiled_fn = runner.model_runner.model.forward

    def can_run(self, forward_batch: "ForwardBatch") -> bool:
        return True

    def has_shape(self, shape_key: Any) -> bool:
        # torch.compile manages its per-shape cache internally; the
        # runner's capture loop ensures every shape in capture_num_tokens
        # has been warmed up. From the runner's perspective, every
        # shape ≤ max_num_tokens is supported (after replay_prepare's
        # bisect-bucket pad).
        return True

    @contextmanager
    def runtime_session(self):
        with enable_tc_piecewise_cuda_graph():
            yield

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        self._capture_stream = stream
        try:
            with self.runtime_session():
                with set_pcg_capture_stream(stream):
                    yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: Any,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
    ) -> None:
        """Per-shape: call 1 warms FX state, call 2 captures the cuda
        graph (inside ``capture_session``). See ``cuda_piecewise_backend.py``.
        """
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()

    def replay(
        self,
        shape_key: Any,
        static_forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        # The same compiled callable serves every shape — torch.compile's
        # internal cache dispatches by tensor shape.
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
