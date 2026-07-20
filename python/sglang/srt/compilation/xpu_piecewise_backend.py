from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch

from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.compilation.cuda_piecewise_backend import (
    CUDAPiecewiseBackend,
    weak_ref_tensors,
)
from sglang.srt.utils.common import print_warning_once


class XPUPiecewiseBackend(CUDAPiecewiseBackend):
    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            entry.runnable = self.sglang_backend.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )

            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if is_in_torch_compile_warmup():
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  # noqa
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            # During normal capture (PiecewiseCudaGraphRunner.capture()),
            # set_pcg_capture_stream() guarantees a valid stream. However,
            # Dynamo may silently recompile on serving batches whose token
            # count exceeds the captured range (e.g. chunked prefill running
            # at 8192 tokens when the capture grid tops out at 512). The
            # recompiled backend instance has no capture stream; fall back to
            # eager for that sub-graph instead of crashing the scheduler.
            # Mirrors the CUDA fallback in CUDAPiecewiseBackend.__call__.
            stream = get_pcg_capture_stream()
            if stream is None:
                print_warning_once(
                    "PCG capture stream is not set; likely a Dynamo runtime "
                    "recompilation. Falling back to eager execution for this "
                    "subgraph."
                )
                return entry.runnable(*args)

            if self.compile_config.get_enable_debug_mode():
                entry.input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]

            xpugraph = torch.xpu.XPUGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.xpu.empty_cache", lambda: None))

                with torch.xpu.graph(
                    xpu_graph=xpugraph, pool=self.graph_pool, stream=stream
                ):
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        output = weak_ref_tensors(output)

            entry.output = weak_ref_tensors(output)
            entry.cudagraph = xpugraph

            compilation_counter.num_cudagraph_captured += 1
            return output

        if self.compile_config.get_enable_debug_mode():
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )
        entry.cudagraph.replay()
        return entry.output
