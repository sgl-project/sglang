from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("CUDA piecewise backend requires CUDA", allow_module_level=True)

import sglang.srt.compilation.cuda_piecewise_backend as cuda_backend
from sglang.srt.compilation.cuda_piecewise_backend import (
    ConcreteSizeEntry,
    CUDAPiecewiseBackend,
)


def test_runtime_recompile_without_capture_stream_falls_back(monkeypatch):
    """A Dynamo replacement backend cannot capture outside a PCG session."""
    compile_config = MagicMock()
    compile_config.get_capture_sizes.return_value = []
    backend = CUDAPiecewiseBackend(
        graph=MagicMock(),
        compile_config=compile_config,
        inductor_config={},
        graph_pool=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        sym_shape_indices=[0],
        compiled_graph_for_general_shape=MagicMock(),
        sglang_backend=MagicMock(),
    )
    backend.first_run_finished = True
    fallback = MagicMock(return_value="fallback-result")
    backend.concrete_size_entries = {
        4: ConcreteSizeEntry(
            runtime_shape=4,
            need_to_compile=False,
            use_cudagraph=True,
            runnable=fallback,
            num_finished_warmup=1,
        )
    }

    monkeypatch.setattr(cuda_backend, "get_pcg_capture_stream", lambda: None)
    monkeypatch.setattr(cuda_backend, "is_in_torch_compile_warmup", lambda: False)
    monkeypatch.setattr(cuda_backend, "print_warning_once", lambda _message: None)

    assert backend(4) == "fallback-result"
    fallback.assert_called_once_with(4)
