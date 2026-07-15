"""Regression coverage for a Dynamo-recompiled TC piecewise backend."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import sglang.srt.compilation.cuda_piecewise_backend as piecewise_backend
from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend
from sglang.test.ci.ci_register import register_cuda_ci

# cuda_piecewise_backend imports the CUDA-only weak_ref_tensor helper. Run this
# regression where that backend can actually be imported and exercised.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _CompileConfig:
    @staticmethod
    def get_capture_sizes():
        return [1]

    @staticmethod
    def get_enable_debug_mode():
        return False


def test_runtime_recompile_without_capture_stream_falls_back_to_eager():
    """A replacement Dynamo backend cannot capture outside the runner session."""

    backend = CUDAPiecewiseBackend(
        graph=None,
        compile_config=_CompileConfig(),
        inductor_config={},
        graph_pool=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        sym_shape_indices=[0],
        compiled_graph_for_general_shape=lambda size: f"eager:{size}",
        sglang_backend=SimpleNamespace(),
    )
    backend.first_run_finished = True
    backend.concrete_size_entries[1].num_finished_warmup = 1

    with (
        patch.object(piecewise_backend, "get_pcg_capture_stream", return_value=None),
        patch.object(piecewise_backend, "print_warning_once") as warning,
    ):
        assert backend(1) == "eager:1"

    warning.assert_called_once()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
