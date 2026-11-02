"""The side-stream context projection of the DFLASH verify step.

Exercises _start_ctx_projection / _take_ctx_projection against a stub worker: the
projection itself is the draft model's, so what is worth pinning here is the handover
-- that the result is the projection of the tensor that was handed in, that a
mismatched tensor still joins the side stream, and that capture and failure both leave
the inline path.
"""

import types
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _worker(project, *, enabled=True):
    """A stub carrying only what the two methods touch."""
    w = types.SimpleNamespace(
        device=torch.device("cuda"),
        _overlap_ctx_projection=enabled,
        _ctx_proj_stream=None,
        _ctx_proj_pending=None,
        draft_model=types.SimpleNamespace(project_target_hidden=project),
        draft_parallel_layout=None,
    )
    w._start_ctx_projection = types.MethodType(DFlashWorkerV2._start_ctx_projection, w)
    w._take_ctx_projection = types.MethodType(DFlashWorkerV2._take_ctx_projection, w)
    return w


def _noop_tp_context(_layout):
    import contextlib

    return contextlib.nullcontext()


@patch("sglang.srt.speculative.dflash_worker_v2.draft_tp_context", new=_noop_tp_context)
def test_handover_returns_the_projection_of_that_tensor():
    hidden = torch.randn(64, 128, device="cuda")
    weight = torch.randn(96, 128, device="cuda")
    w = _worker(lambda h: h @ weight.t())
    w._start_ctx_projection(hidden)
    assert w._ctx_proj_pending is not None
    got = w._take_ctx_projection(hidden)
    torch.cuda.synchronize()
    assert got is not None
    torch.testing.assert_close(got, hidden @ weight.t(), rtol=0, atol=0)
    assert w._ctx_proj_pending is None


@patch("sglang.srt.speculative.dflash_worker_v2.draft_tp_context", new=_noop_tp_context)
def test_a_different_tensor_gets_none_but_still_joins():
    """The join must happen even on a miss: the side stream is reading a buffer the
    next verify replay is free to overwrite."""
    hidden = torch.randn(64, 128, device="cuda")
    other = torch.randn(64, 128, device="cuda")
    weight = torch.randn(96, 128, device="cuda")
    w = _worker(lambda h: h @ weight.t())
    w._start_ctx_projection(hidden)
    assert w._take_ctx_projection(other) is None
    assert w._ctx_proj_pending is None
    # Nothing is left outstanding on the side stream.
    assert w._ctx_proj_stream.query()


@patch("sglang.srt.speculative.dflash_worker_v2.draft_tp_context", new=_noop_tp_context)
def test_disabled_empty_and_none_leave_nothing_pending():
    weight = torch.randn(96, 128, device="cuda")
    project = lambda h: h @ weight.t()
    for w, arg in (
        (_worker(project, enabled=False), torch.randn(64, 128, device="cuda")),
        (_worker(project), None),
        (_worker(project), torch.empty(0, 128, device="cuda")),
    ):
        w._start_ctx_projection(arg)
        assert w._ctx_proj_pending is None
        assert w._take_ctx_projection(arg) is None


@patch("sglang.srt.speculative.dflash_worker_v2.draft_tp_context", new=_noop_tp_context)
def test_a_failing_projection_disables_the_overlap_for_good():
    def boom(_h):
        raise RuntimeError("no")

    w = _worker(boom)
    w._start_ctx_projection(torch.randn(64, 128, device="cuda"))
    assert w._ctx_proj_pending is None
    assert w._overlap_ctx_projection is False
