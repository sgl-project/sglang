"""CPU unit tests for the KDA dispatcher's safe-gate reroute.

Safe-gate models (``lower_bound`` set; Kimi-K3 uses ``-5``) must keep serving
when the selected decode / verify kernel does not implement the safe gate: the
dispatcher reroutes that call to the Triton kernel (the reference the KDA
safe-gate tests assert against) and logs the reroute once per kernel and mode,
instead of raising ``NotImplementedError`` on the first decode step and taking
the server down (observed with ``--linear-attn-decode-backend flashinfer``).
"""

from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.linear import kda_backend as kb
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a-test-cpu")


class _UnboundedGateKernel:
    """Stands in for a kernel that only serves ``lower_bound=None``."""

    def __init__(self):
        self.decode = mock.Mock(return_value="other-decode")
        self.target_verify = mock.Mock(return_value="other-verify")


class _SafeGateKernel(_UnboundedGateKernel):
    supports_safe_gate = True


def _dispatcher(decode_kernel, verify_kernel=None):
    d = kb.KDAKernelDispatcher.__new__(kb.KDAKernelDispatcher)
    d.triton_kernel = TritonKDAKernel()
    d.decode_kernel = decode_kernel
    d.verify_kernel = verify_kernel if verify_kernel is not None else decode_kernel
    return d


def _decode_kwargs():
    t = torch.zeros(1)
    return dict(
        A_log=t,
        dt_bias=t,
        ssm_states=t,
        cache_indices=t,
        query_start_loc=t,
    )


@pytest.fixture(autouse=True)
def _clear_log_cache():
    kb._safe_gate_reroute_logged.clear()
    yield
    kb._safe_gate_reroute_logged.clear()


def test_decode_without_safe_gate_keeps_selected_kernel():
    other = _UnboundedGateKernel()
    d = _dispatcher(other)
    assert d.effective_decode_kernel(None) is other
    with mock.patch.object(d.triton_kernel, "decode") as triton_decode:
        out = d.decode(*([torch.zeros(1)] * 5), lower_bound=None, **_decode_kwargs())
    assert out == "other-decode"
    triton_decode.assert_not_called()
    assert other.decode.call_args.kwargs["lower_bound"] is None


def test_decode_with_safe_gate_reroutes_to_triton_and_logs_once(caplog):
    other = _UnboundedGateKernel()
    d = _dispatcher(other)
    assert d.effective_decode_kernel(-5.0) is d.triton_kernel
    with (
        mock.patch.object(
            d.triton_kernel, "decode", return_value="triton-decode"
        ) as triton_decode,
        caplog.at_level("WARNING", logger=kb.logger.name),
    ):
        for _ in range(3):
            out = d.decode(
                *([torch.zeros(1)] * 5), lower_bound=-5.0, **_decode_kwargs()
            )
    assert out == "triton-decode"
    assert triton_decode.call_count == 3
    assert triton_decode.call_args.kwargs["lower_bound"] == -5.0
    other.decode.assert_not_called()
    notices = [
        r for r in caplog.records if "does not support the safe gate" in r.message
    ]
    assert len(notices) == 1
    assert "_UnboundedGateKernel" in notices[0].message
    assert "decode" in notices[0].message


def test_kernel_declaring_safe_gate_support_is_not_rerouted():
    safe = _SafeGateKernel()
    d = _dispatcher(safe)
    assert d.effective_decode_kernel(-5.0) is safe
    with mock.patch.object(d.triton_kernel, "decode") as triton_decode:
        out = d.decode(*([torch.zeros(1)] * 5), lower_bound=-5.0, **_decode_kwargs())
    assert out == "other-decode"
    triton_decode.assert_not_called()


def test_triton_decode_kernel_is_never_rerouted():
    d = _dispatcher(None)
    d.decode_kernel = d.triton_kernel
    assert d.effective_decode_kernel(-5.0) is d.triton_kernel
    assert not kb._safe_gate_reroute_logged


def test_target_verify_with_safe_gate_reroutes_to_triton():
    other = _UnboundedGateKernel()
    d = _dispatcher(_SafeGateKernel(), verify_kernel=other)
    t = torch.zeros(1)
    kwargs = dict(
        A_log=t,
        dt_bias=t,
        q=t,
        k=t,
        v=t,
        a=t,
        b=t,
        ssm_states=t,
        cache_indices=t,
        query_start_loc=t,
        intermediate_states_buffer=t,
        intermediate_state_indices=t,
        cache_steps=2,
        retrieve_parent_token=None,
    )
    with mock.patch.object(
        d.triton_kernel, "target_verify", return_value="triton-verify"
    ) as triton_verify:
        assert d.target_verify(lower_bound=None, **kwargs) == "other-verify"
        triton_verify.assert_not_called()
        assert d.target_verify(lower_bound=-5.0, **kwargs) == "triton-verify"
        assert triton_verify.call_args.kwargs["lower_bound"] == -5.0
    assert ("_UnboundedGateKernel", "target_verify") in kb._safe_gate_reroute_logged
