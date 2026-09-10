import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _deferred_proxy():
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.total_consumer_count = 1
    proxy.acknowledge_consumption = MagicMock()
    return proxy


def test_release_features_acknowledges_deferred_transport():
    proxy = _deferred_proxy()
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
    mm_inputs = MultimodalInputs(mm_items=[item])

    mm_inputs.release_features()

    proxy.acknowledge_consumption.assert_called_once_with(1)
    assert item.feature is None


def test_release_features_keeps_cleanup_error_request_local():
    proxy = _deferred_proxy()
    proxy.acknowledge_consumption.side_effect = RuntimeError("ack failed")
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
    mm_inputs = MultimodalInputs(mm_items=[item])

    mm_inputs.release_features()

    assert item.feature is None


def test_request_abort_releases_multimodal_features():
    mm_inputs = MagicMock()
    req = object.__new__(Req)
    req.rid = "rejected-vlm-request"
    req.session = None
    req.multimodal_inputs = mm_inputs
    req.grammar = object()
    req.return_logprob = True
    req.logprob_start_len = 0

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(tp_rank=1),
    ):
        req.set_finish_with_abort("invalid multimodal request")

    mm_inputs.release_features.assert_called_once_with()
    assert req.multimodal_inputs is None


def test_session_abort_preserves_shared_multimodal_features():
    mm_inputs = MagicMock()
    req = object.__new__(Req)
    req.rid = "rejected-session-turn"
    req.session = object()
    req.multimodal_inputs = mm_inputs
    req.grammar = object()
    req.return_logprob = True
    req.logprob_start_len = 0

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(tp_rank=1),
    ):
        req.set_finish_with_abort("invalid session turn")

    mm_inputs.release_features.assert_not_called()
    assert req.multimodal_inputs is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
