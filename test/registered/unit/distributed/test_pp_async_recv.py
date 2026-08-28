from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeWork:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


def test_pp_proxy_tensors_waits_for_comm_once():
    work = FakeWork()
    callback = Mock()
    proxy = PPProxyTensors(
        {"hidden_states": torch.arange(4)},
        comm_works=[SimpleNamespace(work=work)],
        comm_postprocess=[callback],
    )

    assert work.wait_count == 0
    assert callback.call_count == 0

    assert torch.equal(proxy["hidden_states"], torch.arange(4))
    assert work.wait_count == 1
    callback.assert_called_once_with()

    # Repeated reads must not wait or postprocess again.
    assert torch.equal(proxy.tensors["hidden_states"], torch.arange(4))
    assert work.wait_count == 1
    callback.assert_called_once_with()


def test_pp_proxy_tensor_slice_waits_before_constructing_view():
    work = FakeWork()
    proxy = PPProxyTensors(
        {"hidden_states": torch.arange(6).reshape(3, 2)},
        comm_works=[SimpleNamespace(work=work)],
    )

    sliced = proxy[1:]

    assert work.wait_count == 1
    assert torch.equal(sliced["hidden_states"], torch.tensor([[2, 3], [4, 5]]))


def test_irecv_tensor_dict_defers_payload_wait():
    coordinator = object.__new__(GroupCoordinator)
    coordinator.world_size = 2
    coordinator.rank_in_group = 1
    coordinator.ranks = [0, 1]
    coordinator.device_group = "device-group"
    coordinator.cpu_group = "cpu-group"
    coordinator.recv_object = Mock(
        return_value=[
            (
                "hidden_states",
                TensorMetadata(torch.device("cpu"), torch.float32, (2, 3)),
            ),
            ("kind", "proxy"),
        ]
    )
    work = FakeWork()

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.irecv", return_value=work) as irecv,
    ):
        tensors, works, postprocess = coordinator.irecv_tensor_dict()

    assert tensors["hidden_states"].shape == (2, 3)
    assert tensors["kind"] == "proxy"
    assert len(works) == 1
    assert works[0].work is work
    assert works[0].payload is tensors["hidden_states"]
    assert postprocess == []
    assert work.wait_count == 0
    irecv.assert_called_once_with(tensors["hidden_states"], src=0, group="cpu-group")
