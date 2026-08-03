import pytest
import torch

from sglang.srt.layers.moe.dwdp.hsa_copy import (
    HsaCopyUnavailableError,
    HsaSdmaCopyEngine,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-2-gpu-large-amd")


@pytest.fixture(scope="module")
def copy_engine():
    if torch.version.hip is None or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two ROCm GPUs")
    try:
        return HsaSdmaCopyEngine()
    except HsaCopyUnavailableError as error:
        pytest.skip(str(error))


def test_hsa_copy_cross_gpu_bytes(copy_engine):
    source = torch.arange(1 << 20, dtype=torch.int32, device="cuda:1")
    destination = torch.empty_like(source, device="cuda:0")
    # Establish HIP peer access before bypassing HIP's memcpy path.
    destination.copy_(source)
    torch.cuda.synchronize()
    destination.zero_()
    ticket = copy_engine.submit(destination, source)
    copy_engine.wait(ticket)
    torch.testing.assert_close(destination.cpu(), source.cpu(), rtol=0, atol=0)


def test_hsa_copy_multiple_peer_engines(copy_engine):
    peer_count = min(torch.cuda.device_count() - 1, 3)
    sources = [
        torch.full((1 << 20,), peer, dtype=torch.uint8, device=f"cuda:{peer}")
        for peer in range(1, peer_count + 1)
    ]
    destinations = [torch.empty_like(source, device="cuda:0") for source in sources]
    for destination, source in zip(destinations, sources):
        destination.copy_(source)
        destination.zero_()
    torch.cuda.synchronize()
    tickets = [
        copy_engine.submit(destination, source)
        for destination, source in zip(destinations, sources)
    ]
    copy_engine.wait_all(tickets)
    for peer, destination in enumerate(destinations, start=1):
        assert torch.all(destination == peer)
