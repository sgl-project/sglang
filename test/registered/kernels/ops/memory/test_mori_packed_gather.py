"""Real GPU gather + page-write geometry; fake transport, no RDMA required."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.mori.packed import MoriPackedKVSender
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("capacity_pages", [1, 3])
def test_random_page_gather_and_complete_destination_bytes(page_size, capacity_pages):
    device = torch.cuda.current_device()
    src_pages, dst_pages = [7, 2, 11], [4, 9, 1]
    src_row, dst_row, head_offset = 512, 256, 256
    sources = [
        ((torch.arange(16 * page_size * src_row, device="cuda") + tag) % 251).to(
            torch.uint8
        )
        for tag in (1, 17, 37, 91)
    ]
    destinations = [
        torch.full((16 * page_size * dst_row,), 0xA5, device="cuda", dtype=torch.uint8)
        for _ in sources
    ]
    expected = []
    for source, destination in zip(sources, destinations):
        reference = destination.clone().view(16, page_size, dst_row)
        selected = source.view(16, page_size, src_row)[
            src_pages, :, head_offset : head_offset + dst_row
        ]
        reference[dst_pages] = selected
        expected.append(reference.reshape(-1))
    buffer = StagingBuffer(
        len(sources) * page_size * dst_row * capacity_pages, f"cuda:{device}", device
    )
    calls, deregistrations, recorded = [], [], []
    status = SimpleNamespace(InProgress=lambda: False, Failed=lambda: False)

    class Engine:
        def register_memory(self, *args):
            return "registered-pack-buffer"

        def allocate_transfer_uid(self):
            return len(calls) + 1

        def batch_write(self, src, local, dst, remote, sizes, uids):
            calls.append(len(sizes[0]))
            for offset, destination_offset, size in zip(local[0], remote[0], sizes[0]):
                dst[0][destination_offset : destination_offset + size].copy_(
                    buffer.buffer[offset : offset + size]
                )
            return [status]

        def deregister_memory(self, desc):
            deregistrations.append(desc)

    def complete(statuses):
        torch.cuda.synchronize()
        return None

    sender = MoriPackedKVSender(
        engine=Engine(),
        gpu_id=device,
        size_bytes=buffer.get_size(),
        buffer=buffer,
        memory_location="GPU",
        wait_completion=complete,
        record_statuses=recorded.extend,
    )
    # Source values must be ready before the transport worker's gather.
    torch.cuda.synchronize()
    sender.send(
        src_ptrs=[source.data_ptr() for source in sources],
        dst_descs=destinations,
        src_pages=src_pages,
        dst_pages=dst_pages,
        src_page_bytes=page_size * src_row,
        dst_page_bytes=page_size * dst_row,
        page_size=page_size,
        head_offset_bytes=head_offset,
    )
    for actual, reference in zip(destinations, expected):
        assert torch.equal(actual, reference)  # Includes all untouched sentinel pages.
    assert len(calls) == len(sources) * (3 // capacity_pages)
    assert len(recorded) == len(calls)
    assert sender.close() and sender.close()
    assert deregistrations == ["registered-pack-buffer"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
