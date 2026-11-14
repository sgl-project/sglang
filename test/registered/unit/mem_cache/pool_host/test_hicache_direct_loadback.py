"""Real page-first H2D copies, including reuse across pools, layers and streams."""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from sgl_kernel.kvcacheio import transfer_kv_per_layer_direct_pf_lf

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

# Unlike test_kvcacheio.py, these tests do not call the unrelated transfer
# kernels that currently make that module skip all CUDA 13 coverage.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_hip(), reason="CUDA direct H2D is required"
)

_SENTINEL = 17
_MAX_RETAINED_COPIES = 65536


def _token_indices(pages, page_size):
    pages = torch.tensor(pages, dtype=torch.int64)
    offsets = torch.arange(page_size, dtype=torch.int64)
    return (pages[:, None] * page_size + offsets).flatten()


def _assert_load(
    host_pages,
    device_pages,
    *,
    dtype=torch.uint8,
    page_size=1,
    num_layers=1,
    start_layer=1,
    is_mla=True,
    storage_offset=0,
    salt=0,
    stream=None,
):
    """Check all destination bytes, including unselected rows and view guards."""
    assert len(host_pages) == len(device_pages)
    host_capacity = max(host_pages, default=0) + 2
    device_capacity = max(device_pages, default=0) + 2
    host_layers = start_layer + num_layers + 1
    width = 7
    shape = (host_capacity, host_layers, page_size, width)
    host_numel = host_capacity * host_layers * page_size * width
    components = 1 if is_mla else 2
    host_buffers = []
    device_buffers = []
    references = []
    device_storages = []
    src_indices = _token_indices(host_pages, page_size)
    dst_indices = _token_indices(device_pages, page_size)
    selected_pages = torch.tensor(host_pages, dtype=torch.int64)

    for component in range(components):
        host_storage = torch.full(
            (host_numel + storage_offset,), _SENTINEL, dtype=dtype, pin_memory=True
        )
        host = host_storage[storage_offset:].view(shape)
        # Small exactly representable values make the raw-byte oracle valid
        # for uint8, FP16 and BF16 without depending on numeric tolerances.
        values = (torch.arange(host_numel) + salt + component * 31) % 251
        host.copy_(values.to(dtype).view(shape))
        host_buffers.append(host)
        for layer in range(num_layers):
            reference = torch.full(
                (device_capacity * page_size * width + storage_offset,),
                _SENTINEL,
                dtype=dtype,
            )
            device_storage = reference.to("cuda")
            device_buffers.append(device_storage[storage_offset:].view(-1, width))
            reference[storage_offset:].view(-1, width)[dst_indices] = host[
                selected_pages, start_layer + layer
            ].reshape(-1, width)
            references.append(reference)
            device_storages.append(device_storage)

    stream = stream if stream is not None else torch.cuda.Stream()
    # Destination initialization is produced on the caller's current stream.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=host_buffers,
            dst_ptrs=device_buffers,
            src_indices=src_indices,
            dst_indices=dst_indices,
            layer_id=start_layer,
            page_size=page_size,
        )
    stream.synchronize()
    for actual, expected in zip(device_storages, references):
        assert torch.equal(actual.cpu().view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [1, 16, 256])
@pytest.mark.parametrize("is_mla", [True, False])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize(
    "host_pages,device_pages", [([1, 2, 3], [2, 3, 4]), ([4, 0, 2], [1, 5, 3])]
)
def test_direct_loadback_layouts(
    dtype, page_size, is_mla, num_layers, host_pages, device_pages
):
    _assert_load(
        host_pages,
        device_pages,
        dtype=dtype,
        page_size=page_size,
        is_mla=is_mla,
        num_layers=num_layers,
        start_layer=2,
        storage_offset=5,
    )


def test_direct_loadback_repeated_calls():
    # A single host thread reuses the workspace while every call replaces its
    # indices, tensor addresses, shape, layer, dtype and K/V component count.
    stream = torch.cuda.Stream()
    cases = [
        (4096, torch.uint8, True, 1, 1, 1),
        (2, torch.float16, False, 3, 16, 2),
        (0, torch.bfloat16, True, 1, 256, 0),
        (2048, torch.bfloat16, False, 2, 1, 3),
        (4, torch.uint8, True, 3, 256, 1),
    ]
    for salt, (pages, dtype, is_mla, layers, page_size, start_layer) in enumerate(
        cases
    ):
        _assert_load(
            list(range(pages - 1, -1, -1)),
            list(range(pages)),
            dtype=dtype,
            page_size=page_size,
            is_mla=is_mla,
            num_layers=layers,
            start_layer=start_layer,
            storage_offset=5 + salt,
            salt=37 * salt,
            stream=stream,
        )


@pytest.mark.parametrize("is_mla", [True, False])
def test_direct_loadback_workspace_limit(is_mla):
    copies_per_page = 1 if is_mla else 2
    stream = torch.cuda.Stream()
    for copies in (_MAX_RETAINED_COPIES, _MAX_RETAINED_COPIES + copies_per_page):
        pages = copies // copies_per_page
        _assert_load(
            list(range(pages - 1, -1, -1)),
            list(range(pages)),
            is_mla=is_mla,
            stream=stream,
        )
    # An oversized call must not leave descriptors in the cached workspace.
    _assert_load([3, 0], [0, 2], is_mla=is_mla, salt=91, stream=stream)


def test_direct_loadback_error_then_valid_call():
    stream = torch.cuda.Stream()
    _assert_load([2, 0], [0, 3], stream=stream)
    host = torch.empty((4, 2, 1, 7), dtype=torch.uint8, pin_memory=True)
    device = torch.empty((4, 7), dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(stream), pytest.raises(RuntimeError, match="same length"):
        transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=[host],
            dst_ptrs=[device],
            src_indices=torch.tensor([0, 1], dtype=torch.int64),
            dst_indices=torch.tensor([0], dtype=torch.int64),
            layer_id=1,
            page_size=1,
        )
    _assert_load(
        [1, 3],
        [2, 0],
        dtype=torch.bfloat16,
        is_mla=False,
        num_layers=2,
        start_layer=2,
        salt=57,
        stream=stream,
    )


def test_direct_loadback_thread_local_workspaces():
    barrier = Barrier(2)
    device_index = torch.cuda.current_device()

    def load(thread):
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream()
            barrier.wait(timeout=30)
            for iteration in range(3):
                pages = 512 + thread * 3 + iteration
                _assert_load(
                    list(range(pages - 1, -1, -1)),
                    list(range(pages)),
                    dtype=torch.float16 if thread else torch.uint8,
                    is_mla=bool(thread),
                    num_layers=2,
                    salt=thread * 43 + iteration,
                    stream=stream,
                )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(load, thread) for thread in range(2)]
        for future in futures:
            future.result(timeout=120)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
