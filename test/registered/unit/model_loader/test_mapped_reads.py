import gc
import os
import tempfile

import pytest
import safetensors.torch
import torch

from sglang.srt.model_loader import mapped_reads
from sglang.srt.model_loader.weight_utils import _read_tensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TENSORS = {
    "a.w13": torch.arange(48 * 40, dtype=torch.int16).reshape(48, 40),
    "b.scale": torch.arange(48 * 5, dtype=torch.uint8).reshape(48, 5),
    "c.w2": torch.randn(40, 24, dtype=torch.bfloat16),
    "d.bias": torch.randn(7, dtype=torch.float32),
    "e.scalar": torch.tensor(0.25),
    # Over 16 MiB: whole reads are split across threads.
    "f.big": torch.randint(-(2**15), 2**15, (6000, 2048), dtype=torch.int16),
}


@pytest.fixture(params=[False, True], ids=["no-prefetch", "prefetched"])
def loaded(request):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.safetensors")
        safetensors.torch.save_file(TENSORS, path)
        yield request.param, dict(_read_tensors(path, prefetched=request.param))


def slices(name, tensor, tp=4):
    for rank in range(tp):
        n = tensor.shape[0] // tp if tensor.dim() else 0
        if n:
            yield "rows", tensor.narrow(0, rank * n, n)
        if tensor.dim() == 2:
            m = tensor.shape[1] // tp
            yield "cols", tensor.narrow(1, rank * m, m)
    yield "whole", tensor


def test_slices_read_the_files_bytes(loaded):
    prefetched, tensors = loaded
    for name, tensor in tensors.items():
        for kind, view in slices(name, tensor):
            read = mapped_reads.read_mapped_slice(view)
            if kind == "cols" and prefetched:
                assert read is None, name  # Column shards stay on the mapping.
                continue
            assert read is not None, (name, kind)
            assert read.dtype == view.dtype and read.shape == view.shape
            assert (
                read.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()
            )
            assert torch.equal(read, view), (name, kind)
            assert torch.equal(read, TENSORS[name][tuple(_index(view, tensor))])


def _index(view, base):
    """The index of ``view`` in ``base`` for a narrow along one dim."""
    if view.dim() == 0:
        return ()
    offset = view.data_ptr() - base.data_ptr()
    elem = offset // base.element_size()
    idx = []
    for dim, (n, stride) in enumerate(zip(view.shape, base.stride())):
        start = elem // stride
        elem -= start * stride
        idx.append(slice(start, start + n))
    return idx


def test_uncached_rows_keep_the_mapping(loaded, monkeypatch):
    _, tensors = loaded
    view = tensors["a.w13"].narrow(0, 12, 12)
    assert mapped_reads.read_mapped_slice(view) is not None
    monkeypatch.setattr(mapped_reads, "_cached", lambda address, nbytes: False)
    assert mapped_reads.read_mapped_slice(view) is None
    assert mapped_reads.host_source(view) is view


def test_unregistered_or_modified_tensors_keep_the_copy(loaded):
    _, tensors = loaded
    assert mapped_reads.read_mapped_slice(torch.ones(4, 4)) is None
    w13 = tensors["a.w13"]
    w13[0, 0] += 1  # Written in place: the mapping no longer matches the file.
    assert mapped_reads.read_mapped_slice(w13.narrow(0, 12, 12)) is None
    # Views of a tensor that is gone are not read either.
    view = tensors.pop("b.scale").narrow(0, 0, 12)
    gc.collect()
    assert mapped_reads.read_mapped_slice(view) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
