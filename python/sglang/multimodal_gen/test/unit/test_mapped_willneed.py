"""advise_willneed hands the kernel page-aligned ranges and never raises.

What matters: the madvise call receives a page-aligned start and a length
that covers the tensor's storage, odd offsets round outward, and failures
(no libc, bad pointers) degrade to zero advice instead of an exception.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    layerwise_offload as lo,
)


class _RecordingLibc:
    def __init__(self, ret=0):
        self.calls = []
        self.ret = ret

    def madvise(self, addr, length, advice):
        self.calls.append((addr.value, length.value, advice))
        return self.ret


@pytest.fixture()
def libc(monkeypatch):
    fake = _RecordingLibc()
    monkeypatch.setattr(lo, "_libc", fake)
    monkeypatch.setattr(lo, "_willneed_headroom_ok", lambda need: True)
    return fake


def test_ranges_are_page_aligned_and_cover_the_storage(libc):
    t = torch.zeros(1024, dtype=torch.float32)
    advised = lo.advise_willneed([t])

    assert advised == 1
    ((addr, length, advice),) = libc.calls
    page = lo._PAGE
    assert advice == lo._MADV_WILLNEED
    assert addr % page == 0
    ptr = t.untyped_storage().data_ptr()
    nbytes = t.untyped_storage().nbytes()
    assert addr <= ptr
    assert addr + length >= ptr + nbytes
    assert length % page == 0


def test_a_failing_madvise_counts_nothing(monkeypatch):
    monkeypatch.setattr(lo, "_libc", _RecordingLibc(ret=-1))
    assert lo.advise_willneed([torch.zeros(16)]) == 0


def test_no_libc_is_a_quiet_noop(monkeypatch):
    monkeypatch.setattr(lo, "_libc", None)
    assert lo.advise_willneed([torch.zeros(16)]) == 0


def test_empty_and_broken_tensors_are_skipped(libc):
    class Broken:
        def untyped_storage(self):
            raise RuntimeError("no storage")

    assert lo.advise_willneed([Broken(), torch.empty(0)]) == 0
    assert libc.calls == []


def test_no_headroom_withholds_the_advice(monkeypatch):
    fake = _RecordingLibc()
    monkeypatch.setattr(lo, "_libc", fake)
    monkeypatch.setattr(lo, "_willneed_headroom_ok", lambda need: False)
    assert lo.advise_willneed([torch.zeros(1024)]) == 0
    assert fake.calls == []


def test_headroom_reads_memavailable(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"

    real_open = open

    def fake_open(path, *a, **k):
        if path == "/proc/meminfo":
            return real_open(meminfo, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)

    meminfo.write_text("MemTotal: 32 kB\nMemAvailable: 16777216 kB\n")  # 16 GiB
    assert lo._willneed_headroom_ok(1 << 30)

    meminfo.write_text("MemTotal: 32 kB\nMemAvailable: 524288 kB\n")  # 0.5 GiB
    assert not lo._willneed_headroom_ok(1 << 30)
