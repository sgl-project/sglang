"""GroupCoordinator's aiter registered-input paths on a stand-in communicator (CPU).

A producer writes aiter's registered input pool (ca_input_buffer) and the next eager
all-reduce / reduce-scatter reads it in place (registered_input=True / registered=True)
instead of copying it there; every other input keeps the copy-in, and a registered read
after an intervening copy-in raises.

The 8-rank byte equality is in
test/registered/kernels/ops/communication/test_aiter_registered_input.py.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ps = pytest.importorskip("sglang.srt.distributed.parallel_state")


class _Buf:
    def __init__(self, size):
        self._buffer = torch.zeros(size, dtype=torch.uint8)
        self.data_ptr = self._buffer.data_ptr()
        self.max_size = size
        self.uncached = False


class _CA:
    disabled = False
    _IS_CAPTURING = False

    def __init__(self, size=1 << 16, car_bytes=1 << 15):
        self._pool = {"input": _Buf(size)}
        self.car_bytes = car_bytes
        self.calls = []

    def should_custom_ar(self, inp, prefill_support=False):
        return inp.numel() * inp.element_size() <= self.car_bytes

    def should_custom_ag(self, inp):
        return True

    def all_reduce(
        self,
        inp,
        *,
        out=None,
        use_new=True,
        open_fp8_quant=False,
        registered_input=False,
    ):
        self.calls.append(("all_reduce", registered_input))
        return inp.clone()

    def custom_all_reduce(self, inp, use_new=True, open_fp8_quant=False):
        self.calls.append(("custom_all_reduce",))
        return inp.clone()

    def reduce_scatter(self, inp, out, dim=0, *, registered=False):
        self.calls.append(("reduce_scatter", registered))

    def all_gather_reg(self, inp, out=None, dim=0):
        self.calls.append(("all_gather_reg", inp.dtype))

    def all_gather_unreg(self, inp, out=None, dim=0):
        self.calls.append(("all_gather_unreg", inp.dtype))


@pytest.fixture
def group(monkeypatch):
    capturing = {"on": False}
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing["on"]
    )
    monkeypatch.setattr(ps, "is_hip", lambda: True)
    monkeypatch.setattr(ps, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(
        ps.GroupCoordinator,
        "_deterministic_collectives_enabled",
        staticmethod(lambda: False),
    )
    monkeypatch.setenv("SGLANG_USE_AITER_AG", "1")
    monkeypatch.setenv("SGLANG_DP_USE_REDUCE_SCATTER", "1")
    monkeypatch.delenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("SGLANG_AITER_CAPTURE_COPY_IN", raising=False)
    g = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
    g.world_size = 8
    g.unique_name = "tp"
    g.device = torch.device("cpu")
    g.ca_comm = _CA()
    g.qr_comm = g.pymscclpp_comm = g.torch_symm_mem_comm = g.pcie_ipc_comm = None
    g.pynccl_comm = None
    return g, capturing


def test_buffer_is_a_view_of_the_pool_base(group):
    g, _ = group
    t = g.ca_input_buffer((64, 128), torch.bfloat16)
    assert t.shape == (64, 128) and t.dtype == torch.bfloat16 and t.is_contiguous()
    assert t.data_ptr() == g.ca_comm._pool["input"].data_ptr
    assert g._in_ca_pool(t)
    assert not g._in_ca_pool(t[1:])
    assert not g._in_ca_pool(torch.empty(64, 128, dtype=torch.bfloat16))


def test_buffer_refusals(group, monkeypatch):
    g, capturing = group
    # Over aiter's eager cutoff the all-reduce would not take the custom all-reduce.
    assert g.ca_input_buffer((129, 128), torch.bfloat16) is None
    assert g.ca_input_buffer((3, 3), torch.bfloat16) is None  # not a multiple of 16 B
    assert g.ca_input_buffer((0, 128), torch.bfloat16) is None
    capturing["on"] = True
    assert g.ca_input_buffer((64, 128), torch.bfloat16) is None
    capturing["on"] = False
    g.ca_comm._IS_CAPTURING = True
    assert g.ca_input_buffer((64, 128), torch.bfloat16) is None
    g.ca_comm._IS_CAPTURING = False
    g.ca_comm.disabled = True
    assert g.ca_input_buffer((64, 128), torch.bfloat16) is None
    g.ca_comm.disabled = False
    g.world_size = 1
    assert g.ca_input_buffer((64, 128), torch.bfloat16) is None
    g.world_size = 8
    monkeypatch.setattr(ps, "is_hip", lambda: False)
    assert g.ca_input_buffer((64, 128), torch.bfloat16) is None


def test_uncached_or_foreign_pool_keeps_the_copy_in(group):
    g, _ = group
    g.ca_comm._pool["input"].uncached = True
    assert g.ca_input_buffer((64, 128), torch.bfloat16) is None
    other = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
    other.world_size, other.device = 8, torch.device("cpu")
    other.ca_comm = type("Other", (), {"disabled": False, "_IS_CAPTURING": False})()
    assert other.ca_input_buffer((64, 128), torch.bfloat16) is None


def test_all_reduce_dispatch(group):
    g, capturing = group
    ca = g.ca_comm
    t = g.ca_input_buffer((64, 128), torch.bfloat16)
    g._all_reduce_out_place(t, "ca")
    assert ca.calls[-1] == ("all_reduce", True)
    g._all_reduce_out_place(torch.ones(64, 128, dtype=torch.bfloat16), "ca")
    assert ca.calls[-1] == ("custom_all_reduce",)
    capturing["on"] = True
    g._all_reduce_out_place(t, "ca")
    assert ca.calls[-1] == ("custom_all_reduce",)


def test_reduce_scatter_dispatch(group):
    g, _ = group
    ca = g.ca_comm
    t = g.ca_input_buffer((64, 128), torch.bfloat16)
    out = torch.empty(8, 128, dtype=torch.bfloat16)
    assert g._maybe_aiter_reduce_scatter(out, t)
    assert ca.calls[-1] == ("reduce_scatter", True)
    assert g._maybe_aiter_reduce_scatter(out, torch.ones(64, 128, dtype=torch.bfloat16))
    assert ca.calls[-1] == ("reduce_scatter", False)


@pytest.mark.parametrize("copy_in", ["all_reduce", "reduce_scatter", "all_gather"])
@pytest.mark.parametrize("consumer", ["all_reduce", "reduce_scatter"])
def test_registered_read_after_a_copy_in_raises(group, copy_in, consumer):
    g, _ = group
    t = g.ca_input_buffer((64, 128), torch.bfloat16)
    other = torch.ones(64, 128, dtype=torch.bfloat16)
    if copy_in == "all_reduce":
        g._all_reduce_out_place(other, "ca")
    elif copy_in == "reduce_scatter":
        g._maybe_aiter_reduce_scatter(torch.empty(8, 128, dtype=torch.bfloat16), other)
    else:
        g._all_gather_into_tensor(torch.empty(512, 128, dtype=torch.bfloat16), other)
    with pytest.raises(RuntimeError, match="copied its input over the registered pool"):
        if consumer == "all_reduce":
            g._all_reduce_out_place(t, "ca")
        else:
            g._maybe_aiter_reduce_scatter(torch.empty(8, 128, dtype=torch.bfloat16), t)


def test_new_buffer_after_a_copy_in_and_capture_copy_ins_are_fine(group):
    g, _ = group
    ca = g.ca_comm
    g.ca_input_buffer((64, 128), torch.bfloat16)  # abandoned before its collective
    g._all_reduce_out_place(torch.ones(64, 128, dtype=torch.bfloat16), "ca")
    t = g.ca_input_buffer((64, 128), torch.bfloat16)
    g._all_reduce_out_place(t, "ca")
    assert ca.calls[-1] == ("all_reduce", True)
    t = g.ca_input_buffer((64, 128), torch.bfloat16)
    ca._IS_CAPTURING = True  # copy-ins inside the capture context use the graph buffers
    g._all_reduce_out_place(torch.ones(64, 128, dtype=torch.bfloat16), "ca")
    ca._IS_CAPTURING = False
    g._all_reduce_out_place(t, "ca")
    assert ca.calls[-1] == ("all_reduce", True)
