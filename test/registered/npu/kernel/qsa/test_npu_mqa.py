"""Model-scoped MQA adapters: canonical package, errors and graph replay.

Legacy generic inputs are now rejection cases, not a reference fallback.
The kernel package carries independent FP64 and larger-shape regressions.
"""
import pytest
import torch
from sgl_kernel_npu.qwen3_8_flash_next import mqa as npu_mqa

from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


def make_inputs(kind):
    torch.manual_seed(73)
    q = torch.randn(8, 4, 128, device="npu", dtype=torch.bfloat16)
    lengths = torch.tensor([0, 1, 15, 16, 17, 31, 128, 144], device="npu", dtype=torch.int32)
    if kind == "packed":
        k = torch.randn(160, 1, 128, device="npu", dtype=q.dtype)
        starts = torch.arange(8, device="npu", dtype=torch.int32)
        return q, k, starts, starts + lengths
    k = torch.randn(10, 16, 1, 128, device="npu", dtype=q.dtype)
    table = torch.arange(9, 0, -1, device="npu", dtype=torch.int32).repeat(8, 1)
    return q, k, table, lengths, 144


def functions(kind):
    if kind == "packed":
        return mqa.qsa_mqa_prefill, mqa.torch_qsa_mqa_prefill
    return mqa.qsa_mqa_decode, mqa.torch_qsa_mqa_decode


@pytest.mark.parametrize("kind", ["packed", "paged"])
def test_mqa_dispatch_and_graph_updates(kind, monkeypatch):
    fn, reference = functions(kind)
    args = make_inputs(kind)
    calls = []
    raw = getattr(npu_mqa, kind)

    def traced(*inputs):
        calls.append(kind)
        return raw(*inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("NPU MQA must not call Torch reference or TileLang")

    monkeypatch.setattr(npu_mqa, kind, traced)
    for name in ("torch_qsa_mqa_prefill", "torch_qsa_mqa_decode",
                 "tilelang_qsa_mqa_prefill", "tilelang_qsa_mqa_decode"):
        monkeypatch.setattr(mqa, name, forbidden)

    def check(out):
        expected = reference(*args)
        assert out.dtype == torch.float32 and out.is_contiguous()
        assert torch.equal(torch.isneginf(out), torch.isneginf(expected))
        torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-5)

    for _ in range(2):
        check(fn(*args))
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fn(*args)
    call_count = len(calls)
    originals = [x.clone() for x in args[:4]]
    addresses = [x.data_ptr() for x in args[:4]]
    for change in (None, 0, 1, 2, 3, "restore"):
        for x, saved in zip(args, originals):
            x.copy_(saved)
        if change in (0, 1):
            args[change].neg_()
        elif change == 2:
            if kind == "packed":
                args[2].copy_(args[3])
            else:
                args[2].copy_(args[2].roll(1, dims=1))
        elif change == 3:
            if kind == "packed":
                args[3].copy_(args[2])
            else:
                args[3].zero_()
        saved_inputs = [x.clone() for x in args[:4]]
        graph.replay()
        torch.npu.synchronize()
        check(out)
        assert len(calls) == call_count  # Replay does not execute Python routing.
        assert addresses == [x.data_ptr() for x in args[:4]]
        for x, saved in zip(args, saved_inputs):
            torch.testing.assert_close(x, saved, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["packed", "paged"])
@pytest.mark.parametrize("case", ["fp16", "fp32", "heads", "dimension", "stride", "int64", "cpu", "shape", "storage"])
def test_mqa_metadata_errors_do_not_fall_back(kind, case, monkeypatch):
    fn, _ = functions(kind)
    args = list(make_inputs(kind))
    if case in ("fp16", "fp32"):
        dtype = torch.float16 if case == "fp16" else torch.float32
        args[0], args[1] = args[0].to(dtype), args[1].to(dtype)
    elif case == "heads":
        args[0] = args[0][:, :3].contiguous()
    elif case == "dimension":
        args[0], args[1] = args[0][..., :64].contiguous(), args[1][..., :64].contiguous()
    elif case == "stride":
        args[0] = args[0].transpose(0, 1).contiguous().transpose(0, 1)
    elif case == "int64":
        args[2] = args[2].long()
    elif case == "cpu":
        args[2] = args[2].cpu()
    elif case == "shape":
        args[3] = args[3][:-1]
    elif kind == "paged":
        args[1] = args[1][:, :8].contiguous()
    else:
        args[1] = args[1].expand(-1, 2, -1).contiguous()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported metadata must not use the reference")

    monkeypatch.setattr(mqa, "torch_qsa_mqa_prefill", forbidden)
    monkeypatch.setattr(mqa, "torch_qsa_mqa_decode", forbidden)
    with pytest.raises(ValueError):
        fn(*args)


@pytest.mark.parametrize("kind", ["packed", "paged"])
@pytest.mark.parametrize("scale", [0, 3.0, 128**0.5])
def test_mqa_custom_scale_rejected(kind, scale):
    with pytest.raises(ValueError, match="Custom MQA scale"):
        functions(kind)[0](*make_inputs(kind), score_scale=scale)


@pytest.mark.parametrize("kind", ["packed", "paged"])
def test_mqa_kernel_errors_propagate(kind, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional MQA failure")

    monkeypatch.setattr(npu_mqa, kind, fail)
    with pytest.raises(RuntimeError, match="intentional MQA failure"):
        functions(kind)[0](*make_inputs(kind))


def test_mqa_independent_output_width_rejected():
    args = list(make_inputs("paged"))
    args[-1] -= 1
    with pytest.raises(ValueError, match="Output width"):
        mqa.qsa_mqa_decode(*args)
