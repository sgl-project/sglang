import os
import subprocess
import sys

import pytest
import torch

from sglang.srt.utils.async_probe import maybe_detect_nan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=210, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_maybe_detect_nan_leaves_valid_inputs_unchanged(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_NAN_LOGITS_CHECK", "1")
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "0")
    for count in (0, 1, 1023, 1024, 1025, 4097):
        values = torch.randn(count, dtype=dtype, device="cuda")
        if count:
            values[0] = float("inf")
            values[-1] = -float("inf")
        expected = values.clone()
        maybe_detect_nan(values, "unexpected NaN")
        torch.cuda.synchronize()
        assert torch.equal(values, expected)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "mode",
    ["eager", "cuda_graph", "compiled", "strided", "strided_graph", "transposed"],
)
def test_maybe_detect_nan_rejects_changed_input(dtype: str, mode: str) -> None:
    # A device assertion invalidates the CUDA context, so isolate each failure.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "runpy.run_path(sys.argv[1])['_inject_nan'](*sys.argv[2:])",
            __file__,
            dtype,
            mode,
        ],
        env={
            **os.environ,
            "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0",
            "TRITON_DEBUG": "0",
            "SGLANG_ENABLE_NAN_LOGITS_CHECK": "1",
            "SGLANG_ENABLE_ASYNC_ASSERT": "0",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert "healthy check passed" in result.stdout, output
    assert result.returncode != 0, "NaN was accepted"
    assert "test NaN failure" in output
    assert "device-side assert" in output


@pytest.mark.parametrize(
    "nan_check,async_assert",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_maybe_detect_nan_flags(
    nan_check: bool, async_assert: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_NAN_LOGITS_CHECK", str(int(nan_check)))
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", str(int(async_assert)))
    for values in (
        None,
        torch.empty(0),
        torch.tensor([0.0, float("inf"), -float("inf")]),
    ):
        maybe_detect_nan(values, "healthy input")
    values = torch.tensor([float("nan")])
    if nan_check or async_assert:
        with pytest.raises(RuntimeError, match="NaN detected! flag check"):
            maybe_detect_nan(values, "flag check")
    else:
        maybe_detect_nan(values, "flag check")


@pytest.mark.parametrize("column_stride", [1, 2])
@pytest.mark.parametrize("transpose", [False, True])
def test_maybe_detect_nan_strided_logits(
    column_stride: int, transpose: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_NAN_LOGITS_CHECK", "1")
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "0")
    storage = torch.full((4, 4098 * column_stride), float("nan"), device="cuda")
    logits = storage[1::2, 1 : 1 + 4097 * column_stride : column_stride]
    logits.copy_(torch.randn_like(logits))
    if transpose:
        logits = logits.T
    assert not logits.is_contiguous()
    expected = logits.clone()
    maybe_detect_nan(logits, "NaNs outside the view must be ignored")
    torch.cuda.synchronize()
    assert torch.equal(logits, expected)


def test_maybe_detect_nan_disabled_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_NAN_LOGITS_CHECK", "0")
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "0")
    values = torch.full((1025,), float("nan"), device="cuda")
    maybe_detect_nan(values, "disabled")
    torch.cuda.synchronize()
    assert torch.isnan(values).all()


def test_maybe_detect_nan_tall_strided_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_NAN_LOGITS_CHECK", "1")
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "0")
    storage = torch.full((65537, 3), float("nan"), device="cuda")
    logits = storage[:, 1:]
    logits.fill_(1)
    for values in (logits, logits[:1].expand_as(logits), logits[:0]):
        maybe_detect_nan(values, "healthy strided logits")
    torch.cuda.synchronize()
    assert torch.equal(logits, torch.ones_like(logits))


def _inject_nan(dtype: str, mode: str) -> None:
    if mode in ("strided", "strided_graph", "transposed"):
        x = torch.zeros((2, 8196), dtype=getattr(torch, dtype), device="cuda")[
            :, 1:8195:2
        ]
        if mode == "transposed":
            x = x.T
    else:
        x = torch.zeros(4097, dtype=getattr(torch, dtype), device="cuda")

    check = maybe_detect_nan
    if mode == "compiled":
        check = torch.compile(check, fullgraph=True, dynamic=True)
    check(x, "test NaN failure")
    torch.cuda.synchronize()
    print("healthy check passed", flush=True)

    if mode in ("cuda_graph", "strided_graph"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            check(x, "test NaN failure")
        x[..., -1] = float("nan")
        graph.replay()
    else:
        x[..., -1] = float("nan")
        check(x, "test NaN failure")
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
