import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.softmax_temp import (
    softmax_temp,
    softmax_temp_flashinfer,
    softmax_temp_triton,
)

try:
    from flashinfer.sampling import softmax as _flashinfer_softmax
except ImportError:
    _flashinfer_softmax = None

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8])
@pytest.mark.parametrize("rows_per_request", [1, 6, 7])
@pytest.mark.parametrize("vocab", [1000, 4096, 129280])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_matches_torch_probs(bs, rows_per_request, vocab, dtype):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(bs * 1000 + rows_per_request)
    logits = (
        torch.randn(bs * rows_per_request, vocab, device=device, generator=g) * 8.0
    ).to(dtype)
    temperatures = (torch.rand(bs, device=device, generator=g) * 1.5 + 0.05).to(
        torch.float32
    )

    ref = softmax_temp(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )
    got = softmax_temp_triton(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )

    assert got.dtype == ref.dtype == torch.float32
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        got.sum(dim=-1), torch.ones_like(got.sum(dim=-1)), rtol=1e-5, atol=1e-5
    )


def test_column_temperatures_accepted():
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(7)
    logits = torch.randn(6, 512, device=device, generator=g).to(torch.bfloat16)
    temperatures = (torch.rand(2, 1, device=device, generator=g) + 0.3).to(
        torch.float32
    )
    ref = softmax_temp(logits=logits, temperatures=temperatures, rows_per_request=3)
    got = softmax_temp_triton(
        logits=logits, temperatures=temperatures, rows_per_request=3
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(
    _flashinfer_softmax is None, reason="flashinfer.sampling.softmax unavailable"
)
@pytest.mark.parametrize("rows_per_request", [5, 6, 11])
@pytest.mark.parametrize("vocab", [4096, 129280])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_flashinfer_matches_torch_probs(rows_per_request, vocab, dtype):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(rows_per_request * 100 + vocab)
    logits = (
        torch.randn(rows_per_request, vocab, device=device, generator=g) * 8.0
    ).to(dtype)
    temperatures = (torch.rand(1, device=device, generator=g) * 1.5 + 0.05).to(
        torch.float32
    )

    ref = softmax_temp(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )
    got = softmax_temp_flashinfer(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )

    assert got.dtype == ref.dtype == torch.float32
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(
        got.sum(dim=-1), torch.ones_like(got.sum(dim=-1)), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(
    _flashinfer_softmax is None, reason="flashinfer.sampling.softmax unavailable"
)
def test_impl_timing_smoke():
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    vocab = 129280
    for rows_per_request in (5, 6, 11):
        logits = torch.randn(
            rows_per_request, vocab, device=device, generator=g, dtype=torch.float32
        )
        temperatures = torch.ones(1, device=device, dtype=torch.float32)
        for name, fn in (
            ("torch", softmax_temp),
            ("triton", softmax_temp_triton),
            ("flashinfer", softmax_temp_flashinfer),
        ):
            for _ in range(10):
                fn(
                    logits=logits,
                    temperatures=temperatures,
                    rows_per_request=rows_per_request,
                )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                fn(
                    logits=logits,
                    temperatures=temperatures,
                    rows_per_request=rows_per_request,
                )
            end.record()
            torch.cuda.synchronize()
            per_call_us = start.elapsed_time(end) / 100 * 1e3
            print(
                f"[softmax_temp] rows={rows_per_request} {name}: {per_call_us:.1f} us/call"
            )
