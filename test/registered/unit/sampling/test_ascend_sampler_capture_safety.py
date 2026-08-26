# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.srt.layers import sampler


def _sample(
    *,
    logits: torch.Tensor | None = None,
    top_k: int = 2,
) -> torch.Tensor:
    logits = logits if logits is not None else torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    return sampler.top_k_top_p_min_p_sampling_from_logits_ascend(
        logits=logits,
        top_ks=torch.tensor([top_k], dtype=torch.int32),
        top_ps=torch.tensor([0.9], dtype=torch.float32),
        min_ps=torch.tensor([0.0], dtype=torch.float32),
        need_min_p_sampling=False,
        sampling_seed=None,
        positions=torch.tensor([0], dtype=torch.int64),
    )


def test_capture_state_is_false_off_npu(monkeypatch) -> None:
    monkeypatch.setattr(sampler, "is_npu", lambda: False)

    assert sampler._ascend_sampling_stream_capturing() is False


def test_capture_mode_avoids_host_guard_and_fused_kernel(monkeypatch) -> None:
    fused_calls = []

    def _fused(*args, **kwargs):
        fused_calls.append((args, kwargs))
        raise AssertionError("fused kernel must not run during graph capture")

    def _forbidden_all(*args, **kwargs):
        raise AssertionError("torch.all must not run during graph capture")

    monkeypatch.setattr(
        sampler,
        "torch_npu",
        SimpleNamespace(npu_top_k_top_p=_fused),
        raising=False,
    )
    monkeypatch.setattr(
        sampler,
        "_ascend_sampling_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(sampler.torch, "all", _forbidden_all)

    output = _sample()

    assert output.shape == (1,)
    assert fused_calls == []


def test_eager_eligible_top_k_uses_fused_kernel(monkeypatch) -> None:
    fused_calls = []

    def _fused(logits, top_ps, top_ks):
        fused_calls.append((logits, top_ps, top_ks))
        return logits

    monkeypatch.setattr(
        sampler,
        "torch_npu",
        SimpleNamespace(npu_top_k_top_p=_fused),
        raising=False,
    )
    monkeypatch.setattr(
        sampler,
        "_ascend_sampling_stream_capturing",
        lambda: False,
    )

    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16)
    output = _sample(logits=logits, top_k=2)

    assert output.shape == (1,)
    assert len(fused_calls) == 1
    assert fused_calls[0][1].dtype == logits.dtype


def test_eager_out_of_range_top_k_uses_fallback(monkeypatch) -> None:
    fused_calls = []

    def _fused(*args, **kwargs):
        fused_calls.append((args, kwargs))
        raise AssertionError("out-of-range top_k must use the fallback")

    monkeypatch.setattr(
        sampler,
        "torch_npu",
        SimpleNamespace(npu_top_k_top_p=_fused),
        raising=False,
    )
    monkeypatch.setattr(
        sampler,
        "_ascend_sampling_stream_capturing",
        lambda: False,
    )

    output = _sample(top_k=2048)

    assert output.shape == (1,)
    assert fused_calls == []
