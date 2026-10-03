import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _import_flashinfer_backend():
    try:
        from sglang.srt.layers.attention import flashinfer_backend
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )
    except Exception as exc:
        pytest.skip(f"FlashInfer backend is unavailable: {exc}")
    return FlashInferAttnBackend, flashinfer_backend


def _make_backend(*, is_draft_runner: bool):
    FlashInferAttnBackend, _ = _import_flashinfer_backend()
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.max_context_len = 128
    backend.num_wrappers = 1
    backend.skip_prefill = False
    backend.is_draft_runner = is_draft_runner
    backend.cuda_graph_custom_mask = None
    backend.use_sliding_window_kv_pool = False
    backend.kv_indptr = [torch.zeros((5,), dtype=torch.int32)]
    return backend


def test_flashinfer_draft_cuda_graph_keeps_indptr_but_skips_custom_mask(
    monkeypatch,
):
    _, flashinfer_backend = _import_flashinfer_backend()
    real_zeros = torch.zeros
    allocations = []

    def fake_zeros(*args, **kwargs):
        allocations.append(
            {
                "shape": args[0] if args else kwargs.get("size"),
                "dtype": kwargs.get("dtype"),
                "device": kwargs.get("device"),
            }
        )
        kwargs = dict(kwargs)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(flashinfer_backend.torch, "zeros", fake_zeros)

    target = _make_backend(is_draft_runner=False)
    target.init_cuda_graph_state(max_bs=4, max_num_tokens=2)

    draft = _make_backend(is_draft_runner=True)
    draft.init_cuda_graph_state(max_bs=4, max_num_tokens=2)

    assert target.cuda_graph_custom_mask is not None
    assert target.cuda_graph_custom_mask.shape == (256,)
    assert target.cuda_graph_custom_mask.dtype == torch.uint8

    assert draft.cuda_graph_custom_mask is None
    assert [x.shape for x in draft.cuda_graph_qk_indptr] == [torch.Size([5])]
    assert [x.shape for x in draft.cuda_graph_qo_indptr] == [torch.Size([5])]

    mask_allocations = [
        item
        for item in allocations
        if item["dtype"] == torch.uint8 and item["shape"] == 256
    ]
    assert len(mask_allocations) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
