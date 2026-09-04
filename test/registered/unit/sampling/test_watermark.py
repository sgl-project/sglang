import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.watermark import (
    WatermarkState,
    _hash_context_token_ids,
    build_watermark_batch_config,
    normalize_watermark_request,
)
from sglang.srt.utils.request_logger import _transform_data_for_logging
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def test_repeated_context_and_greedy_bypass():
    """A repeated context must sample normally instead of forcing forever."""
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device=device,
    )
    req_pool_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state.init_from_prompt(req_pool_indices, [[10, 11], [20, 21]])
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), device=device),
        top_ks=torch.tensor([64, 1], device=device, dtype=torch.int32),
        top_ps=torch.ones(2, device=device),
        min_ps=torch.zeros(2, device=device),
    )

    first_logits = torch.zeros((2, 64), device=device)
    state.force(first_logits, req_pool_indices, sampling_info)

    assert torch.isfinite(first_logits[0]).sum().item() == 1
    assert torch.equal(first_logits[1], torch.zeros(64, device=device))
    assert state.num_watermarked_contexts.tolist() == [1, 0]

    state.append(
        req_pool_indices[:1], torch.tensor([10], device=device, dtype=torch.int32)
    )
    state.append(
        req_pool_indices[:1], torch.tensor([11], device=device, dtype=torch.int32)
    )
    repeated_logits = torch.zeros((2, 64), device=device)
    state.force(repeated_logits, req_pool_indices, sampling_info)

    assert torch.equal(repeated_logits, torch.zeros_like(repeated_logits))
    assert state.num_watermarked_contexts.tolist() == [1, 0]


def test_retracted_request_restores_context_history():
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device=device,
    )
    req_pool_indices = torch.tensor([1], device=device, dtype=torch.int32)
    request = SimpleNamespace(
        retracted_stain=True,
        origin_input_ids=[10, 11],
        output_ids=[10, 11],
        sampling_params=SimpleNamespace(
            watermark=normalize_watermark_request({"context_window": 2}), top_k=64
        ),
        get_fill_ids=lambda: [10, 11, 10, 11],
    )
    forward_mode = SimpleNamespace(
        is_extend_without_speculative=lambda: True,
        is_mixed=lambda: False,
    )
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        reqs=[request],
        decoding_reqs=None,
    )
    history = state.retracted_context_hashes(batch)
    assert history is not None
    expected_hashes = {
        _hash_context_token_ids([10, 11]),
        _hash_context_token_ids([11, 10]),
    }
    assert {
        value if value >= 0 else value + 2**32 for value in history[0]
    } == expected_hashes
    state.init_from_prompt(
        req_pool_indices,
        state.prompt_tails(batch),
        history,
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device=device),
        top_ks=torch.tensor([64], device=device, dtype=torch.int32),
        top_ps=torch.ones(1, device=device),
        min_ps=torch.zeros(1, device=device),
    )
    logits = torch.zeros((1, 64), device=device)
    state.force(logits, req_pool_indices, sampling_info)

    assert torch.equal(logits, torch.zeros_like(logits))
    assert state.num_watermarked_contexts[1].item() == 2


def test_per_request_config_resolution_and_redaction():
    secret = "fedcba9876543210"
    requests = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request(
                    {"key": secret, "context_window": 2}
                )
            )
        ),
        SimpleNamespace(sampling_params=SimpleNamespace(watermark=None)),
    ]

    keys, context_windows, enabled = build_watermark_batch_config(
        requests,
        default_key="0123456789abcdef",
        default_context_window=4,
        device="cuda",
    )

    assert keys.tolist() == [0xFEDCBA9876543210 - (1 << 64), 0x0123456789ABCDEF]
    assert context_windows.tolist() == [2, 4]
    assert enabled.tolist() == [True, True]

    keys, context_windows, enabled = build_watermark_batch_config(
        requests,
        default_key=None,
        default_context_window=4,
        device="cuda",
    )
    assert keys.tolist() == [0xFEDCBA9876543210 - (1 << 64), 0]
    assert context_windows.tolist() == [2, 4]
    assert enabled.tolist() == [True, False]
    assert secret not in repr(requests[0].sampling_params.watermark)
    logged = _transform_data_for_logging(
        {
            "sampling_params": {"watermark": {"key": secret, "context_window": 2}},
            "watermark_key": secret,
            "internal_states": [{"watermark_key": secret}],
        }
    )
    assert logged["sampling_params"]["watermark"]["key"] == "<redacted>"
    assert logged["watermark_key"] == "<redacted>"
    assert logged["internal_states"][0]["watermark_key"] == "<redacted>"

    with pytest.raises(ValueError, match="unknown fields"):
        normalize_watermark_request({"key": secret, "provider": "textseal"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
