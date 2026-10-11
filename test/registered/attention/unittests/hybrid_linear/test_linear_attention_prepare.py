"""CPU contracts for opt-in operator startup preparation."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.prepare import (
    LinearAttentionPrepareSpec,
    linear_backends,
    make_prepare_inputs,
    parse_sizes,
    prepare_model_linear_attention,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_sizes():
    assert parse_sizes("") == ()
    assert parse_sizes(" 1, 4,1 ") == (1, 4)
    for value in ("0", "-1", "not-a-size"):
        with pytest.raises(ValueError):
            parse_sizes(value)


def test_scratch_preserves_pool_strides_and_parameters():
    pool = torch.full((3, 4, 2, 8, 8), 17.0)[:, 0]
    params = torch.ones(2)
    spec = LinearAttentionPrepareSpec(
        2, 2, 2, 8, 8, torch.float32, pool, params, params
    )
    inputs = make_prepare_inputs(spec, "decode", 2, kind="gdn")
    state = inputs["ssm_states"]
    assert state.stride() == pool.stride()
    assert state.data_ptr() != pool.data_ptr()
    state.fill_(-3)
    assert torch.all(pool == 17)
    assert torch.all(params == 1)
    assert inputs["q"].stride(1) == 3 * 2 * 8


def test_mode_and_overlap_resolution():
    prefill = SimpleNamespace(kernel_dispatcher=object())
    decode = SimpleNamespace(kernel_dispatcher=object())
    split = SimpleNamespace(prefill_backend=prefill, decode_backend=decode)
    overlap = SimpleNamespace(primary=split, children=[split, split])
    assert list(linear_backends(overlap, "prefill")) == [prefill] * 3
    assert list(linear_backends(overlap, "decode")) == [decode]


def test_runner_decode_fallback_and_shape_dedup(monkeypatch):
    calls = []

    class Kernel:
        def prepare_for_inference(self, spec, mode, sizes):
            calls.append((mode, sizes, spec.num_v_heads))
            return 1

    kernel = Kernel()
    dispatcher = SimpleNamespace(extend_kernel=kernel, decode_kernel=kernel)
    backend = SimpleNamespace(kernel_dispatcher=dispatcher, mamba_chunk_size=64)
    model = torch.nn.ModuleList(
        [
            RadixLinearAttention(
                i, 2, 2, 2, 8, 8, 8, A_log=torch.ones(2), dt_bias=torch.ones(2)
            )
            for i in range(2)
        ]
    )
    state = torch.ones(3, 2, 8, 8)
    runner = SimpleNamespace(
        device="cuda",
        dtype=torch.float32,
        model=model,
        attn_backend=backend,
        decode_attn_backend=None,
        decode_attn_backend_group=[],
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(temporal=state)
        ),
    )
    monkeypatch.setenv("SGLANG_LINEAR_ATTN_PREPARE_PREFILL_TOKENS", "64,128")
    monkeypatch.setenv("SGLANG_LINEAR_ATTN_PREPARE_DECODE_BATCH_SIZES", "1")
    with patch("torch.cuda.synchronize"):
        result = prepare_model_linear_attention(runner)
    assert calls == [("prefill", (64,), 2), ("prefill", (128,), 2), ("decode", (1,), 2)]
    assert len(result) == 3
    assert torch.all(state == 1)


def test_default_is_opt_in():
    assert envs.SGLANG_PREPARE_LINEAR_ATTN.default is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
