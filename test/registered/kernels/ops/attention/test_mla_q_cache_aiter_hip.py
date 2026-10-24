# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes  # noqa: E402

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not mla_q_cache_aiter_hip.available(),
    reason="Kimi-K3 fused MLA Q/cache tests require AITER on gfx950",
)


def _inputs(tokens: int, cache_dtype=torch.bfloat16, output_dtype=None):
    torch.manual_seed(23)
    heads = 12
    q_nope = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    q_pe = torch.randn(
        tokens, heads, 64, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_nope = torch.randn(
        tokens, 1, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_pe = torch.randn(tokens, 1, 64, device="cuda", dtype=torch.bfloat16).contiguous()
    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "k_nope": k_nope,
        "k_pe": k_pe,
        "kv_cache": torch.zeros(tokens + 4, 1, 576, device="cuda", dtype=cache_dtype),
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
        # The cos/sin tables have one identity row, so every position must be 0.
        "positions": torch.zeros(tokens, device="cuda", dtype=torch.int64),
        "k_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "q_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "cos_cache": torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        "sin_cache": torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        "out": torch.empty(
            tokens,
            heads,
            576,
            device="cuda",
            dtype=cache_dtype if output_dtype is None else output_dtype,
        ),
    }


def _reference(values):
    return (
        torch.cat((values["q_nope"], values["q_pe"]), dim=-1).to(values["out"].dtype),
        torch.cat((values["k_nope"], values["k_pe"]), dim=-1).to(
            values["kv_cache"].dtype
        ),
    )


@pytest.mark.parametrize("tokens", [1, 4, 8, 64, 128, 256])
@pytest.mark.parametrize(
    ("cache_dtype", "output_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (dtypes.fp8, dtypes.fp8),
        (dtypes.fp8, torch.bfloat16),
    ],
)
def test_fused_mla_q_cache_identity_rope(tokens, cache_dtype, output_dtype):
    values = _inputs(tokens, cache_dtype=cache_dtype, output_dtype=output_dtype)
    expected_q, expected_k = _reference(values)
    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    assert actual is values["out"]
    torch.testing.assert_close(actual.float(), expected_q.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        values["kv_cache"][:tokens].float(),
        expected_k.float(),
        atol=0.02,
        rtol=0.02,
    )


def test_fused_mla_q_cache_skips_negative_slots():
    values = _inputs(4)
    values["slot_mapping"][-2:] = -1
    expected_q, expected_k = _reference(values)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    torch.testing.assert_close(values["out"][:2], expected_q[:2])
    torch.testing.assert_close(values["kv_cache"][:2], expected_k[:2])
    assert torch.count_nonzero(values["kv_cache"][2:]).item() == 0


def test_fused_mla_q_cache_uses_each_token_slot_across_page_boundaries():
    values = _inputs(8)
    slots = torch.tensor(
        [1, 127, 128, 129, 255, 256, 257, 299],
        dtype=torch.int64,
        device="cuda",
    )
    values["slot_mapping"] = slots
    values["kv_cache"] = torch.zeros(304, 1, 576, dtype=torch.bfloat16, device="cuda")
    expected_q, expected_k = _reference(values)

    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected_q)
    torch.testing.assert_close(values["kv_cache"][slots], expected_k)


def test_fused_mla_q_cache_graph_replay_uses_changed_inputs():
    values = _inputs(64)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_q_cache_aiter_hip.run(**values)
    graph.replay()
    torch.cuda.synchronize()
    first = values["out"].clone()
    values["q_nope"].copy_(torch.randn_like(values["q_nope"]))
    values["q_pe"].copy_(torch.randn_like(values["q_pe"]))
    values["k_nope"].copy_(torch.randn_like(values["k_nope"]))
    values["k_pe"].copy_(torch.randn_like(values["k_pe"]))
    values["kv_cache"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_q, expected_k = _reference(values)
    assert not torch.equal(first, values["out"])
    torch.testing.assert_close(values["out"], expected_q)
    torch.testing.assert_close(values["kv_cache"][:64], expected_k)


def test_fused_mla_q_cache_support_is_narrow():
    values = _inputs(1)
    assert mla_q_cache_aiter_hip.covered(**values)
    values["positions"] = values["positions"].to(torch.float32)
    assert not mla_q_cache_aiter_hip.covered(**values)
    with pytest.raises(ValueError, match="unsupported inputs"):
        mla_q_cache_aiter_hip.run(**values)


def test_fused_mla_q_cache_writes_into_padded_out():
    values = _inputs(4, cache_dtype=dtypes.fp8, output_dtype=dtypes.fp8)
    heads = values["q_nope"].shape[1]
    padded = torch.empty((4, 16, 576), device="cuda", dtype=dtypes.fp8)
    padded.view(torch.uint8).fill_(7)
    values["out"] = padded
    assert mla_q_cache_aiter_hip.covered(**values)
    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    expected_q, _ = _reference(values)
    torch.testing.assert_close(
        actual[:, :heads].float(), expected_q.float(), atol=0.02, rtol=0.02
    )
    assert torch.equal(actual[:, heads:], padded[:, heads:])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
