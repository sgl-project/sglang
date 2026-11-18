# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes  # noqa: E402

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402
from sglang.test.kernels.mla_q_cache import (  # noqa: E402
    mla_q_cache_inputs,
    mla_q_cache_reference,
)

register_amd_ci(est_time=35, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not mla_q_cache_aiter_hip.available(),
    reason="Kimi-K3 fused MLA Q/cache tests require AITER on gfx950",
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
    values = mla_q_cache_inputs(
        tokens, cache_dtype=cache_dtype, output_dtype=output_dtype
    )
    expected_q, expected_k = mla_q_cache_reference(values)
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
    """A slot < 0 token skips only its KV cache write; its Q row must still be
    written, as the split chain and the Triton kernel write it."""
    values = mla_q_cache_inputs(4)
    values["slot_mapping"][-2:] = -1
    values["out"].fill_(float("nan"))
    expected_q, expected_k = mla_q_cache_reference(values)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    torch.testing.assert_close(values["out"], expected_q)
    torch.testing.assert_close(values["kv_cache"][:2], expected_k[:2])
    assert torch.count_nonzero(values["kv_cache"][2:]).item() == 0


def test_fused_mla_q_cache_uses_each_token_slot_across_page_boundaries():
    values = mla_q_cache_inputs(8)
    slots = torch.tensor(
        [1, 127, 128, 129, 255, 256, 257, 299],
        dtype=torch.int64,
        device="cuda",
    )
    values["slot_mapping"] = slots
    values["kv_cache"] = torch.zeros(304, 1, 576, dtype=torch.bfloat16, device="cuda")
    expected_q, expected_k = mla_q_cache_reference(values)

    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected_q)
    torch.testing.assert_close(values["kv_cache"][slots], expected_k)


def test_fused_mla_q_cache_graph_replay_uses_changed_inputs():
    values = mla_q_cache_inputs(64)
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
    expected_q, expected_k = mla_q_cache_reference(values)
    assert not torch.equal(first, values["out"])
    torch.testing.assert_close(values["out"], expected_q)
    torch.testing.assert_close(values["kv_cache"][:64], expected_k)


def test_fused_mla_q_cache_support_is_narrow():
    values = mla_q_cache_inputs(1)
    assert mla_q_cache_aiter_hip.covered(**values)
    values["positions"] = values["positions"].to(torch.float32)
    assert not mla_q_cache_aiter_hip.covered(**values)
    with pytest.raises(ValueError, match="unsupported inputs"):
        mla_q_cache_aiter_hip.run(**values)


def test_fused_mla_q_cache_writes_into_padded_out():
    values = mla_q_cache_inputs(4, cache_dtype=dtypes.fp8, output_dtype=dtypes.fp8)
    heads = values["q_nope"].shape[1]
    padded = torch.empty((4, 16, 576), device="cuda", dtype=dtypes.fp8)
    padded.view(torch.uint8).fill_(7)
    values["out"] = padded
    assert mla_q_cache_aiter_hip.covered(**values)
    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    expected_q, _ = mla_q_cache_reference(values)
    torch.testing.assert_close(
        actual[:, :heads].float(), expected_q.float(), atol=0.02, rtol=0.02
    )
    assert torch.equal(actual[:, heads:], padded[:, heads:])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
