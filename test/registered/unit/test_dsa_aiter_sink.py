import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@pytest.mark.parametrize("cuda,hip", [(True, False), (False, True)])
def test_graph_dsa_split_surface_supports_cuda_and_hip(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, hip: bool
) -> None:
    monkeypatch.setattr(dsa_utils, "is_cuda", lambda: cuda)
    monkeypatch.setattr(dsa_utils, "is_hip", lambda: hip)
    monkeypatch.setattr(dsa_utils, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(dsa_utils, "is_in_breakable_cuda_graph", lambda: True)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
    )

    assert dsa_utils.is_graph_dsa_split_op_surface(forward_batch)


def test_apply_aiter_attention_sink_matches_explicit_denominator() -> None:
    output = torch.tensor([[[2.0, -4.0], [3.0, 5.0]]], dtype=torch.bfloat16)
    lse = torch.tensor([[1.25, -0.5]], dtype=torch.float32)
    sinks = torch.tensor([0.75, 0.25], dtype=torch.float32)

    actual = dsa_backend.apply_aiter_attention_sink(output, lse, sinks)
    expected = output.float() * (
        torch.exp(lse) / (torch.exp(lse) + torch.exp(sinks))
    ).unsqueeze(-1)

    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("method_name", ["_forward_aiter", "_forward_aiter_extend"])
def test_aiter_sparse_mla_requests_lse_and_applies_sink(
    monkeypatch: pytest.MonkeyPatch, method_name: str
) -> None:
    backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
    backend.device = torch.device("cpu")
    backend.need_pad_heads = True
    backend.head_repeat_factor = 2
    backend.kv_indptr = torch.zeros(2, dtype=torch.int32)
    backend.kv_indices = torch.zeros(4, dtype=torch.int32)

    layer = SimpleNamespace(
        tp_q_head_num=8,
        head_dim=6,
        v_head_dim=4,
        scaling=0.5,
        logit_cap=0.0,
    )
    q_all = torch.zeros((1, 8, 6), dtype=torch.bfloat16)
    kv_cache = torch.zeros((4, 1, 6), dtype=torch.bfloat16)
    page_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)
    sinks = torch.linspace(-0.5, 0.5, 8, dtype=torch.float32)

    monkeypatch.setattr(dsa_backend, "fp8_dtype", torch.float8_e4m3fn, raising=False)
    monkeypatch.setattr(
        dsa_backend,
        "get_valid_kv_indices",
        lambda page_table, kv_indptr, kv_indices, batch_size: None,
        raising=False,
    )

    seen = {}

    def fake_mla_decode_fwd(q, kv, out, *args, **kwargs):
        seen["return_lse"] = kwargs.get("return_lse")
        out.fill_(2.0)
        lse = torch.full(q.shape[:2], 1.25, dtype=torch.float32)
        return out, lse

    monkeypatch.setattr(
        dsa_backend, "mla_decode_fwd", fake_mla_decode_fwd, raising=False
    )

    if method_name == "_forward_aiter":
        metadata = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len_q=1,
        )
        actual = backend._forward_aiter(
            q_all,
            kv_cache,
            page_table,
            layer,
            metadata,
            bs=1,
            attn_sink=sinks,
        )
    else:
        actual = backend._forward_aiter_extend(
            q_all,
            kv_cache,
            page_table,
            layer,
            attn_sink=sinks,
        )

    expected_factor = torch.sigmoid(torch.full((1, 8), 1.25) - sinks)
    expected = 2.0 * expected_factor.unsqueeze(-1).expand(1, 8, 4)
    assert seen["return_lse"] is True
    assert actual.shape == (1, 8, 4)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
