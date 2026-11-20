"""A learnable attention sink adds exp(sink) to each head's softmax denominator
exactly once. Under DCP every rank holds a partial over its KV stripe for all of
the group's gathered heads, so the sink must enter exactly one partial."""

import math
import sys
import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_dcp import fold_attention_sink
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.dcp import lse_weighted_combine_cpu

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BACKEND = "sglang.srt.layers.attention.dsa_backend"


def _sink_softmax(logits, values, valid, sink):
    """Unsharded reference: [rows, heads, keys] logits, [keys, D] values."""
    logits = logits.masked_fill(~valid[:, None, :], -math.inf)
    weights = torch.exp(logits - sink[None, :, None])
    denom = 1.0 + weights.sum(-1, keepdim=True)
    return torch.einsum("rhk,kd->rhd", weights / denom, values)


def _partial(logits, values, valid):
    """One rank's sink-free partial as the TRT-LLM kernel returns it: zero output
    and -inf LSE for rows without keys, base-2 LSE otherwise."""
    logits = logits.masked_fill(~valid[:, None, :], -math.inf)
    lse = torch.logsumexp(logits, dim=-1)
    empty = torch.isinf(lse)
    probs = torch.exp(logits - lse.masked_fill(empty, 0.0)[..., None])
    out = torch.einsum("rhk,kd->rhd", probs, values).masked_fill(empty[..., None], 0)
    return out, lse / math.log(2.0)


class TestDcpAttentionSink(unittest.TestCase):
    def test_owner_rank_fold_matches_unsharded_sink_softmax(self):
        generator = torch.Generator().manual_seed(0)
        rows, local_heads, keys, dim = 4, 3, 13, 8
        values = torch.randn(keys, dim, generator=generator, dtype=torch.float64)
        # Row 0 has one key, so every other rank's stripe is empty; row 3 has none.
        lengths = torch.tensor([1, 5, keys, 0])
        valid = torch.arange(keys)[None, :] < lengths[:, None]
        for dcp_size in (1, 2, 4):
            heads = local_heads * dcp_size
            logits = 3 * torch.randn(
                rows, heads, keys, generator=generator, dtype=torch.float64
            )
            sink = 2 * torch.randn(heads, generator=generator, dtype=torch.float64)
            outputs, lses = [], []
            for rank in range(dcp_size):
                stripe = torch.arange(keys) % dcp_size == rank
                out, lse = _partial(logits, values, valid & stripe[None, :])
                fold_attention_sink(
                    out,
                    lse,
                    sink[rank * local_heads : (rank + 1) * local_heads],
                    head_offset=rank * local_heads,
                )
                outputs.append(out)
                lses.append(lse)
            merged = lse_weighted_combine_cpu(
                torch.stack(outputs), torch.stack(lses), is_lse_base_on_e=False
            )
            with self.subTest(dcp_size=dcp_size):
                torch.testing.assert_close(
                    merged.double(),
                    _sink_softmax(logits, values, valid, sink),
                    atol=1e-5,
                    rtol=1e-5,
                )


def _trtllm_decode_with_sink(*, dcp_size, dcp_rank, sink, out, lse):
    """Run the real _forward_trtllm decode with the TRT-LLM kernel replaced by a
    stub that returns the given sink-free partial."""
    rows, heads = lse.shape
    seq_lens = torch.full((rows,), 4, dtype=torch.int32)
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            cache_seqlens_int32=seq_lens,
            page_table_1=torch.zeros((rows, 4), dtype=torch.int32),
            max_seq_len_k=4,
            dcp_topk_length=seq_lens,
        ),
        kv_cache_dtype=torch.bfloat16,
        _trtllm_fused_q=False,
        kv_layers=SimpleNamespace(
            read=lambda _layer_id: SimpleNamespace(key=lambda: torch.zeros((8, 3)))
        ),
        physical_page_size=1,
        kv_cache_dim=3,
        use_fused_topk=False,
        qk_nope_head_dim=2,
        kv_lora_rank=out.shape[-1],
        qk_rope_head_dim=1,
        workspace_buffer=None,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=dcp_rank,
        dsa_shards_index_k=dcp_size > 1,
        _dcp_page_table_cache=None,
        _multi_ctas_kv_counter_for=lambda _rows: None,
        get_device_int32_arange=lambda n: torch.arange(n, dtype=torch.int32),
    )
    for name in ("_pad_topk_indices", "_pad_trtllm_sparse_page_table"):
        setattr(
            backend, name, MethodType(getattr(DeepseekSparseAttnBackend, name), backend)
        )
    flashinfer = ModuleType("flashinfer")
    flashinfer.decode = ModuleType("flashinfer.decode")
    with (
        patch.dict(
            "sys.modules",
            {"flashinfer": flashinfer, "flashinfer.decode": flashinfer.decode},
        ),
        patch(f"{_BACKEND}.get_in_autotune_dummy_run", return_value=False),
        patch(
            f"{_BACKEND}.transform_index_page_table_decode",
            side_effect=lambda *, page_table, topk_indices, page_size: topk_indices,
        ),
        patch(
            f"{_BACKEND}.localize_dcp_page_table",
            side_effect=lambda _cache, *, page_table, **_: (page_table, seq_lens),
        ),
        patch(
            f"{_BACKEND}.dcp_trtllm_sparse_attention",
            return_value=(out.clone(), lse.clone()),
        ),
    ):
        return DeepseekSparseAttnBackend._forward_trtllm(
            backend,
            q=torch.zeros((rows, heads, 3)),
            k=None,
            v=None,
            layer=SimpleNamespace(
                layer_id=0, tp_q_head_num=heads, head_dim=3, scaling=1.0
            ),
            forward_batch=SimpleNamespace(),
            seq_lens=seq_lens,
            save_kv_cache=False,
            topk_indices=torch.zeros((rows, 2), dtype=torch.int32),
            attn_sink=sink,
        )


class TestTrtllmDecodeFoldsSink(unittest.TestCase):
    """The TRT-LLM decode path folds the sink on its own heads only: a DCP partial
    returns (out, lse) with this rank's slot folded, plain decode folds every head."""

    def test_sink_fold_follows_the_backend_dcp_rank(self):
        generator = torch.Generator().manual_seed(0)
        rows, local_heads, dim = 3, 2, 4
        for dcp_size, dcp_rank in ((2, 1), (1, 0)):
            heads = local_heads * dcp_size
            out = torch.randn(rows, 1, heads, dim, generator=generator)
            lse = torch.randn(rows, heads, generator=generator)
            sink = torch.randn(local_heads, generator=generator)
            expected_out, expected_lse = out.clone(), lse.clone()
            fold_attention_sink(
                expected_out.view(rows, heads, dim),
                expected_lse,
                sink,
                head_offset=dcp_rank * local_heads,
            )
            with self.subTest(dcp_size=dcp_size, dcp_rank=dcp_rank):
                result = _trtllm_decode_with_sink(
                    dcp_size=dcp_size, dcp_rank=dcp_rank, sink=sink, out=out, lse=lse
                )
                if dcp_size == 1:
                    torch.testing.assert_close(result, expected_out)
                    continue
                result_out, result_lse = result
                torch.testing.assert_close(result_out, expected_out)
                torch.testing.assert_close(result_lse, expected_lse)
                other = slice(0, local_heads)
                torch.testing.assert_close(result_out[:, :, other], out[:, :, other])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
