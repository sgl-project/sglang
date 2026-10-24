"""Regression checks for real rows and current metadata in BCG eager regions."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha


@pytest.mark.parametrize("n", [0, 1, 356, 512])
def test_bcg_kv_write_uses_matching_real_rows(n):
    pool = Mock()
    batch = SimpleNamespace(
        global_num_token_non_padded_cpu=n,
        out_cache_loc=torch.arange(512, dtype=torch.int64) + 1000,
        out_cache_loc_is_physical=True,
    )
    layer = SimpleNamespace(attn_mha=object())
    kv = torch.zeros(512, 512, dtype=torch.bfloat16)
    rope = torch.zeros(512, 1, 64, dtype=torch.bfloat16)
    with (
        patch.object(forward_mha, "get_forward_batch", return_value=batch),
        patch.object(forward_mha, "get_token_to_kv_pool", return_value=pool),
    ):
        forward_mha.DeepseekMHAForwardMixin._bcg_set_mla_kv.run_eager(layer, kv, rope)
    if not n:
        pool.set_mla_kv_buffer.assert_not_called()
        return
    _, loc, written_kv, written_rope = pool.set_mla_kv_buffer.call_args.args
    assert loc.physical
    assert loc.loc.shape[0] == written_kv.shape[0] == written_rope.shape[0] == n
    assert torch.equal(loc.loc, batch.out_cache_loc[:n])


def test_bcg_mha_reads_current_batch_and_zeroes_padding():
    q = torch.ones(512, 2, 4)
    first = SimpleNamespace(global_num_token_non_padded_cpu=356)
    second = SimpleNamespace(global_num_token_non_padded_cpu=17)
    seen = []

    def attention(q, k, v, batch):
        seen.append(batch)
        assert len(q) == len(k) == len(v) == batch.global_num_token_non_padded_cpu
        return q * len(seen)

    layer = SimpleNamespace(
        num_local_heads=2, v_head_dim=4, _chunked_mha_attention=attention
    )
    with patch.object(forward_mha, "get_forward_batch", side_effect=[first, second]):
        a = forward_mha.DeepseekMHAForwardMixin._bcg_chunked_mha.run_eager(
            layer, q, q, q
        )
        b = forward_mha.DeepseekMHAForwardMixin._bcg_chunked_mha.run_eager(
            layer, q, q, q
        )
    assert seen == [first, second]
    assert torch.all(a[:356] == 1) and torch.all(a[356:] == 0)
    assert torch.all(b[:17] == 2) and torch.all(b[17:] == 0)
