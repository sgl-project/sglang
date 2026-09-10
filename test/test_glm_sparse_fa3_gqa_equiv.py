"""GLM sparse decode: native-GQA FA3 over gathered TopK KV — numerical equivalence.

``fa3_token_sparse_attention`` runs FlashAttention-3 over the per-(batch, kv_head)
gathered TopK KV using the kernel's native GQA broadcast (q carries ``num_heads``,
the gathered cache carries ``num_kv_heads``). This test asserts that path is
numerically equivalent to a pure-torch full-attention reference computed over the
SAME gathered KV, in bf16.

This is the unit-level guard the repo previously lacked. An earlier kv-head
native-GQA FA3 attempt was abandoned because its bf16 accumulation order differed
from the dense backend and drifted under temp=0 greedy decode into a degeneration
loop near seq_len ~1940. The long case here (cache_len 2049) exercises that band,
and a second assertion bounds the delta against the old per-q-head single-head
fold the path used to use.

Standalone-stubbed (mirrors test_glm_sparse_maxpool_forceselect.py) so the
glm_sparse leaf modules import without running the full sglang __init__. Requires
CUDA + sm90 (FA3).
"""

from __future__ import annotations

import sys
import types
import unittest

# Stub the sglang package hierarchy so the glm_sparse leaf modules import without
# running the full sglang __init__ (mirrors the other standalone tests). The FA3
# wrapper lives under sglang.jit_kernel, which fa3_token_sparse_attention imports
# lazily inside the function, so we leave that import to resolve normally.
for _name in [
    "sglang", "sglang.srt", "sglang.srt.layers",
    "sglang.srt.layers.attention", "sglang.srt.layers.attention.glm_sparse",
]:
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [f"python/{_name.replace('.', '/')}"]
    sys.modules[_name] = _pkg

sys.path.insert(0, "python")

import torch


def _make_gathered(batch, num_kv_heads, cache_len, head_dim, valid_lens, seed=0):
    """Build a gathered-KV scratch buffer + current-token KV + cache_seqlens.

    ``gathered_k/v`` are [batch, num_kv_heads, cache_len, head_dim] (the layout
    the model gather allocates) with valid history front-packed in [0, valid)
    and the rest (incl. the trailing scratch slot) zeroed so padding cannot leak
    into the reference. ``cache_seqlens`` is [batch] (valid count, identical
    across kv-heads, matching the model path).
    """
    torch.manual_seed(seed)
    gathered_k = torch.zeros(
        batch, num_kv_heads, cache_len, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    gathered_v = torch.zeros_like(gathered_k)
    for b in range(batch):
        L = valid_lens[b]
        gathered_k[b, :, :L] = torch.randn(
            num_kv_heads, L, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        gathered_v[b, :, :L] = torch.randn(
            num_kv_heads, L, head_dim, dtype=torch.bfloat16, device="cuda"
        )
    k_cur = torch.randn(batch, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cur = torch.randn(batch, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cache_seqlens = torch.tensor(valid_lens, dtype=torch.int32, device="cuda")
    return gathered_k, gathered_v, k_cur, v_cur, cache_seqlens


def _reference_full_attention(q, gathered_k, gathered_v, k_cur, v_cur, cache_seqlens, scaling):
    """Pure-torch full attention over the gathered KV + appended current token.

    fp32 accumulation; returns [batch, num_heads, head_dim]. For each query head
    qh, attends over the qh's kv-head's valid history [0, L) plus the current
    token (causal: the lone query sees the whole [0, L] window). ``gathered_*``
    are [batch, num_kv_heads, cache_len, head_dim].
    """
    batch, num_heads, head_dim = q.shape
    num_kv_heads = gathered_k.shape[1]
    group_size = num_heads // num_kv_heads
    out = torch.empty(batch, num_heads, head_dim, dtype=torch.float32, device=q.device)
    for b in range(batch):
        L = int(cache_seqlens[b].item())
        for qh in range(num_heads):
            kvh = qh // group_size
            keys = torch.cat(
                [gathered_k[b, kvh, :L].float(), k_cur[b, kvh].float().unsqueeze(0)], dim=0
            )  # [L+1, d]
            vals = torch.cat(
                [gathered_v[b, kvh, :L].float(), v_cur[b, kvh].float().unsqueeze(0)], dim=0
            )  # [L+1, d]
            logits = (q[b, qh].float() @ keys.t()) * scaling  # [L+1]
            probs = torch.softmax(logits, dim=-1)
            out[b, qh] = probs @ vals
    return out


def _old_single_head_fold(q, gathered_k, gathered_v, k_cur, v_cur, cache_seqlens, scaling):
    """The previous per-q-head single-head FA3 fold, for accumulation-order delta.

    Expands kv-head granular gathered KV to q-heads and folds (batch, q_head) into
    the batch dim, running batch*q_heads single-head FA3 problems — the exact
    behavior the native-GQA path replaces. ``gathered_*`` are [batch,
    num_kv_heads, cache_len, head_dim]; returns [batch, num_heads, head_dim].
    """
    from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache

    batch, num_heads, head_dim = q.shape
    num_kv_heads = gathered_k.shape[1]
    cache_len = gathered_k.shape[2]
    group_size = num_heads // num_kv_heads
    n = batch * num_heads

    # [b, kv_heads, cache_len, d] -> expand each kv-head over its q-head group
    gk = gathered_k.repeat_interleave(group_size, dim=1)  # [b, q_heads, cache_len, d]
    gv = gathered_v.repeat_interleave(group_size, dim=1)
    kc = k_cur.repeat_interleave(group_size, dim=1)  # [b, q_heads, d]
    vc = v_cur.repeat_interleave(group_size, dim=1)
    seqlens = cache_seqlens[:, None].expand(batch, num_heads).reshape(n).to(torch.int32)

    q_fold = q.reshape(n, 1, 1, head_dim)
    k_cache = gk.reshape(n, cache_len, 1, head_dim)
    v_cache = gv.reshape(n, cache_len, 1, head_dim)
    k_new = kc.reshape(n, 1, 1, head_dim)
    v_new = vc.reshape(n, 1, 1, head_dim)

    out = flash_attn_with_kvcache(
        q=q_fold, k_cache=k_cache, v_cache=v_cache, k=k_new, v=v_new,
        cache_seqlens=seqlens, softmax_scale=scaling, causal=True,
    )
    return out.reshape(batch, num_heads, head_dim).float()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFa3GqaEquivalence(unittest.TestCase):
    """fa3_token_sparse_attention (native GQA) vs full-attention reference."""

    BATCH = 2
    NUM_KV_HEADS = 4
    GROUP_SIZE = 3
    HEAD_DIM = 128

    @property
    def num_heads(self):
        return self.NUM_KV_HEADS * self.GROUP_SIZE

    def _run_native_gqa(self, q, gathered_k, gathered_v, k_cur, v_cur, cache_seqlens, scaling):
        from sglang.srt.layers.attention.glm_sparse.sparse_attention import (
            fa3_token_sparse_attention,
        )

        # The function appends the current token in-place into gathered_k/v's
        # trailing slot, so clone to keep the inputs pristine for the reference.
        gk = gathered_k.clone()
        gv = gathered_v.clone()
        out = fa3_token_sparse_attention(q, gk, gv, k_cur, v_cur, cache_seqlens, scaling)
        return out.reshape(self.BATCH, self.num_heads, self.HEAD_DIM)

    def _check_case(self, valid_lens, cache_len, seed):
        scaling = self.HEAD_DIM ** -0.5
        torch.manual_seed(seed)
        q = torch.randn(
            self.BATCH, self.num_heads, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        gathered_k, gathered_v, k_cur, v_cur, cache_seqlens = _make_gathered(
            self.BATCH, self.NUM_KV_HEADS, cache_len, self.HEAD_DIM, valid_lens, seed=seed + 1
        )

        out = self._run_native_gqa(
            q, gathered_k, gathered_v, k_cur, v_cur, cache_seqlens, scaling
        )
        self.assertFalse(torch.isnan(out).any(), "native-GQA FA3 produced NaN")
        self.assertFalse(torch.isinf(out).any(), "native-GQA FA3 produced Inf")

        ref = _reference_full_attention(
            q, gathered_k, gathered_v, k_cur, v_cur, cache_seqlens, scaling
        )
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

        # Bound the bf16 accumulation-order delta against the OLD single-head
        # fold the native-GQA path replaces (the path that previously avoided the
        # seq_len ~1940 greedy-decode drift). Small delta => the switch does not
        # materially change the per-step numerics.
        old = _old_single_head_fold(
            q, gathered_k, gathered_v, k_cur, v_cur, cache_seqlens, scaling
        )
        max_abs_diff = (out.float() - old).abs().max().item()
        self.assertLess(
            max_abs_diff, 5e-2,
            f"native-GQA vs single-head-fold max_abs_diff={max_abs_diff:.4g} too large",
        )

    def test_short_cache(self):
        # valid counts below topk; cache_len = max(valid) + 1.
        self._check_case(valid_lens=[37, 256], cache_len=257, seed=100)

    def test_long_cache_across_1940(self):
        # Exercise the seq_len ~1940 band where the old native-GQA attempt drifted.
        self._check_case(valid_lens=[2000, 1900], cache_len=2049, seed=200)


if __name__ == "__main__":
    unittest.main()
