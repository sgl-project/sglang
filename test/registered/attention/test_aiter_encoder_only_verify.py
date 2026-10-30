"""AITER target verify honors ENCODER_ONLY (bidirectional) attention layers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = False
if is_hip():
    from sglang.srt.utils.aiter import aiter_unified_attention_supports_non_causal

    _RUNNABLE = aiter_unified_attention_supports_non_causal()
if _RUNNABLE:
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

_DRAFT_NUM = 4
_PAGE_SIZE = 16
# (prefix lengths, query heads, kv heads, head dim)
_SHAPES = {
    # Kimi-K3 DSpark draft at TP8; a short prefix takes the 2D kernel
    "dspark_short": ((37, 300), 8, 2, 64),
    # past 512 keys the verify takes the split-KV 3D kernel
    "dspark_long": ((37, 3000), 8, 2, 64),
    # the causal-only unified_attention_3d_mtp shape (GQA 16, head dim 256)
    "mtp_shape": ((600, 1500), 32, 2, 256),
}


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.start_layer = 0
        self.layer_num = 1
        self.layer_transfer_counter = None
        self.layer_router = PlainPoolRouter(self, "full")

    def get_kv_buffer(self, _layer_id):
        return self.k_cache, self.v_cache


def _reference(q, keys, values, prefix_lens, scaling, causal):
    """Each request's block attends to its prefix and its own block; causal masks
    the block's tail."""
    num_q_heads, num_kv_heads = q.shape[1], keys[0].shape[1]
    group = num_q_heads // num_kv_heads
    out = []
    for i, prefix_len in enumerate(prefix_lens):
        rows = slice(i * _DRAFT_NUM, (i + 1) * _DRAFT_NUM)
        k = keys[i].float().repeat_interleave(group, dim=1)
        v = values[i].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), k) * scaling
        if causal:
            visible = torch.ones(
                _DRAFT_NUM, prefix_len + _DRAFT_NUM, dtype=torch.bool, device=q.device
            ).tril(diagonal=prefix_len)
            scores = scores.masked_fill(~visible, float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v))
    return torch.cat(out)


@unittest.skipUnless(
    _RUNNABLE, "requires HIP with an AITER unified_attention that takes causal=False"
)
class TestAiterEncoderOnlyTargetVerify(CustomTestCase):
    def _run_verify(self, *, shape, attn_type, kv_dtype):
        prefix_lens, num_q_heads, num_kv_heads, head_dim = shape
        torch.manual_seed(0)
        device = "cuda"
        bs = len(prefix_lens)
        seq_lens = [n + _DRAFT_NUM for n in prefix_lens]
        pages_per_seq = -(-max(seq_lens) // _PAGE_SIZE)

        k_cache = torch.zeros(
            bs * pages_per_seq,
            _PAGE_SIZE,
            num_kv_heads,
            head_dim,
            dtype=kv_dtype,
            device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        keys, values = [], []
        for i, seq_len in enumerate(seq_lens):
            # The verify block's K/V sit in the cache already, as the write
            # before attention leaves them.
            k, v = (
                torch.randn(
                    seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
                ).to(kv_dtype)
                for _ in range(2)
            )
            k_cache.view(bs, -1, num_kv_heads, head_dim)[i, :seq_len] = k
            v_cache.view(bs, -1, num_kv_heads, head_dim)[i, :seq_len] = v
            keys.append(k)
            values.append(v)
        q = torch.randn(
            bs * _DRAFT_NUM, num_q_heads, head_dim, dtype=torch.bfloat16, device=device
        )

        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = False
        backend._use_unified_verify = True
        backend.kv_cache_dtype = kv_dtype
        backend.page_size = _PAGE_SIZE
        unit_scale = torch.ones(1, dtype=torch.float32, device=device)
        backend.k_scale = backend.v_scale = unit_scale
        backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
        backend.forward_metadata = SimpleNamespace(
            kv_indices=torch.arange(
                bs * pages_per_seq, dtype=torch.int32, device=device
            ).view(bs, pages_per_seq),
            swa_page_table=None,
            qo_indptr=torch.arange(
                0, bs * _DRAFT_NUM + 1, _DRAFT_NUM, dtype=torch.int32, device=device
            ),
            max_q_len=_DRAFT_NUM,
            verify_seqused_k=None,
        )
        scaling = head_dim**-0.5
        layer = SimpleNamespace(
            layer_id=0,
            attn_type=attn_type,
            is_cross_attention=False,
            sliding_window_size=None,
            tp_q_head_num=num_q_heads,
            tp_k_head_num=num_kv_heads,
            tp_v_head_num=num_kv_heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            k_scale=None,
            v_scale=None,
            scaling=scaling,
            logit_cap=0.0,
        )
        forward_batch = SimpleNamespace(
            attn_attend_prefix_cache=False,
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc=None,
            seq_lens=torch.tensor(prefix_lens, dtype=torch.int64, device=device),
        )
        out = backend.forward_extend(
            q, None, None, layer, forward_batch, save_kv_cache=False
        ).view(-1, num_q_heads, head_dim)
        return out, _reference(
            q,
            keys,
            values,
            prefix_lens,
            scaling,
            causal=attn_type != AttentionType.ENCODER_ONLY,
        )

    def test_encoder_only_verify_is_bidirectional_within_the_block(self):
        """An ENCODER_ONLY draft (DSpark) must see its whole block from every row,
        on the 2D and split-KV kernels and on the shape the causal-only
        unified_attention_3d_mtp serves; the backend used to mask the block."""
        for name, shape in _SHAPES.items():
            kv_dtypes = (
                (fp8_dtype,) if name == "mtp_shape" else (torch.bfloat16, fp8_dtype)
            )
            for kv_dtype in kv_dtypes:
                for attn_type in (AttentionType.ENCODER_ONLY, AttentionType.DECODER):
                    with self.subTest(
                        shape=name, kv_dtype=kv_dtype, attn_type=attn_type.name
                    ):
                        out, expected = self._run_verify(
                            shape=shape, attn_type=attn_type, kv_dtype=kv_dtype
                        )
                        torch.testing.assert_close(
                            out.float(), expected, atol=3e-2, rtol=3e-2
                        )


if __name__ == "__main__":
    unittest.main()
