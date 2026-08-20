"""The mask intel_amx hands to extend_attention_cpu in TARGET_VERIFY.

`extend_attention_cpu`'s implicit mask is causal, which is what a decoder layer
wants and what lets the kernel skip the explicit-mask path entirely. A DFLASH
draft block is not causal: its queries are one block that must all see each
other, and that is only expressible as an explicit all-visible qlen mask.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

BATCH_SIZE = 2
DRAFT_TOKEN_NUM = 4
COMMITTED_LEN = 7
EXTEND_LEN = 3
HEAD_NUM = 1
HEAD_DIM = 8


class _StubKVPool:
    def __init__(self):
        self.buffer = torch.zeros(16, HEAD_NUM, HEAD_DIM)

    def set_kv_buffer(self, layer, loc, k, v):
        pass

    def get_key_buffer(self, layer_id):
        return self.buffer

    def get_value_buffer(self, layer_id):
        return self.buffer


def _layer(attn_type):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HEAD_NUM,
        qk_head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        is_cross_attention=False,
        attn_type=attn_type,
        scaling=1.0,
        logit_cap=0.0,
        sliding_window_size=-1,
    )


def _verify_forward_batch(num_tokens):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=BATCH_SIZE,
        seq_lens=torch.full((BATCH_SIZE,), COMMITTED_LEN, dtype=torch.int32),
        spec_info=SimpleNamespace(
            draft_token_num=DRAFT_TOKEN_NUM, tree_topk=1, custom_mask=None
        ),
        extend_seq_lens=torch.full((BATCH_SIZE,), EXTEND_LEN, dtype=torch.int32),
        extend_start_loc=torch.tensor([0, EXTEND_LEN], dtype=torch.int32),
        req_pool_indices=torch.arange(BATCH_SIZE, dtype=torch.int32),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int32),
        encoder_out_cache_loc=None,
        encoder_lens=None,
    )


def _tree_mask_reaching_the_kernel(attn_type):
    """Drive forward_extend the way a forward pass does and return the mask
    argument that reached extend_attention_fwd."""
    num_tokens = BATCH_SIZE * DRAFT_TOKEN_NUM
    forward_batch = _verify_forward_batch(num_tokens)

    backend = object.__new__(IntelAMXAttnBackend)
    backend.device = "cpu"
    backend.token_to_kv_pool = _StubKVPool()
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(BATCH_SIZE, 64, dtype=torch.int32)
    )
    backend.swa_out_cache_loc = None
    backend.forward_metadata = (None, DRAFT_TOKEN_NUM)
    backend.extend_metadata = backend._build_extend_metadata(forward_batch)
    backend._non_causal_masks = {}

    seen = {}
    backend.extend_attention_fwd = lambda *args: seen.update(tree_mask=args[-1])

    qkv = torch.zeros(num_tokens, HEAD_NUM * HEAD_DIM)
    backend.forward_extend(qkv, qkv, qkv, _layer(attn_type), forward_batch)
    return seen["tree_mask"]


class TestIntelAMXNonCausalMask(unittest.TestCase):
    def test_non_causal_verify_layer_gets_an_all_visible_mask(self):
        tree_mask = _tree_mask_reaching_the_kernel(AttentionType.ENCODER_ONLY)

        self.assertEqual(tree_mask.dtype, torch.bool)
        self.assertEqual(
            tree_mask.shape, (BATCH_SIZE * DRAFT_TOKEN_NUM * DRAFT_TOKEN_NUM,)
        )
        self.assertTrue(bool(tree_mask.all()))

    def test_causal_verify_layer_keeps_the_mask_free_path(self):
        # Supplying a mask here would cost the kernel's fast path for no gain.
        self.assertIsNone(_tree_mask_reaching_the_kernel(AttentionType.DECODER))


if __name__ == "__main__":
    unittest.main()
