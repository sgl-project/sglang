import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BATCH_SIZE = 2
DRAFT_TOKEN_NUM = 4
DRAFT_TOPK = 2
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


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HEAD_NUM,
        qk_head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        is_cross_attention=False,
        scaling=1.0,
        logit_cap=0.0,
        sliding_window_size=-1,
    )


def _forward_batch(forward_mode, num_tokens):
    spec_info = (
        SimpleNamespace(draft_token_num=DRAFT_TOKEN_NUM, tree_topk=1, custom_mask=None)
        if forward_mode.is_target_verify()
        else None
    )
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=BATCH_SIZE,
        seq_lens=torch.full((BATCH_SIZE,), COMMITTED_LEN, dtype=torch.int32),
        spec_info=spec_info,
        extend_seq_lens=torch.full((BATCH_SIZE,), EXTEND_LEN, dtype=torch.int32),
        extend_start_loc=torch.tensor([0, EXTEND_LEN], dtype=torch.int32),
        req_pool_indices=torch.arange(BATCH_SIZE, dtype=torch.int32),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int32),
        encoder_out_cache_loc=None,
        encoder_lens=None,
    )


def _run_forward_extend(forward_batch, num_tokens):
    """Drive the backend the way a forward pass does and return the seq_lens
    tensor that reached the kernel."""
    backend = object.__new__(IntelAMXAttnBackend)
    backend.device = "cpu"
    backend.token_to_kv_pool = _StubKVPool()
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(BATCH_SIZE, 64, dtype=torch.int32)
    )
    backend.swa_out_cache_loc = None
    backend.forward_metadata = (None, DRAFT_TOKEN_NUM)
    backend.extend_metadata = backend._build_extend_metadata(forward_batch)

    seen = {}

    def _record(*args):
        seen["seq_lens"] = args[8]

    backend.extend_attention_fwd = _record

    qkv = torch.zeros(num_tokens, HEAD_NUM * HEAD_DIM)
    backend.forward_extend(qkv, qkv, qkv, _layer(), forward_batch)
    return seen["seq_lens"]


def _run_forward_decode(draft_decode_metadata):
    """Drive forward_decode and return the seq_lens tensor that reached the
    kernel."""
    backend = object.__new__(IntelAMXAttnBackend)
    backend.device = "cpu"
    backend.num_head = HEAD_NUM
    backend.v_head_dim = HEAD_DIM
    backend.token_to_kv_pool = _StubKVPool()
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(BATCH_SIZE, 64, dtype=torch.int32)
    )
    backend.draft_decode_metadata = draft_decode_metadata
    num_tokens = (
        BATCH_SIZE
        if draft_decode_metadata is None
        else draft_decode_metadata[1].shape[0]
    )
    backend.forward_metadata = (
        torch.zeros(num_tokens, HEAD_NUM, 8, HEAD_DIM + 1),
        None,
    )

    seen = {}

    def _record(*args):
        seen["seq_lens"] = args[10]

    backend.decode_attention_fwd = _record

    qkv = torch.zeros(num_tokens, HEAD_NUM * HEAD_DIM)
    forward_batch = _forward_batch(ForwardMode.DECODE, num_tokens)
    backend.forward_decode(qkv, qkv, qkv, _layer(), forward_batch)
    return seen["seq_lens"]


class TestIntelAMXVerifySeqLens(unittest.TestCase):
    def test_verify_seq_lens_cover_the_draft_kv(self):
        """A verify batch writes its draft KV into
        [seq_lens, seq_lens + draft_token_num) without bumping seq_lens, so the
        kernel has to be told the extended length. forward_extend used to
        overwrite the verify-adjusted value with forward_batch.seq_lens, which
        hid every draft KV entry and made each draft token attend only to the
        committed prefix."""
        num_tokens = BATCH_SIZE * DRAFT_TOKEN_NUM
        forward_batch = _forward_batch(ForwardMode.TARGET_VERIFY, num_tokens)

        seq_lens = _run_forward_extend(forward_batch, num_tokens)

        self.assertTrue(
            torch.equal(
                seq_lens,
                torch.full(
                    (BATCH_SIZE,), COMMITTED_LEN + DRAFT_TOKEN_NUM, dtype=torch.int64
                ),
            )
        )

    def test_extend_seq_lens_are_passed_through(self):
        """The verify adjustment must not leak into ordinary prefill, where
        seq_lens already accounts for the extended tokens."""
        num_tokens = BATCH_SIZE * EXTEND_LEN
        forward_batch = _forward_batch(ForwardMode.EXTEND, num_tokens)

        seq_lens = _run_forward_extend(forward_batch, num_tokens)

        self.assertTrue(
            torch.equal(
                seq_lens, torch.full((BATCH_SIZE,), COMMITTED_LEN, dtype=torch.int64)
            )
        )

    def test_draft_decode_uses_the_expanded_seq_lens(self):
        """The multi-step draft backend hands each step an expanded
        (req_to_token, seq_lens, req_pool_indices) triple sized
        batch_size * topk. forward_decode used to keep the expanded
        req_to_token and req_pool_indices but replace the expanded seq_lens
        with forward_batch.seq_lens, so the kernel derived num_seqs from the
        unexpanded tensor and aborted on its attn_logits size check."""
        expanded = (
            torch.full((BATCH_SIZE,), COMMITTED_LEN, dtype=torch.int32)
        ).repeat_interleave(DRAFT_TOPK) + 1

        seq_lens = _run_forward_decode(
            (
                torch.zeros(BATCH_SIZE * DRAFT_TOPK, 64, dtype=torch.int32),
                expanded,
                torch.arange(BATCH_SIZE * DRAFT_TOPK, dtype=torch.int64),
            )
        )

        self.assertTrue(torch.equal(seq_lens, expanded.to(torch.int64)))

    def test_plain_decode_falls_back_to_the_batch_seq_lens(self):
        """Without draft metadata there is nothing to expand, so the batch's
        own lengths must still be used."""
        seq_lens = _run_forward_decode(None)

        self.assertTrue(
            torch.equal(
                seq_lens, torch.full((BATCH_SIZE,), COMMITTED_LEN, dtype=torch.int64)
            )
        )


if __name__ == "__main__":
    unittest.main()
