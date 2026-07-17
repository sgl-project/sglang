"""Unit tests for the MoE-LoRA DP-gathered tail stamping in _add_moe_lora_info.

Under --enable-dp-attention the MoE runs on DP-GATHERED tokens, so the per-token
LoRA mapping must cover [0, moe_num_tokens) while the per-rank segments only fill
[0, num_tokens). These tests pin the Tier-1 (single loaded adapter) semantics of
the gathered tail [num_tokens, moe_num_tokens):

  * active rank (local requests use the single adapter, ``_single_loaded_buffer_id``
    armed): tail stamped with that adapter's buffer id;
  * true idle rank (num_tokens == 0, ``_idle_rank_active_buffer_id`` armed): the
    whole mapping stamped and the adapter enabled;
  * base-only local batches, multi-adapter batches, and stale idle stamps on
    token-bearing batches: tail stays -1 (adapter-disabled) — foreign tokens are
    never given a delta this rank cannot attribute.

The backend object is stubbed (only the fields _add_moe_lora_info reads are
populated) so the tests run hermetically without a server or dist groups; the
gathered length is forced via forward_batch.global_dp_buffer_len.

Usage:
    python -m pytest test/registered/unit/lora/test_moe_lora_tail_stamp.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GATHERED = 16  # forced gathered length (> any local num_tokens used below)
# buffer slots: 0 = the None/base uid slot (rank 0), 1/2 = adapters (rank 8), 3 = free
RANKS = [0, 8, 8, 0]


def _forward_batch(num_tokens: int):
    mode = SimpleNamespace(
        is_extend=lambda: False,
        is_idle=lambda: num_tokens == 0,
        is_cuda_graph=lambda: False,
    )
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=num_tokens,  # decode: 1 token per seq
        extend_seq_lens_cpu=None,
        global_dp_buffer_len=GATHERED,
        global_num_tokens_cpu=[num_tokens, GATHERED - num_tokens],
    )


def _batch_info(weight_indices, num_tokens):
    bs = len(weight_indices)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=torch.arange(bs + 1, dtype=torch.int32, device=DEVICE),
        weight_indices=torch.tensor(weight_indices, dtype=torch.int32, device=DEVICE),
        lora_ranks=torch.tensor(RANKS, dtype=torch.int64, device=DEVICE),
        scalings=torch.ones(len(RANKS), dtype=torch.float, device=DEVICE),
        max_len=1,
        seg_lens=torch.ones(bs, dtype=torch.int32, device=DEVICE),
        permutation=None,
        expected_tokens=num_tokens,
    )


def _run(weight_indices, num_tokens, idle_bid=None, single_bid=None):
    stub = SimpleNamespace(
        is_moe_lora=True,
        _idle_rank_active_buffer_id=idle_bid,
        _single_loaded_buffer_id=single_bid,
    )
    out = BaseLoRABackend._add_moe_lora_info(
        stub, _forward_batch(num_tokens), _batch_info(weight_indices, num_tokens)
    )
    info = out.moe_lora_info
    return info.token_lora_mapping.tolist(), info.adapter_enabled.tolist()


class TestMoELoRATailStamp(unittest.TestCase):
    def test_base_only_local_batch_keeps_disabled_tail(self):
        # All local tokens on the base (None-uid) slot, nothing armed: the
        # gathered tail must stay -1 and no adapter may be enabled.
        mapping, enabled = _run([0, 0, 0, 0], num_tokens=4)
        self.assertTrue(all(x == -1 for x in mapping[4:]), mapping)
        self.assertEqual(enabled, [0, 0, 0, 0])

    def test_stale_idle_stamp_ignored_on_token_bearing_batch(self):
        # Defense in depth: an idle stamp must only be consumed when the rank is
        # truly idle (num_tokens == 0), never on a batch with local tokens.
        mapping, enabled = _run([0, 0, 0, 0], num_tokens=4, idle_bid=2)
        self.assertTrue(all(x == -1 for x in mapping[4:]), mapping)
        self.assertEqual(enabled[2], 0)

    def test_multi_adapter_batch_keeps_disabled_tail(self):
        # Two adapters used locally: the foreign tokens' adapter identity is
        # unknowable host-side, so the tail must stay -1 (no mis-stamping with
        # whatever the last local token happened to use).
        mapping, _ = _run([1, 1, 2, 2], num_tokens=4)
        self.assertTrue(all(x == -1 for x in mapping[4:]), mapping)

    def test_idle_rank_stamps_whole_mapping_and_enables_adapter(self):
        # True idle rank under Tier-1: the whole gathered mapping is stamped with
        # the single loaded adapter and the adapter is enabled, so foreign tokens
        # routed to this rank's experts get the LoRA delta.
        mapping, enabled = _run([], num_tokens=0, idle_bid=2)
        self.assertTrue(all(x == 2 for x in mapping), mapping)
        self.assertEqual(enabled[2], 1)

    def test_active_rank_stamps_tail_with_single_adapter(self):
        # Local tokens actively use the single loaded adapter: the tail is
        # stamped with its buffer id (not a copy of an arbitrary local slot).
        mapping, enabled = _run([2, 2], num_tokens=2, single_bid=2)
        self.assertTrue(all(x == 2 for x in mapping[2:]), mapping)
        self.assertEqual(enabled[2], 1)

    def test_single_adapter_loaded_but_local_base_keeps_disabled_tail(self):
        # An adapter is loaded but this rank's local tokens are all base (the
        # manager arms no stamp in this case): tail stays -1.
        mapping, enabled = _run([0, 0], num_tokens=2)
        self.assertTrue(all(x == -1 for x in mapping[2:]), mapping)
        self.assertEqual(enabled, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
