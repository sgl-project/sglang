"""Unit tests for the YOCO fast-prefill split in Gemma4TextModel.

The full forward path needs CUDA and a real Gemma4 checkpoint, so these
tests cover the eligibility gates and the per-request last-token index
math on CPU, with a stubbed ForwardBatch and attention backend.
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

import torch

from sglang.srt.models import gemma4_causal as gemma4_causal_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeForwardMode:
    def is_extend_without_speculative(self):
        return True


class _DecodeForwardMode(_FakeForwardMode):
    def is_extend_without_speculative(self):
        return False


class _FakeAttnBackend:
    def __init__(self):
        self.init_calls: List[tuple] = []

    def init_forward_metadata(self, forward_batch):
        # Record what the model sees at each rebuild so tests can assert the
        # right truncation/restore happened.
        self.init_calls.append(
            (
                int(forward_batch.extend_seq_lens.sum().item()),
                int(forward_batch.extend_prefix_lens.sum().item()),
                list(forward_batch.extend_seq_lens_cpu),
            )
        )


def _make_fake_forward_batch(
    extend_seq_lens: List[int],
    seq_lens: Optional[List[int]] = None,
    *,
    return_logprob: bool = False,
    decode_only: bool = False,
):
    if seq_lens is None:
        seq_lens = list(extend_seq_lens)
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_seq_lens),
        extend_prefix_lens=torch.tensor(
            [s - e for s, e in zip(seq_lens, extend_seq_lens)],
            dtype=torch.int32,
        ),
        extend_prefix_lens_cpu=[s - e for s, e in zip(seq_lens, extend_seq_lens)],
        extend_logprob_start_lens_cpu=(
            [0] * len(extend_seq_lens) if return_logprob else None
        ),
        extend_num_tokens=sum(extend_seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        return_logprob=return_logprob,
        forward_mode=_DecodeForwardMode() if decode_only else _FakeForwardMode(),
    )


class _FakePPGroup:
    world_size = 1


def _make_fake_model(
    *,
    num_hidden_layers: int = 35,
    num_kv_shared_layers: int = 20,
    flag_enabled: bool = True,
    pp_world_size: int = 1,
    spec_enabled: bool = False,
):
    config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_kv_shared_layers=num_kv_shared_layers,
    )
    pp_group = _FakePPGroup()
    pp_group.world_size = pp_world_size
    fake = SimpleNamespace(
        config=config,
        pp_group=pp_group,
        # Mirror the static gate computed in Gemma4TextModel.__init__.
        first_kv_shared_layer_idx=num_hidden_layers - num_kv_shared_layers,
        fast_prefill_supported=(
            flag_enabled
            and num_kv_shared_layers > 0
            and pp_world_size == 1
            and not spec_enabled
        ),
    )
    cls = gemma4_causal_module.Gemma4TextModel
    for name in ("_enable_fast_prefill", "_yoco_truncate_to_last_tokens"):
        setattr(fake, name, getattr(cls, name).__get__(fake, type(fake)))
    return fake


class TestGemma4FastPrefillEligibility(CustomTestCase):
    def test_flag_enabled(self):
        fake = _make_fake_model()
        fb = _make_fake_forward_batch([10, 5, 7])
        self.assertTrue(fake._enable_fast_prefill(fb))

    def test_flag_disabled(self):
        # Default: --kv-sharing-fast-prefill not set -> never eligible.
        fake = _make_fake_model(flag_enabled=False)
        fb = _make_fake_forward_batch([10, 5, 7])
        self.assertFalse(fake._enable_fast_prefill(fb))

    def test_no_kv_shared_layers(self):
        fake = _make_fake_model(num_kv_shared_layers=0)
        fb = _make_fake_forward_batch([10, 5, 7])
        self.assertFalse(fake._enable_fast_prefill(fb))

    def test_pipeline_parallel(self):
        fake = _make_fake_model(pp_world_size=2)
        fb = _make_fake_forward_batch([10, 5, 7])
        self.assertFalse(fake._enable_fast_prefill(fb))

    def test_speculative_decoding(self):
        # Spec decoding (EAGLE3, FROZEN_KV_MTP) captures hidden states we
        # would otherwise drop.
        fake = _make_fake_model(spec_enabled=True)
        fb = _make_fake_forward_batch([10, 5, 7])
        self.assertFalse(fake._enable_fast_prefill(fb))

    def test_pure_decode_batch(self):
        # All requests have a single new token -> nothing to truncate.
        fake = _make_fake_model()
        fb = _make_fake_forward_batch([1, 1, 1])
        self.assertFalse(fake._enable_fast_prefill(fb))

    def test_decode_forward_mode(self):
        fake = _make_fake_model()
        fb = _make_fake_forward_batch([10], decode_only=True)
        self.assertFalse(fake._enable_fast_prefill(fb))

    def test_prompt_logprobs_disable(self):
        # extend_logprob_start_lens_cpu = [0, 0] => prompt logprobs requested.
        fake = _make_fake_model()
        fb = _make_fake_forward_batch([10, 5], return_logprob=True)
        self.assertFalse(fake._enable_fast_prefill(fb))


class TestGemma4YocoTruncate(CustomTestCase):
    def test_truncate_indices_and_restore(self):
        fake = _make_fake_model()
        fb = _make_fake_forward_batch(extend_seq_lens=[3, 4, 2], seq_lens=[3, 4, 2])

        fake_backend = _FakeAttnBackend()
        with patch.object(
            gemma4_causal_module, "get_attn_backend", lambda: fake_backend
        ):
            hidden = (
                torch.arange(3 + 4 + 2, dtype=torch.float32).unsqueeze(-1).repeat(1, 8)
            )
            positions = torch.arange(9, dtype=torch.int64)
            per_layer = torch.zeros(9, 35, 16)

            (
                h_t,
                p_t,
                ple_t,
                last_indices,
                restore_fn,
            ) = fake._yoco_truncate_to_last_tokens(fb, hidden, positions, per_layer)

            # last_indices = cumsum([3, 4, 2]) - 1 = [2, 6, 8]
            self.assertEqual(last_indices.tolist(), [2, 6, 8])
            self.assertEqual(h_t.shape, (3, 8))
            self.assertTrue(torch.equal(h_t[:, 0], torch.tensor([2.0, 6.0, 8.0])))
            self.assertEqual(p_t.tolist(), [2, 6, 8])
            self.assertEqual(ple_t.shape, (3, 35, 16))

            # forward_batch is now truncated to 1 new token per request.
            self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1])
            self.assertEqual(fb.extend_prefix_lens.tolist(), [2, 3, 1])
            self.assertEqual(fb.extend_seq_lens_cpu, [1, 1, 1])
            self.assertEqual(fb.extend_num_tokens, 3)
            self.assertEqual(len(fake_backend.init_calls), 1)
            self.assertEqual(fake_backend.init_calls[0], (3, 6, [1, 1, 1]))

            # Restore puts the original values back and rebuilds again.
            restore_fn()
            self.assertEqual(fb.extend_seq_lens.tolist(), [3, 4, 2])
            self.assertEqual(fb.extend_prefix_lens.tolist(), [0, 0, 0])
            self.assertEqual(fb.extend_seq_lens_cpu, [3, 4, 2])
            self.assertEqual(fb.extend_num_tokens, 9)
            self.assertEqual(len(fake_backend.init_calls), 2)
            self.assertEqual(fake_backend.init_calls[1], (9, 0, [3, 4, 2]))


if __name__ == "__main__":
    unittest.main()
