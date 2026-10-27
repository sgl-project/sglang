"""DSpark draft query width on the FlashAttention backend (sgl-project/sglang#46476).

For DSpark draft checkpoints with ``sample_from_anchor=False`` the draft
forwards gamma + 1 rows per request (anchor + gamma draft positions), while
the generic target-verify helper returns ``num_draft_tokens - 1``. The
FlashAttention backend must therefore size its per-request query width from
``ModelRunner.decode_num_tokens_per_req()`` -- the shared source of truth
the Triton backend already uses -- not from the target-verify helper, or
``cu_seqlens_q`` under-covers the flattened query tensor whenever batch
size > 1 (e.g. gamma=7, batch 2: [0, 7, 14] instead of [0, 8, 16]).

CPU-only: the width derivation and the eager target-verify metadata are
pure shape arithmetic; no kernel is launched. The CUDA-graph metadata
builds its ``cu_seqlens_q`` from the same ``speculative_num_draft_tokens``
field, so it inherits the corrected width.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention import flashattention_backend as fab
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Issue example: gamma = 7, so speculative_num_draft_tokens = gamma + 1.
_GAMMA = 7
_NUM_DRAFT_TOKENS = _GAMMA + 1


def _decode_runner(algorithm, *, is_draft_worker, sample_from_anchor):
    runner = ModelRunner.__new__(ModelRunner)
    runner.spec_algorithm = algorithm
    runner.is_draft_worker = is_draft_worker
    runner.model_config = SimpleNamespace(
        hf_config={"sample_from_anchor": sample_from_anchor}
    )
    return runner


class TestDsparkDraftWidthSourceOfTruth(CustomTestCase):
    """ModelRunner.decode_num_tokens_per_req() is the draft forward width."""

    def test_dspark_draft_width_follows_sample_from_anchor(self):
        # resolve_num_tokens_per_req() reads the spec bag only when the
        # width is not passed explicitly; the anchor=True path goes through
        # it, so publish a bag rather than depending on runtime context.
        with patch.object(
            spec_utils,
            "get_spec",
            return_value=SimpleNamespace(
                speculative_num_draft_tokens=_NUM_DRAFT_TOKENS
            ),
        ):
            # sample_from_anchor=False: draft forwards gamma + 1 rows.
            self.assertEqual(
                _decode_runner(
                    SpeculativeAlgorithm.DSPARK,
                    is_draft_worker=True,
                    sample_from_anchor=False,
                ).decode_num_tokens_per_req(num_draft_tokens=_NUM_DRAFT_TOKENS),
                _NUM_DRAFT_TOKENS,
            )
            # sample_from_anchor=True: draft forwards gamma rows.
            self.assertEqual(
                _decode_runner(
                    SpeculativeAlgorithm.DSPARK,
                    is_draft_worker=True,
                    sample_from_anchor=True,
                ).decode_num_tokens_per_req(num_draft_tokens=_NUM_DRAFT_TOKENS),
                _GAMMA,
            )

    def test_target_workers_and_other_algorithms_unchanged(self):
        with patch.object(
            spec_utils,
            "get_spec",
            return_value=SimpleNamespace(
                speculative_num_draft_tokens=_NUM_DRAFT_TOKENS
            ),
        ):
            # The target-verify helper returns num_draft_tokens unchanged
            # for target workers and non-DSpark algorithms.
            for algorithm, is_draft_worker in (
                (SpeculativeAlgorithm.DSPARK, False),
                (SpeculativeAlgorithm.EAGLE, True),
                (SpeculativeAlgorithm.EAGLE, False),
            ):
                self.assertEqual(
                    _decode_runner(
                        algorithm,
                        is_draft_worker=is_draft_worker,
                        sample_from_anchor=False,
                    ).decode_num_tokens_per_req(num_draft_tokens=_NUM_DRAFT_TOKENS),
                    _NUM_DRAFT_TOKENS,
                )


def _draft_worker_runner(*, is_draft_worker, decode_width):
    req_to_token = torch.zeros((16, 64), dtype=torch.int32)
    return SimpleNamespace(
        sliding_window_size=None,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            context_len=64,
            attention_arch=AttentionArch.MHA,
            is_local_attention_model=False,
            head_dim=128,
            hf_text_config=SimpleNamespace(
                num_attention_heads=4, attn_logit_softcapping=None
            ),
            get_num_kv_heads=lambda _: 1,
        ),
        device="cpu",
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token, size=16),
        token_to_kv_pool=object(),
        kv_cache_dtype=torch.float16,
        kv_cache_dtype_str="auto",
        page_size=1,
        is_draft_worker=is_draft_worker,
        kv_index_translator=SimpleNamespace(is_translating=False),
        decode_num_tokens_per_req=Mock(return_value=decode_width),
    )


class TestFlashAttentionDraftWorkerWidthWiring(CustomTestCase):
    """The draft-worker backend takes its width from decode_num_tokens_per_req.

    The mock deliberately returns the anchor=False width (8); the old
    target-verify derivation would have produced 7 for DSpark, so the
    assertion pins the wiring, not just the value.
    """

    def _backend(self, *, is_draft_worker, decode_width):
        runner = _draft_worker_runner(
            is_draft_worker=is_draft_worker, decode_width=decode_width
        )
        spec = SimpleNamespace(
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=_NUM_DRAFT_TOKENS,
            speculative_algorithm="DSPARK",
        )
        parallel = SimpleNamespace(
            attn_cp_size=1,
            attn_tp_size=1,
            enable_prefill_cp=False,
            attn_dp_enabled=False,
        )
        with (
            patch.object(fab, "get_spec", return_value=spec),
            patch.object(fab, "get_parallel", return_value=parallel),
            patch.object(fab, "get_device_capability", return_value=(9, 0)),
            patch.object(
                fab,
                "get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(enable_deterministic_inference=False)
                ),
            ),
            patch.object(
                fab, "get_model", return_value=SimpleNamespace(is_embedding=False)
            ),
            patch.object(
                fab,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=8192),
            ),
            patch.object(
                fab,
                "get_memory",
                return_value=SimpleNamespace(disable_radix_cache=False),
            ),
        ):
            backend = fab.FlashAttentionBackend(runner)
        return runner, backend

    def test_draft_worker_uses_decode_width(self):
        runner, backend = self._backend(is_draft_worker=True, decode_width=8)
        self.assertEqual(backend.speculative_num_draft_tokens, 8)
        runner.decode_num_tokens_per_req.assert_called_once_with(
            num_draft_tokens=_NUM_DRAFT_TOKENS
        )

    def test_target_worker_width_untouched(self):
        runner, backend = self._backend(is_draft_worker=False, decode_width=8)
        self.assertEqual(backend.speculative_num_draft_tokens, _NUM_DRAFT_TOKENS)
        runner.decode_num_tokens_per_req.assert_not_called()


def _target_verify_metadata(width, batch_size):
    backend = fab.FlashAttentionBackend.__new__(fab.FlashAttentionBackend)
    backend.topk = 1
    backend.speculative_num_draft_tokens = width
    backend.max_context_len = 64
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((batch_size, 128), dtype=torch.int32)
    )
    backend.kv_index_translator = SimpleNamespace(is_translating=False)
    backend.use_sliding_window_kv_pool = False
    backend.page_size = 1
    backend._maybe_init_local_attn_metadata = lambda *_: None
    seq_lens = torch.tensor([10 + i for i in range(batch_size)], dtype=torch.int64)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=batch_size,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
        spec_info=SimpleNamespace(ragged_verify_layout=None),
        encoder_lens=None,
    )
    backend.init_forward_metadata(forward_batch)
    return backend.forward_metadata


class TestFlashAttentionDsparkCuSeqlensQ(CustomTestCase):
    """cu_seqlens_q must cover the flattened draft query tensor exactly."""

    def test_anchor_false_width_covers_batches_2_and_8(self):
        for batch_size in (2, 8):
            metadata = _target_verify_metadata(_NUM_DRAFT_TOKENS, batch_size)
            self.assertEqual(metadata.max_seq_len_q, _NUM_DRAFT_TOKENS)
            self.assertEqual(
                metadata.cu_seqlens_q.tolist(),
                list(range(0, batch_size * _NUM_DRAFT_TOKENS + 1, _NUM_DRAFT_TOKENS)),
            )
            # The last boundary is the flattened query length: bs * width
            # rows, one per (request, draft position) pair.
            self.assertEqual(
                metadata.cu_seqlens_q[-1].item(),
                batch_size * _NUM_DRAFT_TOKENS,
            )

    def test_anchor_true_width_preserved(self):
        for batch_size in (2, 8):
            metadata = _target_verify_metadata(_GAMMA, batch_size)
            self.assertEqual(metadata.max_seq_len_q, _GAMMA)
            self.assertEqual(
                metadata.cu_seqlens_q.tolist(),
                list(range(0, batch_size * _GAMMA + 1, _GAMMA)),
            )


if __name__ == "__main__":
    unittest.main()
