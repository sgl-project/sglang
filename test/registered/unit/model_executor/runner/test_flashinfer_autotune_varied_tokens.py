"""The FlashInfer autotune dummy forward must run on varied token ids.

FlashInfer's MoE tuners profile every token bucket with the dummy forward's own
routed top-k rows. A constant-token dummy routes every row to the same experts,
so the tuner picks tactics for a load that serving never produces.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import autotune_dummy_run_mode
from sglang.srt.model_executor.runner import base_runner
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 1000
BATCH_SIZE = 384


def _buffers(max_tokens: int) -> SimpleNamespace:
    """Zero-filled static decode buffers, as the runners allocate them."""
    return SimpleNamespace(
        input_ids=torch.zeros(max_tokens, dtype=torch.int64),
        positions=torch.zeros(max_tokens, dtype=torch.int64),
        out_cache_loc=torch.zeros(max_tokens, dtype=torch.int64),
        mrope_positions=torch.zeros((3, max_tokens), dtype=torch.int64),
        num_token_non_padded=None,
        req_pool_indices=torch.zeros(max_tokens, dtype=torch.int64),
        seq_lens=torch.ones(max_tokens, dtype=torch.int64),
        seq_lens_cpu=torch.ones(max_tokens, dtype=torch.int64),
        next_token_logits_buffer=None,
        encoder_lens=None,
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
        custom_mask=None,
        rids_int=None,
        bootstrap_room_ids_int=None,
        ngram_embedding_info=None,
    )


def _rank_of_two_gpu_tp(*, tp_rank: int, attn_dp_size: int):
    """Topology of ``tp_rank`` in a TP2 / EP2 group split into ``attn_dp_size`` DP groups."""
    attn_tp_size = 2 // attn_dp_size
    return get_parallel().override(
        tp_size=2,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        dp_rank=None,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=tp_rank % attn_tp_size,
        attn_dp_size=attn_dp_size,
        attn_dp_rank=tp_rank // attn_tp_size,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_ep_size=2,
        moe_ep_rank=tp_rank,
        moe_dp_size=1,
        moe_tp_size=1,
        tp_group=Mock(),
    )


def _dummy_run(*, buffers: SimpleNamespace, in_autotune: bool) -> torch.Tensor:
    """Run ``BaseRunner._dummy_run`` on CPU stand-ins; return the ids the model saw."""
    seen = []
    model_runner = SimpleNamespace(
        device="cpu",
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        model_config=SimpleNamespace(vocab_size=VOCAB_SIZE, hidden_size=8),
        model=SimpleNamespace(
            forward=lambda input_ids, positions, forward_batch: seen.append(
                input_ids.clone()
            )
        ),
        lora_manager=None,
        runner_ids=Mock(),
        attn_backend=Mock(),
        prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
    )
    collaborators = dict(
        get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
        get_flags=lambda: SimpleNamespace(
            capture=SimpleNamespace(enable_torch_compile=False)
        ),
        get_exec=lambda: SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
        ),
        get_server_return_hidden_states_mode=Mock(),
        require_mlp_tp_gather=lambda: False,
        require_attn_tp_gather=lambda: False,
        require_gathered_buffer=lambda: False,
        enable_num_token_non_padded=lambda: False,
        create_dummy_verify_input=lambda *args: None,
        DpPaddingMode=Mock(),
        ForwardBatch=lambda **kwargs: SimpleNamespace(**kwargs),
        set_dp_buffer_len=Mock(),
        set_is_extend_in_batch=Mock(),
        forward_context=lambda *args, **kwargs: nullcontext(),
    )
    with ExitStack() as stack:
        stack.enter_context(patch.multiple(base_runner, **collaborators))
        if in_autotune:
            stack.enter_context(autotune_dummy_run_mode(run_lm_head=False))
        BaseRunner._dummy_run(
            SimpleNamespace(model_runner=model_runner),
            batch_size=BATCH_SIZE,
            buffers=buffers,
        )
    (input_ids,) = seen
    return input_ids


class TestAutotuneDummyRunTokenIds(CustomTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(_rank_of_two_gpu_tp(tp_rank=0, attn_dp_size=1))

    def test_autotune_dummy_feeds_varied_in_range_ids_then_restores(self):
        buffers = _buffers(512)
        buffers.input_ids[BATCH_SIZE:] = 7
        ids = _dummy_run(buffers=buffers, in_autotune=True)
        self.assertEqual(ids.shape, (BATCH_SIZE,))
        self.assertGreaterEqual(int(ids.min()), 0)
        self.assertLess(int(ids.max()), VOCAB_SIZE)
        # 384 uniform draws from 1000 ids: ~320 distinct expected.
        self.assertGreater(ids.unique().numel(), 250)
        # Restored, and the rows past the dummy were never written.
        self.assertEqual(int(buffers.input_ids[:BATCH_SIZE].abs().sum()), 0)
        self.assertTrue(bool((buffers.input_ids[BATCH_SIZE:] == 7).all()))

    def test_tp_ranks_sharing_a_batch_feed_the_same_ids(self):
        # They run one batch; their tuning timings are reduced together.
        runs = []
        for tp_rank in (0, 1):
            with _rank_of_two_gpu_tp(tp_rank=tp_rank, attn_dp_size=1):
                runs.append(_dummy_run(buffers=_buffers(BATCH_SIZE), in_autotune=True))
        self.assertTrue(torch.equal(runs[0], runs[1]))

    def test_attention_dp_ranks_feed_distinct_ids(self):
        # Their rows are gathered for the MoE; equal rows would repeat each route.
        runs = []
        for tp_rank in (0, 1):
            with _rank_of_two_gpu_tp(tp_rank=tp_rank, attn_dp_size=2):
                runs.append(_dummy_run(buffers=_buffers(BATCH_SIZE), in_autotune=True))
        self.assertFalse(torch.equal(runs[0], runs[1]))

    def test_kill_switch_keeps_the_constant_token_dummy(self):
        with envs.SGLANG_FLASHINFER_AUTOTUNE_VARIED_TOKENS.override(False):
            ids = _dummy_run(buffers=_buffers(BATCH_SIZE), in_autotune=True)
        self.assertEqual(int(ids.abs().sum()), 0)

    def test_other_dummy_runs_keep_the_static_buffer(self):
        ids = _dummy_run(buffers=_buffers(BATCH_SIZE), in_autotune=False)
        self.assertEqual(int(ids.abs().sum()), 0)


class TestAutotuneCacheKey(CustomTestCase):
    def test_varied_and_constant_token_tunes_use_separate_caches(self):
        # A cached shape skips profiling: caches tuned on the constant-token dummy
        # must retune, and the kill switch must not reuse varied-token tactics.
        runner = SimpleNamespace(
            device="cuda",
            dtype=torch.bfloat16,
            is_draft_worker=False,
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
        paths = {}
        with ExitStack() as stack, tempfile.TemporaryDirectory() as cache_dir:
            stack.enter_context(
                patch.multiple(
                    autotune,
                    get_model=lambda: SimpleNamespace(
                        model_path="m", quantization=None
                    ),
                    get_exec=lambda: SimpleNamespace(
                        moe=SimpleNamespace(moe_runner_backend="flashinfer_trtllm"),
                        kernel=SimpleNamespace(flashinfer_autotune_skip_ops=()),
                    ),
                )
            )
            stack.enter_context(
                patch.dict(sys.modules, flashinfer=SimpleNamespace(__version__="0"))
            )
            stack.enter_context(
                patch.object(torch.cuda, "get_device_capability", lambda _: (10, 3))
            )
            stack.enter_context(envs.SGLANG_CACHE_DIR.override(cache_dir))
            stack.enter_context(_rank_of_two_gpu_tp(tp_rank=0, attn_dp_size=1))
            for varied in (True, False):
                with envs.SGLANG_FLASHINFER_AUTOTUNE_VARIED_TOKENS.override(varied):
                    paths[varied] = autotune.flashinfer_autotune_cache_path(runner)
        self.assertNotEqual(paths[True].parent, paths[False].parent)


if __name__ == "__main__":
    unittest.main()
