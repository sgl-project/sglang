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
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 1000
NUM_TOKENS = 384


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


def _ids_in_run_ctx(buffers: SimpleNamespace) -> torch.Tensor:
    """Enter the autotune dummy run context; return the ids a forward would see."""
    model_runner = SimpleNamespace(
        canary_manager=None,
        model_config=SimpleNamespace(vocab_size=VOCAB_SIZE),
    )
    with autotune.autotune_dummy_run_ctx(model_runner, buffers):
        return buffers.input_ids.clone()


def _buffers() -> SimpleNamespace:
    return SimpleNamespace(input_ids=torch.zeros(NUM_TOKENS, dtype=torch.int64))


class TestAutotuneDummyRunTokenIds(CustomTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(_rank_of_two_gpu_tp(tp_rank=0, attn_dp_size=1))

    def test_autotune_dummy_feeds_varied_in_range_ids_then_restores(self):
        buffers = _buffers()
        ids = _ids_in_run_ctx(buffers)
        self.assertGreaterEqual(int(ids.min()), 0)
        self.assertLess(int(ids.max()), VOCAB_SIZE)
        # 384 uniform draws from 1000 ids: ~320 distinct expected.
        self.assertGreater(ids.unique().numel(), 250)
        self.assertEqual(int(buffers.input_ids.abs().sum()), 0)

    def test_tp_ranks_sharing_a_batch_feed_the_same_ids(self):
        # They run one batch; their tuning timings are reduced together.
        runs = []
        for tp_rank in (0, 1):
            with _rank_of_two_gpu_tp(tp_rank=tp_rank, attn_dp_size=1):
                runs.append(_ids_in_run_ctx(_buffers()))
        self.assertTrue(torch.equal(runs[0], runs[1]))

    def test_attention_dp_ranks_feed_distinct_ids(self):
        # Their rows are gathered for the MoE; equal rows would repeat each route.
        runs = []
        for tp_rank in (0, 1):
            with _rank_of_two_gpu_tp(tp_rank=tp_rank, attn_dp_size=2):
                runs.append(_ids_in_run_ctx(_buffers()))
        self.assertFalse(torch.equal(runs[0], runs[1]))

    def test_kill_switch_keeps_the_constant_token_dummy(self):
        with envs.SGLANG_FLASHINFER_AUTOTUNE_VARIED_TOKENS.override(False):
            ids = _ids_in_run_ctx(_buffers())
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
