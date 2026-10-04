"""The draft must bind the target embed/lm_head BEFORE the pool is profiled.

MTP draft classes allocate their own embedding + lm_head shell; the
target-tensor rebinding in ``set_embed_and_head`` (``del`` + empty_cache) is
what makes that memory reclaimable. The scheduler sizes the KV pool from the
free-memory baseline captured at import time minus the *currently resident*
weights (``account_preloaded_weights``), so if the rebinding only runs at the
tail of ``alloc_memory_pool()`` the pool is sized against the shell-inflated
footprint and collapses (Qwen3.8-27B-NVFP4 + NEXTN 2/1/3 on 2x16GB, kv-cache-dtype
nvfp4: 51,456 tokens instead of 195,968). ``EagleDraftWorker.__init__``
therefore performs the bind itself, right after the draft runner is built;
the calls in ``alloc_memory_pool()`` remain (idempotent rebind of the same
tensors).
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _noop_ctx(*args, **kwargs):
    yield


class TestEmbedBindBeforePoolProfiling(unittest.TestCase):
    def test_construction_binds_embed_and_head(self):
        calls = []

        class FakeTpModelWorker:
            def __init__(self, **kwargs):
                self.model_runner = SimpleNamespace(
                    tp_group=object(), model_config=SimpleNamespace(context_len=8192)
                )

        def fake_init_token_map(self):
            calls.append("init_token_map")

        def fake_init_lm_head(self):
            calls.append("init_lm_head")

        server_args = SimpleNamespace()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=8192)
            ),
            random_seed=1,
        )

        M = "sglang.srt.speculative.eagle_worker_v2."
        with (
            patch(M + "TpModelWorker", FakeTpModelWorker),
            patch(M + "draft_tp_context", _noop_ctx),
            patch(M + "draft_pp_context", _noop_ctx),
            patch(M + "speculative_moe_backend_context", _noop_ctx),
            patch(M + "speculative_moe_a2a_backend_context", _noop_ctx),
            patch(M + "draft_model_build_scope", _noop_ctx),
            patch(M + "get_plan_stream", lambda device: (object(), _noop_ctx())),
            patch.object(EagleDraftWorker, "init_token_map", fake_init_token_map),
            patch.object(EagleDraftWorker, "init_lm_head", fake_init_lm_head),
            patch.object(
                EagleDraftWorker, "_init_dsa_index_share_state", lambda self: None
            ),
            patch.object(
                EagleDraftWorker, "_rebuild_topk1_chain_buffers", lambda self: None
            ),
        ):
            # __init__ reads the parallel/device/spec bags; a real override on
            # the context keeps them consistent without a GPU.
            override = get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_eagle_topk=1,
                speculative_num_steps=2,
                speculative_num_draft_tokens=3,
                speculative_use_rejection_sampling=False,
            )
            override.install()
            try:
                EagleDraftWorker(
                    server_args=server_args,
                    gpu_id=0,
                    nccl_port=0,
                    target_worker=target_worker,
                )
            finally:
                override.restore()

        self.assertEqual(
            calls,
            ["init_token_map", "init_lm_head"],
            "EagleDraftWorker.__init__ must bind the target embed/lm_head "
            "during construction; deferring to alloc_memory_pool() sizes the "
            "KV pool against the un-freed draft embedding/lm_head shell.",
        )


if __name__ == "__main__":
    unittest.main()
