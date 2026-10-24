# SPDX-License-Identifier: Apache-2.0
"""CPU checks of when the EAGLE draft worker binds the target's embedding and lm_head:
under embedding replication before the scheduler sizes the KV pool (so the draft's own
full table is freed first), otherwise in ``alloc_memory_pool`` as before."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run(replicate: bool) -> list:
    """The order of the draft worker's calls from the end of ``__init__`` through
    ``alloc_memory_pool``."""
    calls = []
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.draft_worker = MagicMock()
    worker.draft_worker.alloc_memory_pool.side_effect = lambda **_: calls.append(
        "draft_pool"
    )
    with (
        envs.SGLANG_ENABLE_EMBED_REPLICATION.override(replicate),
        patch.object(
            EagleDraftWorker, "init_token_map", lambda self: calls.append("token_map")
        ),
        patch.object(
            EagleDraftWorker, "init_lm_head", lambda self: calls.append("lm_head")
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ),
    ):
        worker.target_embed_and_head_shared = False
        worker._maybe_share_embed_before_pool_sizing()
        calls.append("target_pool")
        worker.alloc_memory_pool()
    return calls


class TestEagleDraftEmbedShare(CustomTestCase):
    def test_replicated_embedding_is_shared_before_the_pool_is_sized(self):
        self.assertEqual(
            _run(True), ["token_map", "lm_head", "target_pool", "draft_pool"]
        )

    def test_otherwise_it_is_shared_when_the_draft_pool_is_allocated(self):
        self.assertEqual(
            _run(False), ["target_pool", "draft_pool", "token_map", "lm_head"]
        )

    def test_init_ends_with_the_early_share(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(EagleDraftWorker.__init__)))
        last = tree.body[0].body[-1]
        self.assertEqual(
            ast.unparse(last), "self._maybe_share_embed_before_pool_sizing()"
        )


if __name__ == "__main__":
    unittest.main()
