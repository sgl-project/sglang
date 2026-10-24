"""The EAGLE/MTP draft must share target embed/head before KV sizing.

The draft model allocates its own embed_tokens/lm_head while it loads. If those
duplicates are still resident when the scheduler profiles free memory for the
target KV pool, the pool is permanently undersized. The worker therefore binds
them to the target in __init__ and only re-uses the bound state later.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.speculative import eagle_worker_v2
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestShareEmbedAndHead(CustomTestCase):
    def _worker(self):
        worker = object.__new__(EagleDraftWorker)
        worker._embed_and_head_shared = False
        worker.init_token_map = MagicMock()
        worker.init_lm_head = MagicMock()
        return worker

    def test_share_runs_only_once(self):
        worker = self._worker()
        order = []
        worker.init_token_map.side_effect = lambda: order.append("token_map")
        worker.init_lm_head.side_effect = lambda: order.append("lm_head")

        worker._share_embed_and_head()
        worker._share_embed_and_head()

        worker.init_token_map.assert_called_once_with()
        worker.init_lm_head.assert_called_once_with()
        self.assertEqual(order, ["token_map", "lm_head"])

    def test_alloc_memory_pool_reuses_shared_state(self):
        worker = self._worker()
        worker.draft_worker = MagicMock()
        worker._share_embed_and_head()

        spec = SimpleNamespace(speculative_use_rejection_sampling=False)
        with patch.object(eagle_worker_v2, "get_spec", return_value=spec):
            worker.alloc_memory_pool()

        worker.init_token_map.assert_called_once_with()
        worker.init_lm_head.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
