import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.speculative import eagle_megakernel
from sglang.srt.speculative.eagle_megakernel import init_eagle_draft_runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestEagleLifecycle(unittest.TestCase):
    def test_stock_and_nvidia_keep_their_strategy(self):
        worker = NS(draft_runner=NS())
        nvidia = object.__new__(eagle_megakernel.EagleMegakernel)
        for specialized in (None, nvidia):
            with patch.object(
                eagle_megakernel, "init_eagle_megakernel", return_value=specialized
            ) as resolve:
                runtime = init_eagle_draft_runtime(draft_worker=worker)
                resolve.assert_called_once_with(draft_worker=worker)
            self.assertIs(
                runtime, specialized or eagle_megakernel.STOCK_EAGLE_DRAFT_RUNTIME
            )


if __name__ == "__main__":
    unittest.main()
