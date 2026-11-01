"""``ExpertLocationDispatchInfo.init_new`` must not require metadata it discards.

With no ``ep_dispatch_algorithm`` — the default — ``init_new`` returns ``None``
before reading the global expert-location metadata. A model class that does not
define ``get_model_config_for_expert_location`` leaves that metadata ``None``,
and asserting on it first turned EPLB-off serving into a bare ``AssertionError``.
When an algorithm *is* configured the metadata is genuinely required, and the
assert names the missing hook.
"""

import unittest
from unittest.mock import patch

from sglang.srt import runtime_context as rc
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestInitNewWithoutMetadata(CustomTestCase):
    def setUp(self):
        rc.reset_context()
        rc.publish(ServerArgs(model_path="dummy"), role="test")
        # The state a model class without the expert-location hook leaves behind.
        self._patch = patch.object(
            expert_location_dispatch,
            "get_global_expert_location_metadata",
            lambda: None,
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        rc.reset_context()

    def test_returns_none_without_dispatch_algorithm(self):
        self.assertIsNone(rc.get_exec().moe.ep_dispatch_algorithm)
        self.assertIsNone(ExpertLocationDispatchInfo.init_new(layer_id=0))

    def test_assert_names_the_hook_when_an_algorithm_is_set(self):
        rc.get_context().override("test", ep_dispatch_algorithm="static")
        with self.assertRaises(AssertionError) as ctx:
            ExpertLocationDispatchInfo.init_new(layer_id=0)
        self.assertIn("get_model_config_for_expert_location", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
