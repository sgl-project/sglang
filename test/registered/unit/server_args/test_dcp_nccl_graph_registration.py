"""Unit tests for the DCP a2a NCCL graph buffer registration policy."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import parallel_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cpu_ci(est_time=1, suite="stage-b-test-cpu-intel")


class TestDcpNcclGraphRegistration(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("NCCL_GRAPH_REGISTER", None)
        self.resolve_backend = self.enterContext(
            patch.object(parallel_hook, "run_post_process_pass")
        )
        self.enterContext(
            patch.object(parallel_hook, "resolving_view", side_effect=lambda args: args)
        )
        self.enterContext(
            patch.object(
                parallel_hook,
                "get_platform",
                return_value=SimpleNamespace(is_cuda=True),
            )
        )

    def _resolve(self, backend="a2a", dcp_size=8):
        cfg = SimpleNamespace(
            dcp_size=dcp_size,
            dcp_comm_backend=backend,
            enable_prefill_cp=False,
            dcp_replicate_q_proj=False,
        )
        parallel_hook.handle_decode_context_parallelism(cfg)

    def test_resolved_a2a_disables_graph_registration(self):
        self.resolve_backend.side_effect = lambda args, _: setattr(
            args, "dcp_comm_backend", "a2a"
        )
        self._resolve(backend=None)
        self.assertEqual(os.environ.get("NCCL_GRAPH_REGISTER"), "0")

    def test_explicit_registration_setting_is_preserved(self):
        for value, warns in (("0", False), ("1", True)):
            with self.subTest(value=value):
                os.environ["NCCL_GRAPH_REGISTER"] = value
                with patch.object(parallel_hook.logger, "warning") as warning:
                    self._resolve()
                self.assertEqual(os.environ["NCCL_GRAPH_REGISTER"], value)
                if warns:
                    warning.assert_called_once()
                else:
                    warning.assert_not_called()

    def test_non_a2a_backends_leave_env_untouched(self):
        for backend, dcp_size in (("fi_a2a", 8), ("ag_rs", 8), ("ag_rs", 1)):
            with self.subTest(backend=backend, dcp_size=dcp_size):
                self._resolve(backend=backend, dcp_size=dcp_size)
                self.assertNotIn("NCCL_GRAPH_REGISTER", os.environ)

    def test_invalid_a2a_leaves_env_untouched(self):
        with self.assertRaisesRegex(ValueError, "requires --dcp-size"):
            self._resolve(dcp_size=1)
        self.assertNotIn("NCCL_GRAPH_REGISTER", os.environ)


if __name__ == "__main__":
    unittest.main()
