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

    def test_a2a_disables_graph_registration_by_default(self):
        self._resolve()
        self.assertEqual(os.environ.get("NCCL_GRAPH_REGISTER"), "0")

    def test_guard_uses_the_resolved_backend(self):
        def resolve(args, _):
            args.dcp_comm_backend = "a2a"

        self.resolve_backend.side_effect = resolve
        self._resolve(backend=None)
        self.assertEqual(os.environ.get("NCCL_GRAPH_REGISTER"), "0")

    def test_explicit_registration_opt_out_is_preserved(self):
        os.environ["NCCL_GRAPH_REGISTER"] = "0"
        with patch.object(parallel_hook.logger, "warning") as warning:
            self._resolve()
        self.assertEqual(os.environ["NCCL_GRAPH_REGISTER"], "0")
        warning.assert_not_called()

    def test_explicit_registration_opt_in_is_preserved_with_warning(self):
        os.environ["NCCL_GRAPH_REGISTER"] = "1"
        with self.assertLogs(parallel_hook.logger, level="WARNING") as logs:
            self._resolve()
        self.assertEqual(os.environ["NCCL_GRAPH_REGISTER"], "1")
        self.assertIn("NCCL_GRAPH_REGISTER=1", "\n".join(logs.output))
        self.assertIn("DCP a2a", "\n".join(logs.output))

    def test_other_backends_do_not_set_registration(self):
        for backend, size in (("fi_a2a", 8), ("ag_rs", 8), ("ag_rs", 1)):
            with self.subTest(backend=backend, dcp_size=size):
                self._resolve(backend=backend, dcp_size=size)
                self.assertNotIn("NCCL_GRAPH_REGISTER", os.environ)

    def test_invalid_a2a_dcp_size_does_not_set_registration(self):
        with self.assertRaisesRegex(ValueError, "requires --dcp-size"):
            self._resolve(dcp_size=1)
        self.assertNotIn("NCCL_GRAPH_REGISTER", os.environ)


if __name__ == "__main__":
    unittest.main()
