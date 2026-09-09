"""Model packages use one extension for server arguments and worker startup."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRustServerExtension(CustomTestCase):
    def test_launch_uses_the_model_extension_and_instance_worker_state(self):
        extension = ModuleType("model_server")
        extension.Server = Mock()

        class ModelServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

            def _start_multimodal(self, scheduler):
                self.server.start_mm_workers(sentinel.spec, 8)

        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_size=2, attn_dp_rank=1),
            model_config=SimpleNamespace(is_multimodal=True),
        )
        with (
            patch.object(ModelServer, "_partition_cores", return_value=(None, None)),
            patch.object(
                server_module,
                "get_mm",
                return_value=SimpleNamespace(mm_processor_worker_num=8),
            ),
            patch.object(
                server_module,
                "get_serving",
                return_value=SimpleNamespace(host="::", port=30000),
            ),
            patch.object(
                server_module, "_build_server_args", return_value=sentinel.args
            ) as build_args,
        ):
            instance = ModelServer.launch(scheduler)

        build_args.assert_called_once_with(scheduler, extension=extension)
        extension.Server.assert_called_once_with(
            sentinel.args, cores=None, port_offset=1
        )
        instance.server.start_mm_workers.assert_called_once_with(sentinel.spec, 8)
        self.assertIsInstance(instance, ModelServer)
        self.assertEqual(instance.http_port, 30001)
        self.assertTrue(instance._multimodal_enabled)


if __name__ == "__main__":
    unittest.main()
