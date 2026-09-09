"""Auth configuration handoff for the embedded Rust HTTP frontend."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server import server as rust_server_module  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRustServerAuthConfig(CustomTestCase):
    def test_launch_passes_resolved_keys_only_to_server_constructor(self):
        server_constructor = Mock(return_value=sentinel.server)
        extension = SimpleNamespace(Server=server_constructor)
        serving = SimpleNamespace(
            host="127.0.0.1",
            port=31000,
            api_key="resolved-user-key",
            admin_api_key="resolved-admin-key",
        )
        scheduler = SimpleNamespace(
            # Deliberately different values: launch must consume the resolved
            # serving namespace, not re-read the mutable ServerArgs record.
            server_args=SimpleNamespace(
                host="stale-host",
                port=32000,
                api_key="stale-user-key",
                admin_api_key="stale-admin-key",
            ),
            ps=SimpleNamespace(dp_size=1),
            model_config=SimpleNamespace(is_multimodal=False),
        )

        with (
            patch(
                "sglang.srt.rust_extensions.load_rust_extension",
                return_value=extension,
            ),
            patch.object(rust_server_module, "get_serving", return_value=serving),
            patch.object(
                rust_server_module, "_partition_cores", return_value=(None, None)
            ),
            patch.object(
                rust_server_module,
                "_build_server_args",
                return_value=sentinel.rust_server_args,
            ),
        ):
            launched = RustServer.launch(scheduler)

        self.assertIs(launched.server, sentinel.server)
        server_constructor.assert_called_once_with(
            sentinel.rust_server_args,
            cores=None,
            port_offset=None,
            api_key="resolved-user-key",
            admin_api_key="resolved-admin-key",
        )


if __name__ == "__main__":
    unittest.main()
