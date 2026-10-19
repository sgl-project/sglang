"""Model packages use one extension for server arguments and worker startup."""

import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


@contextmanager
def launch_context(server_cls):
    """Patch what `launch` reads so it runs as DP rank 1 of a two-rank node,
    unpinned; yields the `_build_server_args` mock."""
    with (
        patch.object(
            server_module,
            "get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=False)),
        ),
        patch.object(
            server_module,
            "get_parallel",
            return_value=SimpleNamespace(
                nnodes=1,
                pp_size=1,
                num_dp_ranks=2,
                attn_dp_rank=1,
                tp_size=2,
                tp_rank=1,
                attn_tp_size=1,
                attn_cp_size=1,
            ),
        ),
        patch.object(server_cls, "_partition_cores", return_value=(None, None)),
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
        yield build_args


class TestRustServerExtension(CustomTestCase):
    def test_launch_uses_the_model_extension_and_instance_worker_state(self):
        extension = ModuleType("model_server")
        extension.Server = Mock()

        class ModelServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

            @classmethod
            def _resolve_multimodal(cls, scheduler):
                return None

            def _start_multimodal(self, scheduler):
                self.server.start_mm_workers(sentinel.spec, 8)

        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True),
        )
        with launch_context(ModelServer) as build_args:
            instance = ModelServer.launch(scheduler)

        build_args.assert_called_once_with(
            scheduler, extension=extension, health_generate_headers={}
        )
        extension.Server.assert_called_once_with(
            sentinel.args, cores=None, port_offset=1
        )
        instance.server.start_mm_workers.assert_called_once_with(sentinel.spec, 8)
        self.assertIsInstance(instance, ModelServer)
        self.assertEqual(instance.http_port, 30001)
        self.assertTrue(instance._multimodal_enabled)

    def test_startup_time_takes_the_engine_shape(self):
        """The frontend's end-to-end phase runs from its process's spawn."""
        server = RustServer(Mock(), http_port=30000)
        self.assertIsNone(server.startup_time)
        process = SimpleNamespace(create_time=lambda: 100.0)
        with (
            patch.object(server_module, "time", SimpleNamespace(time=lambda: 110.0)),
            patch.object(
                server_module, "psutil", SimpleNamespace(Process=lambda: process)
            ),
        ):
            server.set_startup_time(
                {
                    "load_weight": 1.0,
                    "kv_cache_allocation": 0.5,
                    "scheduler_e2e": 3.0,
                    "cuda_graph": {"decode": 2.0},
                }
            )
        startup_time = server.startup_time
        self.assertEqual(startup_time["tokenizer_e2e"], 10.0)
        self.assertEqual(startup_time["scheduler_e2e"], 3.0)
        self.assertEqual(startup_time["load_weight"], 1.0)
        self.assertEqual(startup_time["cuda_graph"]["decode"], 2.0)

    def test_launch_resolves_the_mm_pipeline_before_the_server_starts(self):
        """The listener answers once the server is constructed, so a model
        without a Rust MM pipeline must be refused before that; a supported
        one gets workers for the spec resolved up front."""
        extension = ModuleType("server")
        extension.Server = Mock()

        class BuiltinServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

        scheduler = SimpleNamespace(
            server_args=sentinel.server_args,
            processor=sentinel.processor,
            model_config=SimpleNamespace(
                is_multimodal=True,
                hf_config=SimpleNamespace(model_type="unsupported_vlm"),
            ),
        )
        with (
            launch_context(BuiltinServer),
            patch.object(server_module, "RustMmProcessor") as mm_processor,
            patch.object(
                BuiltinServer, "_build_mm_spec", return_value=sentinel.worker_spec
            ),
        ):
            mm_processor.return_value.resolve_spec.return_value = None
            with self.assertRaisesRegex(
                RuntimeError, "no Rust MM pipeline for model_type='unsupported_vlm'"
            ):
                BuiltinServer.launch(scheduler)
            extension.Server.assert_not_called()

            mm_processor.return_value.resolve_spec.return_value = sentinel.spec
            instance = BuiltinServer.launch(scheduler)

        extension.Server.assert_called_once()
        instance.server.start_mm_workers.assert_called_once_with(
            sentinel.worker_spec, 8
        )
        self.assertIs(instance.mm_spec, sentinel.spec)
        self.assertTrue(instance._multimodal_enabled)


if __name__ == "__main__":
    unittest.main()
