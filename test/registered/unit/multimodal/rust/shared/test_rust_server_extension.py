"""Model packages use one extension for server arguments and worker startup."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.config import _build_server_args  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRustServerExtension(CustomTestCase):
    def test_config_handoff_uses_serving_tokenizer_and_model_capabilities(self):
        server_args = ServerArgs(
            model_path="dummy",
            tokenizer_path="public-tokenizer",
            served_model_name="served-model",
        )
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)
        extension = SimpleNamespace(
            ServerArgs=SimpleNamespace,
            ModelConfig=SimpleNamespace,
            DefaultSamplingParams=SimpleNamespace,
            DisaggregationMode=SimpleNamespace(
                Null="null", Prefill="prefill", Decode="decode"
            ),
        )
        scheduler = SimpleNamespace(
            server_args=server_args,
            model_config=SimpleNamespace(
                context_len=4096,
                vocab_size=256,
                is_multimodal=True,
                is_generation=True,
                is_image_understandable_model=True,
                is_audio_understandable_model=False,
                hf_config=SimpleNamespace(
                    model_type="llama", architectures=["LlamaForCausalLM"]
                ),
                get_default_sampling_params=lambda: {"temperature": 0.5},
            ),
            rust_server_tokenizer_path=lambda: "/data/tokenizer/tokenizer.json",
            max_total_num_tokens=8192,
        )

        args = _build_server_args(scheduler, extension=extension)

        self.assertEqual(args.public_tokenizer_path, "public-tokenizer")
        self.assertEqual(args.tokenizer_path, "/data/tokenizer/tokenizer.json")
        self.assertTrue(args.model_config.is_generation)
        self.assertTrue(args.model_config.has_image_understanding)
        self.assertFalse(args.model_config.has_audio_understanding)
        self.assertEqual(args.model_config.architectures, ["LlamaForCausalLM"])

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
            model_config=SimpleNamespace(is_multimodal=True),
        )
        with (
            patch.object(
                server_module,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
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
