import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers import PretrainedConfig

from sglang.srt.configs import linear_attn_model_registry
from sglang.srt.layers.attention.attention_registry import (
    OOT_LINEAR_ATTENTION_BACKENDS,
    attn_backend_wrapper,
    register_oot_linear_attention_backend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class CustomGDNBackend(GDNAttnBackend):
    pass


class LinearBackendRegistryTest(unittest.TestCase):
    def _select_registered_model_backend(self, backend_class, registration, oot, key):
        config = PretrainedConfig(
            architectures=["RegistryTest"], full_attention_layer_ids=[1]
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config, hf_text_config=config),
            use_mla_backend=False,
            is_draft_worker=False,
        )
        platform = Mock()
        platform.is_out_of_tree.return_value = oot
        platform.get_dispatch_key_name.return_value = key
        mamba = SimpleNamespace(
            linear_attn_backend="triton",
            linear_attn_decode_backend=None,
            linear_attn_prefill_backend=None,
            linear_attn_verify_backend=None,
        )
        factory = Mock()
        with (
            patch.dict(OOT_LINEAR_ATTENTION_BACKENDS, {}, clear=True),
            patch.object(linear_attn_model_registry, "_LINEAR_ATTN_MODEL_REGISTRY", []),
            patch("sglang.srt.platforms.current_platform", platform),
            patch(
                "sglang.srt.layers.attention.linear.utils.get_exec",
                return_value=SimpleNamespace(mamba=mamba),
            ),
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend.check_cuda_graph_backend",
                return_value=False,
            ),
            patch.object(backend_class, "__init__", return_value=None) as constructor,
        ):
            linear_attn_model_registry.register_linear_attn_model(
                linear_attn_model_registry.LinearAttnModelSpec(
                    config_class=PretrainedConfig,
                    backend_class_name=f"{backend_class.__module__}.{backend_class.__qualname__}",
                )
            )
            runner.model_config.linear_attn_registry_result = (
                linear_attn_model_registry.get_linear_attn_config(config)
            )
            register_oot_linear_attention_backend(
                f"{registration.__module__}.{registration.__qualname__}",
                factory,
                "test",
            )
            actual = attn_backend_wrapper(runner, Mock())
        return actual, runner, factory, constructor

    def test_registered_models_use_the_platform_factory(self):
        for backend_class in (GDNAttnBackend, KDAAttnBackend, CustomGDNBackend):
            with self.subTest(backend=backend_class):
                actual, runner, factory, constructor = (
                    self._select_registered_model_backend(
                        backend_class, backend_class, True, "test"
                    )
                )
                self.assertIs(actual.linear_attn_backend, factory.return_value)
                self.assertEqual(actual.full_attn_layers, [1])
                factory.assert_called_once_with(runner)
                constructor.assert_not_called()

    def test_builtin_and_unregistered_selections_keep_the_original_class(self):
        for backend_class, registration, oot, key in (
            (GDNAttnBackend, GDNAttnBackend, False, "test"),
            (KDAAttnBackend, KDAAttnBackend, False, "test"),
            (GDNAttnBackend, GDNAttnBackend, True, "other"),
            (KDAAttnBackend, GDNAttnBackend, True, "test"),
            (CustomGDNBackend, GDNAttnBackend, True, "test"),
        ):
            with self.subTest(backend=backend_class, oot=oot, key=key):
                actual, runner, factory, constructor = (
                    self._select_registered_model_backend(
                        backend_class, registration, oot, key
                    )
                )
                self.assertIsInstance(actual.linear_attn_backend, backend_class)
                self.assertEqual(actual.full_attn_layers, [1])
                constructor.assert_called_once_with(runner)
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
