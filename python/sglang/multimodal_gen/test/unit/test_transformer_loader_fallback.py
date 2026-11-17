import re
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    NativeComponentLoaderRequired,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)


class TestTransformerLoaderFallbackAdmission(unittest.TestCase):
    @staticmethod
    def _server_args(*, fsdp_requested=False, **overrides):
        values = {
            "component_precisions": {},
            "component_quantizations": {},
            "component_weights_paths": {},
            "component_quantization_ignored_layers": {},
            "transformer_weights_path": None,
            "nunchaku_config": None,
            "quantization": None,
            "pipeline_config": PipelineConfig(),
            "tp_size": 1,
            "sp_degree": 1,
            "ulysses_degree": 1,
            "ring_degree": 1,
            "kv_gather_degree": 1,
            "enable_cfg_parallel": False,
            "dp_size": 1,
            "use_fsdp_inference": False,
            "resolve_component_attention_backend": mock.Mock(return_value=(None, None)),
            "resolve_component_backend_by_role": mock.Mock(return_value={}),
            "requested_component_attention_backend": mock.Mock(return_value=None),
            "should_direct_gpu_weight_load_component": mock.Mock(return_value=False),
            "should_use_fsdp_for_component": mock.Mock(return_value=fsdp_requested),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _mocked_load(loader):
        customized_load = mock.patch.object(
            loader,
            "_load_customized_with_context",
            side_effect=NativeComponentLoaderRequired("native loader required"),
        )
        native_load = mock.patch.object(
            loader, "_load_native_with_context", return_value=object()
        )
        available_memory = mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.current_platform.get_available_gpu_memory",
            return_value=0.0,
        )
        return customized_load, native_load, available_memory

    def test_distributed_execution_rejects_before_native_load(self):
        cases = (
            ("tp_size", 2, "tp_size=2"),
            ("sp_degree", 2, "sp_degree=2"),
            ("ulysses_degree", 2, "ulysses_degree=2"),
            ("ring_degree", 2, "ring_degree=2"),
            ("kv_gather_degree", 2, "kv_gather_degree=2"),
            ("fsdp_requested", True, "FSDP"),
        )

        for field, value, expected_error in cases:
            with self.subTest(field=field):
                loader = TransformerLoader()
                server_args = self._server_args(**{field: value})
                customized_load, native_load, available_memory = self._mocked_load(
                    loader
                )

                with customized_load, native_load as native, available_memory:
                    with self.assertRaisesRegex(
                        RuntimeError, re.escape(expected_error)
                    ):
                        loader.load(
                            "/model/transformer_2",
                            server_args,
                            "transformer_2",
                            "diffusers",
                        )

                native.assert_not_called()
                server_args.should_use_fsdp_for_component.assert_called_with(
                    "transformer_2"
                )

    def test_replicated_cfg_and_dp_keep_native_fallback(self):
        loader = TransformerLoader()
        server_args = self._server_args(
            enable_cfg_parallel=True,
            dp_size=2,
            use_fsdp_inference=True,
        )
        customized_load, native_load, available_memory = self._mocked_load(loader)

        with customized_load, native_load as native, available_memory:
            component, consumed = loader.load(
                "/model/transformer_2",
                server_args,
                "transformer_2",
                "diffusers",
            )

        self.assertIsNotNone(component)
        self.assertEqual(consumed, 0.0)
        native.assert_called_once()
        server_args.should_use_fsdp_for_component.assert_called_with("transformer_2")

    def test_unreadable_checkpoint_is_not_a_missing_implementation(self):
        # the native fallback answers "no customized implementation for this
        # architecture"; a checkpoint that cannot be read is a different failure,
        # and routing it into the fallback reports a missing implementation for a
        # model that has one
        loader = TransformerLoader()
        missing_shard = FileNotFoundError(
            2, "No such file or directory", "/cache/transformer/shard-00002.safetensors"
        )
        customized_load = mock.patch.object(
            loader, "_load_customized_with_context", side_effect=missing_shard
        )
        native_load = mock.patch.object(
            loader, "_load_native_with_context", return_value=object()
        )
        available_memory = mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.current_platform.get_available_gpu_memory",
            return_value=0.0,
        )

        with customized_load, native_load as native, available_memory:
            with self.assertRaises(FileNotFoundError) as caught:
                loader.load(
                    "/model/transformer_2",
                    self._server_args(),
                    "transformer_2",
                    "diffusers",
                )

        self.assertIn("shard-00002.safetensors", str(caught.exception))
        native.assert_not_called()


class TestIdeogramCheckpointLayout(unittest.TestCase):
    def test_layout_selected_before_construction(self):
        import tempfile
        from pathlib import Path

        import torch
        from safetensors.torch import save_file

        from sglang.multimodal_gen.configs.models.dits.ideogram import (
            Ideogram4DistilledDiTConfig,
            Ideogram4DiTConfig,
        )
        from sglang.multimodal_gen.runtime.loader.component_loaders import (
            transformer_loader as module,
        )
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            TransformerQuantLoadSpec,
        )
        from sglang.multimodal_gen.runtime.models.dits.ideogram import (
            Ideogram4Transformer2DModel,
        )

        for layout in ("fp8", "bf16", "missing_scale", "distilled", "quantized"):
            with (
                self.subTest(layout=layout),
                tempfile.TemporaryDirectory() as directory,
            ):
                dtype = torch.float8_e4m3fn if layout != "bf16" else torch.bfloat16
                weights = {"input_proj.weight": torch.zeros(4, 8).to(dtype)}
                if layout == "fp8":
                    weights["input_proj.weight_scale"] = torch.ones(4)
                path = str(Path(directory) / "model.safetensors")
                save_file(weights, path)
                config = (
                    Ideogram4DistilledDiTConfig()
                    if layout == "distilled"
                    else Ideogram4DiTConfig()
                )
                args = TestTransformerLoaderFallbackAdmission._server_args(
                    pipeline_config=PipelineConfig(dit_config=config),
                    model_paths={},
                    transformer_weights_path=path if layout == "bf16" else None,
                    revision=None,
                    should_start_component_on_cpu=lambda name: True,
                    minimax_h3_adaln_cache_path=None,
                    minimax_h3_adaln_online=False,
                    direct_gpu_weight_loading=False,
                )
                spec = TransformerQuantLoadSpec(
                    safetensors_list=[path],
                    quant_config=mock.Mock() if layout == "quantized" else None,
                    nunchaku_config=None,
                    param_dtype=torch.bfloat16,
                )
                loader = TransformerLoader()
                with (
                    mock.patch.object(
                        module,
                        "get_diffusers_component_config",
                        return_value={"_class_name": "Ideogram4Transformer2DModel"},
                    ),
                    mock.patch.object(
                        module, "resolve_transformer_gguf_to_load", return_value=None
                    ),
                    mock.patch.object(
                        module, "resolve_transformer_quant_load_spec", return_value=spec
                    ),
                    mock.patch.object(
                        module.ModelRegistry,
                        "resolve_model_cls",
                        return_value=(Ideogram4Transformer2DModel, None),
                    ),
                    mock.patch.object(
                        module,
                        "get_local_torch_device",
                        return_value=torch.device("cpu"),
                    ),
                    mock.patch.object(
                        module,
                        "_default_quantized_attention_backend",
                        return_value=None,
                    ),
                    mock.patch.object(
                        loader,
                        "load_state_dict_model",
                        side_effect=RuntimeError("stop at construction"),
                    ) as construct,
                ):
                    expected = (
                        "missing scale"
                        if layout == "missing_scale"
                        else "stop at construction"
                    )
                    error_type = (
                        ComponentCheckpointUnsupportedError
                        if layout == "missing_scale"
                        else RuntimeError
                    )
                    with self.assertRaisesRegex(error_type, expected):
                        loader.load_customized(directory, args, "transformer")
                if layout == "missing_scale":
                    construct.assert_not_called()
                else:
                    construct.assert_called_once()
                    self.assertEqual(
                        config.use_weight_only_fp8_linears,
                        layout in ("fp8", "quantized"),
                    )
                    self.assertEqual(
                        construct.call_args.kwargs["dtype"], torch.bfloat16
                    )

                    if layout in ("fp8", "bf16", "distilled"):
                        from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
                            WeightOnlyFP8Linear,
                        )

                        with (
                            mock.patch(
                                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                                return_value=1,
                            ),
                            torch.device("meta"),
                        ):
                            model = Ideogram4Transformer2DModel(config, {})
                        self.assertEqual(
                            isinstance(model.input_proj, WeightOnlyFP8Linear),
                            layout == "fp8",
                        )
                        self.assertEqual(
                            model.input_proj.weight.dtype == torch.float8_e4m3fn,
                            layout == "fp8",
                        )


if __name__ == "__main__":
    unittest.main()
