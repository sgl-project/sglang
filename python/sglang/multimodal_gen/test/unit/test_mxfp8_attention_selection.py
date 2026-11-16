# SPDX-License-Identifier: Apache-2.0
"""CPU tests for opt-in Ascend MXFP8 attention backend selection."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.attention_roles import AttentionRole
from sglang.multimodal_gen.runtime.layers.attention import layer, selector
from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class MXFP8Config:
    pass


class ModelSlimConfig:
    pass


class TestMXFP8AttentionSelection(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(os.environ, SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION="true")
        )
        self.is_npu = self.enterContext(
            patch.object(ascend_fa.current_platform, "is_npu", return_value=True)
        )
        self.enterContext(
            patch.object(
                torch, "npu", SimpleNamespace(get_soc_version=lambda: 260), create=True
            )
        )
        self.ops = SimpleNamespace(
            npu_dynamic_mx_quant=object(), npu_fused_infer_attention_score_v2=object()
        )
        self.enterContext(patch.object(torch.ops, "npu", self.ops))
        self.server_args = SimpleNamespace(attention_backend=None)
        self.enterContext(
            patch.object(
                selector, "get_global_server_args", return_value=self.server_args
            )
        )
        self.enterContext(patch.object(selector, "forced_attn_backend", None))
        self.enterContext(
            patch.object(selector, "_cached_get_attn_backend", self._backend)
        )

    @staticmethod
    def _backend(_head_size, _dtype, _supported, selected):
        selected = selected or AttentionBackendEnum.TORCH_SDPA
        return SimpleNamespace(
            get_enum=lambda: selected,
            unsupported_requirements=lambda _requirements: (),
            get_impl_cls=lambda: lambda **kwargs: SimpleNamespace(**kwargs),
            supports_ring_rotation=lambda: True,
        )

    def _resolve(self, **kwargs):
        kwargs.setdefault("quant_config", MXFP8Config())
        return selector.get_attn_backend(64, torch.bfloat16, **kwargs).get_enum()

    def test_online_and_offline_configs_select_fa(self):
        for config in (MXFP8Config(), ModelSlimConfig()):
            with self.subTest(config=type(config).__name__):
                self.assertIs(
                    self._resolve(quant_config=config), AttentionBackendEnum.FA
                )

    def test_disabled_opt_in_keeps_platform_default(self):
        with patch.dict(os.environ, SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION="false"):
            self.assertIs(self._resolve(), AttentionBackendEnum.TORCH_SDPA)

    def test_unsupported_configs_keep_platform_default(self):
        for config in (None, object()):
            with self.subTest(config=config):
                self.assertIs(
                    self._resolve(quant_config=config), AttentionBackendEnum.TORCH_SDPA
                )

    def test_non_npu_keeps_platform_default(self):
        self.is_npu.return_value = False
        self.assertIs(self._resolve(), AttentionBackendEnum.TORCH_SDPA)

    def test_missing_operator_keeps_platform_default(self):
        del self.ops.npu_fused_infer_attention_score_v2
        self.assertIs(self._resolve(), AttentionBackendEnum.TORCH_SDPA)

    def test_unsupported_device_keeps_platform_default(self):
        with patch.object(torch.npu, "get_soc_version", return_value=250):
            self.assertIs(self._resolve(), AttentionBackendEnum.TORCH_SDPA)

    def test_cross_attention_keeps_platform_default(self):
        self.assertIs(
            self._resolve(is_cross_attention=True), AttentionBackendEnum.TORCH_SDPA
        )

    def test_backend_constraints_still_apply(self):
        self.assertIs(
            self._resolve(
                supported_attention_backends={AttentionBackendEnum.TORCH_SDPA}
            ),
            AttentionBackendEnum.TORCH_SDPA,
        )

    def test_explicit_backends_and_layer_defaults_take_precedence(self):
        sdpa = AttentionBackendEnum.TORCH_SDPA
        for arg in ("selected_attention_backend", "default_attention_backend"):
            with self.subTest(arg=arg):
                self.assertIs(self._resolve(**{arg: sdpa}), sdpa)
        with patch.object(selector, "forced_attn_backend", sdpa):
            self.assertIs(self._resolve(), sdpa)
        with patch.object(self.server_args, "attention_backend", "torch_sdpa"):
            self.assertIs(self._resolve(), sdpa)
        with selector.component_attn_backend_context_manager(
            sdpa, component_name="transformer"
        ):
            self.assertIs(self._resolve(), sdpa)
        with selector.component_attn_backend_context_manager(
            None,
            component_name="transformer",
            backend_by_role={AttentionRole.SELF: sdpa},
        ):
            self.assertIs(self._resolve(), sdpa)

    def test_layers_forward_quant_config_and_cross_attention_role(self):
        with (
            patch.object(layer, "get_compute_dtype", return_value=torch.bfloat16),
            patch.object(layer, "get_ring_parallel_world_size", return_value=1),
            patch.object(layer, "wrap_attention_impl_forward"),
            patch.object(layer, "_maybe_install_backend_autotune"),
            patch.object(
                layer, "_resolve_sp_attention_mode", return_value=("a2a", False)
            ),
        ):
            for cls in (
                layer.LocalAttention,
                layer.UlyssesAttention,
                layer.USPAttention,
            ):
                for is_cross in (False, True):
                    with self.subTest(layer=cls.__name__, is_cross=is_cross):
                        config = MXFP8Config()
                        attention = cls(
                            num_heads=1,
                            head_size=64,
                            quant_config=config,
                            is_cross_attention=is_cross,
                        )
                        expected = (
                            AttentionBackendEnum.TORCH_SDPA
                            if is_cross
                            else AttentionBackendEnum.FA
                        )
                        self.assertIs(attention.backend, expected)
                        self.assertIs(attention.attn_impl.quant_config, config)


if __name__ == "__main__":
    unittest.main()
