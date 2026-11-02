"""FA3 segment gating must honor a transformer.self override over broader backends."""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.attention_roles import AttentionRole
from sglang.multimodal_gen.runtime.pipelines.hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.test.test_utils import CustomTestCase


def _args(role=None, component=None, global_backend=None):
    return SimpleNamespace(
        resolve_component_backend_by_role=lambda *_: (
            {AttentionRole.SELF: role} if role else {}
        ),
        resolve_component_attention_backend=lambda *_: (component, None),
        attention_backend=global_backend,
    )


class TestResolveSelfAttnBackend(CustomTestCase):
    def resolve(self, **kwargs):
        return HunyuanImage3Pipeline._resolve_transformer_self_attn_backend(
            _args(**kwargs)
        )

    def test_self_role_beats_global(self):
        out = self.resolve(role=AttentionBackendEnum.TORCH_SDPA, global_backend="fa3")
        self.assertEqual(out, "torch_sdpa")

    def test_self_role_beats_component(self):
        out = self.resolve(
            role=AttentionBackendEnum.TORCH_SDPA, component=AttentionBackendEnum.FA
        )
        self.assertEqual(out, "torch_sdpa")

    def test_component_enum_resolves_to_name(self):
        self.assertEqual(self.resolve(component=AttentionBackendEnum.FA), "fa")

    def test_global_fallback(self):
        self.assertEqual(self.resolve(global_backend="fa3"), "fa3")


if __name__ == "__main__":
    unittest.main()
