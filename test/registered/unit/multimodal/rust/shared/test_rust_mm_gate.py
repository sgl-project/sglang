"""The Rust-MM launch gate's family selection.

``RUST_MM_FAMILIES`` decides which models the Rust pipeline serves;
for everything else ``rust_mm_family_for`` must return ``None``, which
``RustServer.launch`` turns into a hard launch error. Pins that non-Qwen
multimodal models — Inkling being the in-tree case — keep their Python
processor and never match a Rust family, so growing the registry cannot
silently reroute them.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.multimodal_processor import (  # noqa: E402
    get_mm_processor_cls,
    import_processors,
)
from sglang.srt.runtime_context import publish
from sglang.srt.rust_server.multimodal import rust_mm_family_for  # noqa: E402
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


def processor_cls_for(architecture, model_type):
    """Through the production selection, as `resolve_spec` calls it."""
    hf_config = SimpleNamespace(architectures=[architecture], model_type=model_type)
    # `model_impl` is read from the bags now, so it has to be published rather
    # than handed over on a stand-in.
    publish(ServerArgs(model_path="dummy", model_impl="sglang"), role="tokenizer")
    return get_mm_processor_cls(hf_config)


class TestRustMmGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import_processors("sglang.srt.multimodal.processors")

    def test_qwen_vl_resolves_its_family(self):
        cls = processor_cls_for("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl")
        family = rust_mm_family_for(cls, "qwen2_5_vl")
        self.assertEqual(family and family.name, "qwen_vl")

    def test_glm5_next_resolves_its_family(self):
        cls = processor_cls_for("Glm5NextForConditionalGeneration", "glm5_next")
        family = rust_mm_family_for(cls, "glm5_next")
        self.assertEqual(family and family.name, "glm5_next")

    def test_deepseek_v41_resolves_its_family(self):
        cls = processor_cls_for("DeepseekV4ForCausalLM", "deepseek_v41")
        family = rust_mm_family_for(cls, "deepseek_v41")
        self.assertEqual(family and family.name, "deepseek_v41")

    def test_deepseek_v41_serves_only_recipe_geometry(self):
        cls = processor_cls_for("DeepseekV4ForCausalLM", "deepseek_v41")
        family = rust_mm_family_for(cls, "deepseek_v41")
        default = dict(
            image_token_id=7,
            vision_patch_size=14,
            vision_downsample_ratio=3,
            vision_min_pixels=544 * 544,
            vision_max_n_token=1024,
            vision_max_wh_ratio=None,
        )

        def resolve(**config):
            hf_config = SimpleNamespace(**config)
            return family.resolve(
                processor=None, hf_config=hf_config, feature_shm=False
            )

        spec = resolve(**default)
        self.assertEqual(spec.pipeline["vision_max_n_token"], 1024)
        # deepseek-recipe-image cannot resize to any other geometry.
        for name, value in [
            ("vision_max_n_token", 128),
            ("vision_min_pixels", 0),
            ("vision_max_wh_ratio", 4),
        ]:
            self.assertIsNone(resolve(**{**default, name: value}), name)

    def test_inkling_keeps_its_python_processor(self):
        from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor

        cls = processor_cls_for("InklingForConditionalGeneration", "inkling_model")
        self.assertIs(cls, InklingMultimodalProcessor)
        self.assertIsNone(rust_mm_family_for(cls, "inkling_model"))

    def test_family_requires_both_processor_and_model_type(self):
        qwen = processor_cls_for("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl")
        self.assertIsNone(rust_mm_family_for(qwen, "inkling_model"))
        # Identity, not name: an override class must not match (the
        # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE contract).
        impostor = type("QwenVLImageProcessor", (), {})
        self.assertIsNone(rust_mm_family_for(impostor, "qwen2_5_vl"))


if __name__ == "__main__":
    unittest.main()
