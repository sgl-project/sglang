"""The native-MM launch gate's family selection (managers/rust_server.py).

``NATIVE_MM_FAMILIES`` decides which models the Rust pipeline serves natively;
for everything else ``native_mm_family_for`` must return ``None``, which
``RustServer.launch`` turns into a hard launch error. Pins that non-Qwen
multimodal models — Inkling being the in-tree case — keep their Python
processor and never match a native family, so growing the registry cannot
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
from sglang.srt.managers.rust_server import native_mm_family_for  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def processor_cls_for(architecture, model_type):
    """Through the production selection, as `resolve_native_spec` calls it."""
    hf_config = SimpleNamespace(architectures=[architecture], model_type=model_type)
    return get_mm_processor_cls(hf_config, SimpleNamespace(model_impl="sglang"))


class TestNativeMmGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import_processors("sglang.srt.multimodal.processors")

    def test_qwen_vl_resolves_its_family(self):
        cls = processor_cls_for("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl")
        family = native_mm_family_for(cls, "qwen2_5_vl")
        self.assertEqual(family and family.name, "qwen_vl")

    def test_inkling_keeps_its_python_processor(self):
        from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor

        cls = processor_cls_for("InklingForConditionalGeneration", "inkling_model")
        self.assertIs(cls, InklingMultimodalProcessor)
        self.assertIsNone(native_mm_family_for(cls, "inkling_model"))

    def test_family_requires_both_processor_and_model_type(self):
        qwen = processor_cls_for("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl")
        self.assertIsNone(native_mm_family_for(qwen, "inkling_model"))
        # Identity, not name: an override class must not match (the
        # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE contract).
        impostor = type("QwenVLImageProcessor", (), {})
        self.assertIsNone(native_mm_family_for(impostor, "qwen2_5_vl"))


if __name__ == "__main__":
    unittest.main()
