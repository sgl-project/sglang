from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSarashina2VisionImport(CustomTestCase):
    def test_model_and_processor_registration(self):
        from sglang.srt.models.sarashina2_vision import (
            EntryClass,
            Sarashina2VisionForCausalLM,
        )
        from sglang.srt.multimodal.processors.sarashina2_vision import (
            Sarashina2VisionProcessor,
        )

        self.assertIs(EntryClass, Sarashina2VisionForCausalLM)
        self.assertEqual(
            Sarashina2VisionProcessor.models, [Sarashina2VisionForCausalLM]
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
