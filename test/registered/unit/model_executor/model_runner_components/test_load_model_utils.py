from unittest.mock import patch

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(0.1, "base-a-test-cpu")


class TestLoadKVCacheScales(CustomTestCase):
    def test_dynamic_scale_model_skips_static_scale_warning(self):
        class DynamicScaleModel:
            kv_cache_scale_mode = "dynamic"

        with (
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "load_model_utils.get_model"
            ) as get_model,
            self.assertLogs(
                "sglang.srt.model_executor.model_runner_components.load_model_utils",
                level="INFO",
            ) as logs,
        ):
            load_kv_cache_scales(model=DynamicScaleModel(), kv_cache_dtype="fp8_e4m3")

        get_model.assert_not_called()
        self.assertIn("scales generated dynamically", "\n".join(logs.output))


if __name__ == "__main__":
    import unittest

    unittest.main()
