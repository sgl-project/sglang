import unittest
from unittest.mock import patch

from sglang.srt.models.llava import LlavaForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class GoodConfig:
    pass


class BrokenConfig:
    pass


class GoodArch:
    pass


class FakeMapping:
    def keys(self):
        return [BrokenConfig, GoodConfig]

    def get(self, config_cls, default=None):
        if config_cls is BrokenConfig:
            raise ValueError("broken mapping entry")
        if config_cls is GoodConfig:
            return GoodArch
        return default


class FakeAutoModel:
    _model_mapping = FakeMapping()


class TestLlavaForConditionalGeneration(CustomTestCase):
    @patch("sglang.srt.models.llava.logger.warning")
    def test_skip_broken_automodel_mapping_entries(self, mock_warning):
        llava_model = object.__new__(LlavaForConditionalGeneration)
        build_mapping = (
            LlavaForConditionalGeneration._config_cls_name_to_arch_name_mapping.__wrapped__
        )

        mapping = build_mapping(llava_model, FakeAutoModel)

        self.assertEqual(mapping[GoodConfig.__name__], GoodArch.__name__)
        self.assertNotIn(BrokenConfig.__name__, mapping)

        mock_warning.assert_called_once()
        self.assertEqual(
            mock_warning.call_args.args,
            (
                "Skipping broken %s mapping for config %s: %s",
                FakeAutoModel.__name__,
                BrokenConfig.__name__,
                unittest.mock.ANY,
            ),
        )


if __name__ == "__main__":
    unittest.main()
