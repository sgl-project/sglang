import unittest
from unittest.mock import Mock, call, patch

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDefaultModelLoader(unittest.TestCase):
    def test_pre_weight_processing_hook_runs_before_postprocessing(self):
        events = Mock()
        model = object()
        model_config = object()

        with (
            patch.object(
                DefaultModelLoader,
                "load_weights_only",
                side_effect=lambda *_: events("load"),
            ),
            patch.object(
                DefaultModelLoader,
                "postprocess_weights",
                side_effect=lambda *_: events("postprocess"),
            ),
            patch.object(
                DefaultModelLoader,
                "pre_weight_processing_hook",
                side_effect=lambda received_model, received_config: events(
                    "hook", received_model, received_config
                ),
            ),
        ):
            DefaultModelLoader.load_weights_and_postprocess(
                model, iter(()), "cpu", model_config=model_config
            )

        self.assertEqual(
            events.call_args_list,
            [call("load"), call("hook", model, model_config), call("postprocess")],
        )

    def test_pre_weight_processing_hook_skips_non_boot_loads(self):
        with (
            patch.object(DefaultModelLoader, "load_weights_only"),
            patch.object(DefaultModelLoader, "postprocess_weights"),
            patch.object(DefaultModelLoader, "pre_weight_processing_hook") as hook,
        ):
            DefaultModelLoader.load_weights_and_postprocess(
                object(), iter(()), "cpu", model_config=None
            )

        hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
