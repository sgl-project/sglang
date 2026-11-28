"""Draft load formats must not inherit target-only loader options."""

import itertools
import json
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.model_loader.loader import _validate_default_loader_extra_config
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBuildLoadConfig(CustomTestCase):
    def _build(
        self, *, extra, target_format="fastsafetensors", draft_format="safetensors"
    ):
        with get_context().override_server_args(
            load_format=target_format, model_loader_extra_config=extra
        ) as server_args:
            return load_model_utils.build_load_config(
                server_args=server_args,
                tp_rank=0,
                load_format=draft_format,
                remote_instance_weight_transporter_engine=None,
                remote_instance_weight_transporter_session_id=None,
                draft_model_idx=None,
                weight_cache_mode="off",
                weight_cache_socket=None,
            )

    def test_draft_drops_target_gds_without_mutating_target(self):
        for draft_format, serialized, gds in itertools.product(
            ("safetensors", "auto"), (False, True), (False, True)
        ):
            with self.subTest(format=draft_format, serialized=serialized, gds=gds):
                original = {
                    "enable_gds": gds,
                    "enable_multithread_load": True,
                    "num_threads": 4,
                }
                extra = json.dumps(original) if serialized else original
                config = self._build(draft_format=draft_format, extra=extra)
                _validate_default_loader_extra_config(
                    extra_config=config.model_loader_extra_config,
                    load_format=config.load_format,
                )
                self.assertEqual(
                    config.model_loader_extra_config,
                    {"enable_multithread_load": True, "num_threads": 4},
                )
                self.assertIs(original["enable_gds"], gds)
                config.model_loader_extra_config["num_threads"] = 8
                self.assertEqual(original["num_threads"], 4)
                target_config = self._build(draft_format=None, extra=extra)
                self.assertEqual(target_config.model_loader_extra_config, original)

    def test_other_loader_profiles_preserve_options(self):
        for target_format, draft_format, extra in (
            ("fastsafetensors", "fastsafetensors", {"enable_gds": True}),
            ("auto", "safetensors", {"num_threads": 8}),
        ):
            with self.subTest(target=target_format, draft=draft_format):
                config = self._build(
                    target_format=target_format, draft_format=draft_format, extra=extra
                )
                _validate_default_loader_extra_config(
                    extra_config=config.model_loader_extra_config,
                    load_format=config.load_format,
                )
                self.assertEqual(config.model_loader_extra_config, extra)

    def test_missing_extra_options(self):
        for extra in (None, {}, "{}"):
            with self.subTest(extra=extra):
                config = self._build(extra=extra)
                self.assertEqual(config.model_loader_extra_config or {}, {})

    def test_unknown_draft_options_are_still_rejected(self):
        config = self._build(extra={"enable_gds": False, "unknown": 1})
        with self.assertRaisesRegex(ValueError, "unknown") as error:
            _validate_default_loader_extra_config(
                extra_config=config.model_loader_extra_config,
                load_format=config.load_format,
            )
        self.assertNotIn("enable_gds", str(error.exception))

    def test_target_gds_type_validation_is_preserved(self):
        for gds in (0, 1, "false"):
            with self.subTest(gds=gds):
                config = self._build(draft_format=None, extra={"enable_gds": gds})
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    _validate_default_loader_extra_config(
                        extra_config=config.model_loader_extra_config,
                        load_format=config.load_format,
                    )


if __name__ == "__main__":
    unittest.main()
