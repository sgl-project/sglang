import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.flashinfer_kimi_k3_tactics import (
    load_kimi_k3_prefill_tactics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKimiK3PrefillTactics(CustomTestCase):
    def setUp(self):
        self.kwargs = dict(
            model_type="kimi_k3",
            tp_size=1,
            pp_size=8,
            ep_size=1,
            chunk_size=16384,
            dtype=torch.bfloat16,
            device="cuda",
        )

    def test_only_the_validated_environment_is_seeded(self):
        tuner = Mock()
        tuner.load_configs.return_value = True
        with (
            patch.dict(
                "sys.modules", {"flashinfer": SimpleNamespace(__version__="0.7.2")}
            ),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
        ):
            self.assertTrue(load_kimi_k3_prefill_tactics(tuner, **self.kwargs))
            path = Path(tuner.load_configs.call_args.args[0])
            records = json.loads(path.read_text())
            self.assertEqual(records.pop("_metadata")["flashinfer_version"], "0.7.2")
            self.assertEqual(len(records), 1)
            key, record = next(iter(records.items()))
            op, runner, shapes, extras = ast.literal_eval(key)
            self.assertEqual(op, "flashinfer::trtllm_fp4_block_scale_moe")
            self.assertEqual(runner, record[0])
            self.assertEqual(record[1], [128, 4])
            self.assertEqual(shapes[0], (16384, 3584))
            self.assertEqual(extras[1:5], (16, 896, 896, 0))
            for field, value in (
                ("model_type", "other"),
                ("tp_size", 8),
                ("pp_size", 1),
                ("ep_size", 4),
                ("chunk_size", 8192),
                ("dtype", torch.float16),
            ):
                with self.subTest(field=field):
                    tuner.reset_mock()
                    args = self.kwargs | {field: value}
                    self.assertFalse(load_kimi_k3_prefill_tactics(tuner, **args))
                    tuner.load_configs.assert_not_called()

    def test_unvalidated_flashinfer_and_other_hardware_keep_defaults(self):
        for version, capability in (
            ("0.7.1", (10, 3)),
            ("0.7.3", (10, 3)),
            ("0.7.2", (10, 0)),
        ):
            with (
                self.subTest(version=version, capability=capability),
                patch.dict(
                    "sys.modules", {"flashinfer": SimpleNamespace(__version__=version)}
                ),
                patch("torch.cuda.get_device_capability", return_value=capability),
            ):
                tuner = Mock()
                self.assertFalse(load_kimi_k3_prefill_tactics(tuner, **self.kwargs))
                tuner.load_configs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
