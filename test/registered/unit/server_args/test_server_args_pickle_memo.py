"""A pickled ServerArgs leaves model_config_of's ModelConfig memo behind.

The scheduler and detokenizer processes get the record by pickle through
multiprocessing's spawn pipe, so a memo holding a large HF config made each
Process.start() wait for the child's imports."""

import pickle
from unittest.mock import patch

from sglang.srt.arg_groups import model_override_base
from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class BigConfig:
    """Stands in for a ModelConfig whose hf_quant_config runs to 300 KB."""

    is_hybrid_swa = False

    def __init__(self):
        self.hf_quant_config = {
            "exclude": [f"model.layers.{i}.mlp" for i in range(20000)]
        }


class TestServerArgsPickleMemo(CustomTestCase):
    def _record(self):
        return prepare_server_args(["--model-path", "/tmp/x", "--log-level", "warning"])

    def test_a_memo_built_from_the_checkpoint_stays_behind(self):
        record = self._record()
        small = len(pickle.dumps(record))
        with patch(
            "sglang.srt.configs.model_config.ModelConfig.from_server_args",
            return_value=BigConfig(),
        ) as build:
            built = model_override_base.model_config_of(record)
            self.assertIsInstance(built, BigConfig)
            arrived = pickle.loads(pickle.dumps(record))
            self.assertNotIn("_model_config", arrived.__dict__)
            self.assertLess(len(pickle.dumps(record)), small + 1024)
            # The sender keeps its memo; the receiver builds its own on first use.
            self.assertIs(model_override_base.model_config_of(record), built)
            self.assertIsInstance(
                model_override_base.model_config_of(arrived), BigConfig
            )
            self.assertEqual(build.call_count, 2)
        self.assertEqual(arrived.launch_command, record.launch_command)

    def test_a_memo_with_no_key_travels(self):
        record = self._record()
        record._model_config = "a fixture's config"
        arrived = pickle.loads(pickle.dumps(record))
        self.assertEqual(arrived._model_config, "a fixture's config")
        self.assertEqual(
            model_override_base.model_config_of(arrived), "a fixture's config"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
