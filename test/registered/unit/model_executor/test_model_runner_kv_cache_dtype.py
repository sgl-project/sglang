"""A runner resolves its KV cache dtype label together with its torch dtype.

Attention backends gate FP8 casts and descales on the label
(``kv_cache_dtype_str``), and the pool is allocated in the torch dtype
(``kv_cache_dtype``); the two must describe the same cache. A pool in the
compute dtype is labelled ``"auto"``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_FP8_KV = SimpleNamespace(kv_cache_quant_algo="FP8")


def _resolve(
    *,
    flag,
    draft_flag=None,
    quant_config=None,
    is_draft_worker=False,
    algorithm="EAGLE",
    draft_backend="fa3",
):
    """The (label, dtype) a runner stamps from the given flags."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(quant_config=quant_config)
    runner.dtype = torch.bfloat16
    runner.is_draft_worker = is_draft_worker
    runner.spec_algorithm = SpeculativeAlgorithm.from_string(algorithm)
    runner.draft_attention_backend = draft_backend
    with (
        patch.object(
            model_runner_module,
            "get_model",
            return_value=SimpleNamespace(kv_cache_dtype=flag),
        ),
        patch.object(
            model_runner_module,
            "get_spec",
            return_value=SimpleNamespace(speculative_draft_kv_cache_dtype=draft_flag),
        ),
    ):
        runner.configure_kv_cache_dtype()
    return runner.kv_cache_dtype_str, runner.kv_cache_dtype


class TestRunnerKvCacheDtypeLabel(CustomTestCase):
    def test_the_target_label_follows_its_flag_and_quant_config(self):
        for flag, quant_config, expected in (
            ("auto", None, ("auto", torch.bfloat16)),
            ("auto", _FP8_KV, ("fp8_e4m3", torch.float8_e4m3fn)),
            ("bf16", _FP8_KV, ("bf16", torch.bfloat16)),
        ):
            with self.subTest(flag=flag, quant_config=quant_config):
                # A target ignores the draft's flag.
                for draft_flag in (None, "auto", "bf16"):
                    self.assertEqual(
                        _resolve(
                            flag=flag, draft_flag=draft_flag, quant_config=quant_config
                        ),
                        expected,
                    )

    def test_the_draft_label_follows_the_flag_it_resolves(self):
        for flag, draft_flag, quant_config, expected in (
            # No draft flag: the target's flag, resolved against the draft model.
            ("auto", None, _FP8_KV, ("fp8_e4m3", torch.float8_e4m3fn)),
            # The draft's own explicit flag.
            ("fp8_e4m3", "bf16", None, ("bf16", torch.bfloat16)),
            # The draft's own `auto` resolving to FP8.
            ("bf16", "auto", _FP8_KV, ("fp8_e4m3", torch.float8_e4m3fn)),
            # BUG REGRESSION: the draft's own `auto` resolving to the compute
            # dtype under an explicit FP8 target was labelled with the target's
            # flag, so the backend descaled a BF16 cache.
            ("fp8_e4m3", "auto", None, ("auto", torch.bfloat16)),
        ):
            with self.subTest(flag=flag, draft_flag=draft_flag, quant=quant_config):
                label, dtype = _resolve(
                    flag=flag,
                    draft_flag=draft_flag,
                    quant_config=quant_config,
                    is_draft_worker=True,
                )
                self.assertEqual((label, dtype), expected)

    def test_a_dflash_fa4_draft_is_labelled_unquantized(self):
        self.assertEqual(
            _resolve(
                flag="fp8_e4m3",
                is_draft_worker=True,
                algorithm="DFLASH",
                draft_backend="fa4",
            ),
            ("auto", torch.bfloat16),
        )


if __name__ == "__main__":
    unittest.main()
