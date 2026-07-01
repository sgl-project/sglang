import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_server_args(**overrides):
    args = SimpleNamespace(
        enable_hisparse=True,
        disable_radix_cache=True,
        kv_cache_dtype="auto",
        dsa_prefill_backend=None,
        dsa_decode_backend=None,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=2,
        speculative_num_draft_tokens=3,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.get_model_config = lambda: SimpleNamespace(hf_config=object())
    return args


class TestHiSparseOnlineC128MTPSupport(CustomTestCase):
    def _validate(self, args, *, online=True, experimental=True, compressor_v2=True):
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(online))
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.override(experimental)
            )
            stack.enter_context(
                envs.SGLANG_OPT_USE_COMPRESSOR_V2.override(compressor_v2)
            )
            stack.enter_context(
                patch("sglang.srt.arg_groups.hisparse_hook._is_hip", return_value=False)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_deepseek_v4",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_deepseek_dsa",
                    return_value=False,
                )
            )
            validate_hisparse(args)

    def test_accepts_dsv4_hisparse_online_c128_eagle(self):
        self._validate(_make_server_args())

    def test_requires_experimental_online_c128_mtp_for_speculative(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP"):
            self._validate(_make_server_args(), experimental=False)

    def test_rejects_non_eagle_speculative_algorithm(self):
        with self.assertRaisesRegex(ValueError, "supports EAGLE"):
            self._validate(_make_server_args(speculative_algorithm="NGRAM"))

    def test_rejects_eagle_topk_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "speculative-eagle-topk 1"):
            self._validate(_make_server_args(speculative_eagle_topk=2))

    def test_rejects_eagle_step3(self):
        with self.assertRaisesRegex(ValueError, "supports EAGLE step1/step2"):
            self._validate(_make_server_args(speculative_num_steps=3))

    def test_requires_compressor_v2(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_OPT_USE_COMPRESSOR_V2=1"):
            self._validate(_make_server_args(), compressor_v2=False)

    def test_non_speculative_online_c128_keeps_existing_hisparse_path(self):
        self._validate(
            _make_server_args(
                speculative_algorithm=None,
                speculative_eagle_topk=None,
                speculative_num_steps=None,
                speculative_num_draft_tokens=None,
            ),
            experimental=False,
            compressor_v2=False,
        )


if __name__ == "__main__":
    unittest.main()
