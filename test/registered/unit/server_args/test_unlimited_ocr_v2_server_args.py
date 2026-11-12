"""Unlimited-OCR v2 server-argument validation."""

import unittest

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import collect_model_override_declarations
from sglang.srt.configs.model_config import UNLIMITED_OCR_V2_ARCH
from sglang.srt.configs.unlimited_ocr import UnlimitedVLConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _v2_hf_config() -> UnlimitedVLConfig:
    return UnlimitedVLConfig(
        architectures=[UNLIMITED_OCR_V2_ARCH],
        sparse_attention_config={"sparse_block_size": 32, "training_stage": "dash"},
    )


def _declared_overrides(**fields) -> dict:
    """What the v2 model override declares for a server launched with ``fields``."""
    declarations = collect_model_override_declarations(
        UNLIMITED_OCR_V2_ARCH,
        ServerArgs(model_path="dummy", **fields),
        hf_config=_v2_hf_config(),
    )
    return {
        field: value
        for _, declaration in declarations
        for field, value in declaration.items()
    }


class TestDocSplitPageWindows(CustomTestCase):
    def test_invalid_document_limits_are_rejected(self):
        for field, values, flag in (
            ("uocr_doc_page_window", (0, -3), "--uocr-doc-page-window"),
            ("uocr_max_page_tokens", (-1,), "--uocr-max-page-tokens"),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    server_args = ServerArgs(model_path="dummy", **{field: value})
                    with self.assertRaisesRegex(ValueError, flag):
                        server_args.resolve_once()

    def test_a_one_page_window_is_accepted(self):
        ServerArgs(
            model_path="dummy", uocr_doc_split=True, uocr_doc_page_window=1
        ).resolve_once()


class TestUnsupportedServingModes(CustomTestCase):
    """Each of these launches would start and then serve wrong or hung requests,
    so the v2 override must refuse them at resolution."""

    def test_the_default_launch_is_accepted(self):
        overrides = _declared_overrides()
        self.assertEqual(overrides["attention_backend"], "unlimited_ocr_v2")
        self.assertEqual(overrides["page_size"], 32)

    def test_mps_enables_startup_warmup_and_window_kv_cap_by_default(self):
        from unittest import mock

        from sglang.srt.arg_groups.model_overrides import unlimited_ocr_v2 as uocr_ov

        def run(*, mem_mib=12124, **fields):
            sa = ServerArgs(model_path="/tmp/uocr", device="mps", **fields)
            with (
                mock.patch.object(uocr_ov, "use_mlx", return_value=False),
                mock.patch.object(
                    uocr_ov, "get_device_memory_capacity", return_value=mem_mib
                ),
            ):
                uocr_ov.apply_unlimited_ocr_v2_mps_defaults(sa)
            return resolving_view(sa)

        split = dict(uocr_doc_split=True, uocr_max_page_tokens=8000)
        # 16 GB Mac (~12 GiB recommended working set) -> 1 window (15,168 tokens).
        resolved_16g = run(**split)
        self.assertEqual(resolved_16g.warmups, "unlimited_ocr_v2")
        self.assertEqual(resolved_16g.max_total_tokens, 15168)
        # 64 GB Mac (~48 GiB working set) scales up to 5 windows (75,840 tokens).
        resolved_64g = run(mem_mib=48 * 1024, **split)
        self.assertEqual(resolved_64g.max_total_tokens, 5 * 15168)
        # The per-window slack is one prefill chunk, so a larger chunk widens it.
        self.assertEqual(
            run(chunked_prefill_size=4096, **split).max_total_tokens, 15168 + 2048
        )
        self.assertIsNone(run(skip_server_warmup=True).warmups)
        self.assertEqual(run(warmups="custom").warmups, "custom")
        self.assertEqual(run(max_total_tokens=16384, **split).max_total_tokens, 16384)
        self.assertIsNone(run(uocr_doc_split=False).max_total_tokens)

    def test_data_parallelism_is_rejected_while_documents_split(self):
        for fields in ({"dp_size": 2}, {"attn_dp_size": 2, "tp_size": 2}):
            with self.subTest(**fields):
                with self.assertRaisesRegex(ValueError, "--uocr-doc-split"):
                    _declared_overrides(uocr_doc_split=True, **fields)

    def test_data_parallelism_is_accepted_without_doc_split(self):
        for fields in ({"dp_size": 2}, {"attn_dp_size": 2, "tp_size": 2}):
            with self.subTest(**fields):
                _declared_overrides(**fields)

    def test_speculative_decoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "speculative decoding"):
            _declared_overrides(speculative_algorithm="NGRAM")

    def test_pipeline_parallelism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            _declared_overrides(pp_size=2)

    def test_pd_disaggregation_is_rejected(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                    _declared_overrides(disaggregation_mode=mode)


if __name__ == "__main__":
    unittest.main()
