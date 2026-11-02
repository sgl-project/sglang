"""Unlimited-OCR v2 server-argument validation."""

import unittest

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
