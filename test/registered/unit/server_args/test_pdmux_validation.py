import tempfile
import unittest
from pathlib import Path

from sglang.srt.arg_groups.validation_hook import check_server_args
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def validate(**overrides):
    args = dict(
        model_path="dummy",
        served_model_name="dummy",
        page_size=16,
        chunked_prefill_size=8192,
        enable_pdmux=True,
        disable_overlap_schedule=True,
    )
    args.update(overrides)
    check_server_args(ServerArgs(**args))


class TestPDMuxValidation(unittest.TestCase):
    def test_token_chunks_are_supported(self):
        validate()

    def test_ordinary_scheduler_is_unchanged(self):
        validate(enable_pdmux=False, disable_overlap_schedule=False)

    def test_single_layer_mtp_and_dspark_are_supported(self):
        for algorithm in ("EAGLE", "NEXTN", "DSPARK"):
            validate(speculative_algorithm=algorithm)

    def test_unsupported_speculative_combinations_are_rejected(self):
        for options, message in (
            (dict(speculative_algorithm="NGRAM"), "MTP/EAGLE and DSpark"),
            (
                dict(speculative_algorithm="EAGLE", enable_multi_layer_eagle=True),
                "multi-layer",
            ),
            (
                dict(speculative_algorithm="EAGLE", speculative_adaptive=True),
                "fixed speculative",
            ),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(AssertionError, message):
                    validate(**options)

    def test_attention_dp_is_supported(self):
        for size in (2, 8):
            validate(enable_dp_attention=True, tp_size=8, attn_dp_size=size)

    def test_mixed_chunks_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "mixed prefill/decode"):
            validate(enable_mixed_chunk=True)

    def test_cli_and_yaml_group_counts_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pdmux.yaml"
            path.write_text("sm_group_num: 3\n")
            with self.assertRaisesRegex(AssertionError, "must match"):
                validate(pdmux_config_path=str(path), sm_group_num=8)
            validate(pdmux_config_path=str(path), sm_group_num=3)


if __name__ == "__main__":
    unittest.main()
