"""SGLANG_DSPARK_VERIFY_MERGED_MIXED falls back (warning, no crash) outside
DSPARK + --enable-mixed-chunk without DP attention / PP."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.arg_groups.validation_hook import verify_merged_mixed_enabled
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kw):
    d = dict(
        speculative_algorithm="DSPARK",
        enable_mixed_chunk=True,
        enable_dp_attention=False,
        pp_size=1,
        enable_decoder_swa_bounded_replay=False,
        enable_two_batch_overlap=False,
    )
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.mark.parametrize(
    "args,reason",
    [
        (_args(), None),
        (_args(speculative_algorithm="EAGLE"), "non-DSPARK"),
        (_args(speculative_algorithm=None), "non-DSPARK"),
        (_args(enable_mixed_chunk=False), "mixed chunk disabled"),
        (_args(enable_dp_attention=True), "DP attention"),
        (_args(pp_size=2), "pipeline parallelism"),
        (_args(enable_decoder_swa_bounded_replay=True), "decoder SWA bounded replay"),
        (_args(enable_two_batch_overlap=True), "two-batch overlap"),
    ],
)
def test_guard(args, reason):
    with (
        mock.patch.dict("os.environ", {"SGLANG_DSPARK_VERIFY_MERGED_MIXED": "1"}),
        mock.patch("sglang.srt.arg_groups.validation_hook.print_warning_once") as warn,
    ):
        assert verify_merged_mixed_enabled(args) is (reason is None)
    if reason is None:
        warn.assert_not_called()
    else:
        warn.assert_called_once()
        assert reason in warn.call_args.args[0]


def test_flag_off_is_upstream():
    with mock.patch.dict("os.environ", {"SGLANG_DSPARK_VERIFY_MERGED_MIXED": "0"}):
        assert verify_merged_mixed_enabled(_args()) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
