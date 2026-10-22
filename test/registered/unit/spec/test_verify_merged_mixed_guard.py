"""SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION falls back (warning, no crash)
outside DSPARK + --enable-mixed-chunk without DP attention / PP."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.arg_groups.validation_hook import verify_merged_mixed_enabled
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kw):
    d = dict(
        speculative_algorithm="DSPARK",
        enable_mixed_chunk=True,
        attn_dp_size=1,
        ep_join_mode=None,
        pp_size=1,
        enable_decoder_swa_bounded_replay=False,
        enable_two_batch_overlap=False,
        dsv4_attn_backend="flashmla",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.BREAKABLE)
        ),
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
        (_args(attn_dp_size=2), "DP attention"),
        (_args(pp_size=2), "pipeline parallelism"),
        (_args(enable_decoder_swa_bounded_replay=True), "decoder SWA bounded replay"),
        (_args(enable_two_batch_overlap=True), "two-batch overlap"),
        (_args(dsv4_attn_backend="trtllm"), "--dsv4-attn-backend"),
        (_args(dsv4_attn_backend="flashmla_mega"), "--dsv4-attn-backend"),
        (
            _args(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend=Backend.FULL)
                )
            ),
            "prefill CUDA graph backend",
        ),
    ],
)
def test_guard(args, reason):
    with (
        envs.SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION.override(True),
        mock.patch("sglang.srt.arg_groups.validation_hook.print_warning_once") as warn,
    ):
        assert verify_merged_mixed_enabled(args) is (reason is None)
    if reason is None:
        warn.assert_not_called()
    else:
        warn.assert_called_once()
        assert reason in warn.call_args.args[0]


def test_hybrid_linear_attn_target():
    with (
        envs.SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION.override(True),
        mock.patch("sglang.srt.arg_groups.validation_hook.print_warning_once") as warn,
    ):
        assert not verify_merged_mixed_enabled(_args(), hybrid_linear_attn=True)
    assert "hybrid linear-attention" in warn.call_args.args[0]


def test_flag_off_is_upstream():
    with envs.SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION.override(False):
        assert verify_merged_mixed_enabled(_args()) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
