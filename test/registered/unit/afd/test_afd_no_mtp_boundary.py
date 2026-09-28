"""The MANF base admits native speculation only when AFD is off."""

from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace

import pytest

from sglang.srt.afd import config as afd_config
from sglang.srt.afd import contracts as afd_contracts
from sglang.test.afd.config_fixtures import _server_args


def test_afd_rejects_speculation_before_runtime_construction():
    args = _server_args(speculative_algorithm="EAGLE", speculative_num_steps=1)
    with pytest.raises(afd_contracts.AFDError, match="AFD_MTP_SPECULATIVE_UNSUPPORTED"):
        afd_config.validate_afd_server_args(args)


def test_native_speculation_is_not_restricted_by_afd():
    args = _server_args(
        afd_execution_mode="off",
        afd_config=None,
        speculative_algorithm="EAGLE",
        speculative_num_steps=1,
    )
    afd_config.validate_afd_server_args(args)
    assert args.speculative_algorithm == "EAGLE"
    assert args.speculative_num_steps == 1


def test_runtime_rejects_verify_rows_instead_of_treating_them_as_decode():
    with pytest.raises(afd_contracts.AFDError, match="AFD_MTP_SPECULATIVE_UNSUPPORTED"):
        afd_contracts.validate_non_speculative_batch(
            SimpleNamespace(spec_info=object())
        )
    assert (
        afd_contracts.validate_non_speculative_batch(SimpleNamespace(spec_info=None))
        is None
    )


@pytest.mark.parametrize("width", [0, -1, 2, 4, True, 1.0, "1"])
def test_request_width_is_rejected_by_sender_receiver_and_shape(width):
    """Keep the wire field but reject unsupported widths on both role boundaries."""
    contracts = afd_contracts
    descriptor = dict(
        kind="STEP",
        step_id=0,
        lane_stage_rows=((4, 4),),
        hidden_size=8,
        dtype="bfloat16",
        num_layers=1,
        graph_eligible=True,
        tokens_per_request=width,
    )
    with pytest.raises(contracts.AFDError, match="AFD_TOKENS_PER_REQUEST_INVALID"):
        contracts.AFDStepDescriptor(**descriptor)
    with pytest.raises(contracts.AFDError, match="AFD_TOKENS_PER_REQUEST_INVALID"):
        planned_shape(
            lane=0,
            lane_rows=((4, 4),),
            hidden_size=8,
            dtype="bfloat16",
            config=afd_config.AFDConfig(),
            tokens_per_request=width,
        )
    with pytest.raises(contracts.AFDError, match="AFD_TOKENS_PER_REQUEST_INVALID"):
        contracts.AFDPairedShape(
            lane=0,
            lane_stage_rows=((4, 4),),
            lane_bucket_rows=((8, 8),),
            hidden_size=8,
            dtype="bfloat16",
            tokens_per_request=width,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
