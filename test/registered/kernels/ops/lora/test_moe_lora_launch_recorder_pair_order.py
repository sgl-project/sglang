"""The launch recorder on the providers that keep rows in pair order.

The Triton and Marlin providers sort the routed pairs by expert behind an
index and leave every row at its pair's position. The cases of
``test_moe_lora_launch_recorder`` cover the CuTeDSL providers and the Marlin
runner tests; these add the Triton providers and the Marlin tiles that pad
the intermediate width, as a record of their own.

Operands: the Triton BF16 and 128-block FP8 providers run the base pipeline
of the recorder test twice on identical inputs, with dropped pairs in the
batch. The prepared routing (the pair map, the expert of every row block,
the padded row count) and, by logical pair, the data at every launch
boundary must be the same in both forwards, and so must the outputs. The
sorted pair index itself is not compared: the order inside an expert's rows
is the aligner's to choose.

Structure: existing provider and runner tests on the Triton vendor and on
padded Marlin tiles run three times under the recorder; the second and
third launch sequences must be identical.

A run that keeps a junit record keeps the cases beside it, as the recorder
test does; two such files compare with
``python -m sglang.test.kernels.moe_lora_launch_recorder before after``.
"""

from __future__ import annotations

import pytest
import test_moe_lora_fp8_channel_provider_numerics as C
import test_moe_lora_fp8_marlin_provider_numerics as M
import test_moe_lora_fp8_provider_numerics as F
import test_moe_lora_fp8_tensor_provider_numerics as T
import test_moe_lora_launch_recorder as L
import test_moe_lora_mxfp4_marlin_provider_numerics as X
import test_moe_lora_nvfp4_provider_numerics as V
import test_moe_lora_prepared_input_reuse as R
import test_moe_lora_shared_outer_numerics as S

from sglang.srt.lora.utils import Phase
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import cutedsl_ready

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# SM100 selects its own Triton tile configurations: one GPU of the pool is used.
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not cutedsl_ready(), reason="the reused weights fixture needs the CuTeDSL stack"
)


def _prepared(state):
    """Row-state tensors that identical inputs must rebuild byte for byte: the
    pair map, the padded row count and the expert of every live row block. The
    aligner sizes the block buffer for the worst case and leaves the rest
    unwritten."""
    blocks = int(state.num_tokens_post_padded) // int(state.config["BLOCK_SIZE_M"])
    return {
        "pair_to_row": (state.pair_to_row, None),
        "num_tokens_post_padded": (state.num_tokens_post_padded, None),
        "expert_ids": (state.expert_ids, lambda ids: ids[:blocks]),
    }


@pytest.fixture(
    scope="module", params=("bf16", "fp8"), ids=("bf16-triton", "fp8-triton")
)
def triton_case(request):
    for provider, w13, w2 in R.providers(request.param, "route_major", "triton"):
        yield provider, w13, w2, request.param


def test_pair_order_forwards_agree_at_every_launch_boundary(
    triton_case, record_property, request
) -> None:
    provider, w13, w2, family = triton_case
    L.fresh_forwards_case(
        provider,
        w13,
        w2,
        family,
        record_property,
        request,
        kinds={"triton"},
        fixture=dict(
            seed=L._SEED,
            tokens=L._TOKENS,
            traffic="padded",
            family=family,
            vendor="triton",
        ),
        prepared=_prepared,
    )


_REUSED_CASES = {
    "bf16-triton-shared-outer-decode": (
        lambda mp: S.test_shared_outer_eager_and_graph_replay_match_fp32_reference(
            mp, "triton", Phase.DECODE
        ),
        {"triton"},
    ),
    "bf16-triton-shared-outer-prefill": (
        lambda mp: S.test_shared_outer_eager_and_graph_replay_match_fp32_reference(
            mp, "triton", Phase.PREFILL
        ),
        {"triton"},
    ),
    "fp8-triton-provider": (
        lambda mp: F.test_fp8_gateup_and_down_match_reference(
            "triton", "route_major", F._H
        ),
        {"triton"},
    ),
    "fp8_channel-triton-provider": (
        lambda mp: C.test_fp8_channel_gateup_and_down_match_reference(
            "triton", "route_major", C._H
        ),
        {"triton"},
    ),
    "fp8_tensor-triton-provider": (
        lambda mp: T.test_fp8_tensor_dynamic_scales_match_reference(T._H),
        {"triton"},
    ),
    # Intermediate widths the Marlin tiles pad: the padded gate/up and
    # activation buffers are operands of these launches.
    "fp8_channel-marlin-padded-provider": (
        lambda mp: M.test_fp8_marlin_provider_serves_64_aligned_intermediates(
            "fp8_channel", 320
        ),
        {"marlin"},
    ),
    "mxfp4-marlin-padded-provider": (
        lambda mp: X.test_mxfp4_marlin_gateup_and_down_match_reference(2, 96),
        {"marlin"},
    ),
    "nvfp4-marlin-provider": (
        lambda mp: V.test_nvfp4_marlin_gateup_and_down_match_reference(2, V._I),
        {"marlin"},
    ),
}


@pytest.mark.parametrize("case", _REUSED_CASES)
def test_pair_order_tests_launch_the_same_sequence_again(
    monkeypatch: pytest.MonkeyPatch, case: str, record_property, request
) -> None:
    L.reused_test_case(
        *_REUSED_CASES[case], case, monkeypatch, record_property, request
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
