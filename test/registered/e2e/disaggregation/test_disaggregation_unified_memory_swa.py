"""PD disaggregation for a hybrid-SWA model on the unified memory pool.

A hybrid-SWA model ships TWO attention components: the full-attention KV on the
ordinary `kv_data_ptrs` channel and the sliding-window KV as `StateType.SWA`.
Under `--enable-unified-memory` both are whole page envelopes into the SAME raw
buffer, distinguished only by their per-page stride, and each is addressed by
its OWN sub-pool's physical page id -- the full and SWA sides run independent
compactions, so one virtual token names two unrelated physical pages.

That makes three ways to be silently wrong rather than loud:
  * shipping virtual ids (the base `translate_kv_indices_for_transfer` is the
    identity, and virtual ids address real bytes);
  * shipping the SWA side's KERNEL-FACING ids, which the read path uses, in
    place of its physical ones;
  * letting compaction relocate a page mid-transfer, which the SWA allocator
    had no `set_disagg_move_gate` to prevent.

Logprob parity against a non-PD unified reference catches all three; GSM8K on
gpt-oss is too noisy to (single-server unified and static both score 0.570 at
200 questions, and PD runs of each span 0.540-0.610).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="2-gpu-large")

UNIFIED_SWA_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-unified-memory",
    # gpt-oss uses attention sinks, which flashinfer does not support; triton
    # reads both sub-pools' per-layer views.
    "--attention-backend",
    "triton",
    "--mem-fraction-static",
    "0.7",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


class TestUnifiedMemoryDisaggregationSWA(
    PDLogprobParityMixin, PDDisaggregationServerBase
):
    """1 prefill + 1 decode, both unified, vs a non-PD unified reference."""

    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 1
    baseline_args = UNIFIED_SWA_ARGS
    extra_prefill_args = UNIFIED_SWA_ARGS
    extra_decode_args = UNIFIED_SWA_ARGS


if __name__ == "__main__":
    unittest.main()
