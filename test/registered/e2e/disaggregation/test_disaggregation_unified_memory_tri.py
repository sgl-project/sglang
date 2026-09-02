"""PD disaggregation for a TRI-pool model on the unified memory pool.

Inkling is the only in-tree architecture that is both mambaish and hybrid-SWA,
so one unified buffer carries three components with three independent
compactions -- ``[conv state (up END) | swa (FLOAT) | full (down END)]`` -- and
PD must ship all three per request: full KV on the ``kv_data_ptrs`` channel,
sliding-window KV as ``StateType.SWA``, ShortConv state as ``StateType.MAMBA``
(via the ``req_to_token_pool`` fallback, since the KV pool here is a
``UnifiedSWAKVPool`` rather than a ``HybridLinearKVPool``).

Two failures this pins that the 2-pool cases cannot:

  * the FLOAT sub-pool moves for reasons neither END does, so a move gate that
    reaches only full and swa still lets a conv slot relocate under an
    in-flight state transfer;
  * ``page_size > 1`` turns on the decode node's SWA-tail prealloc, whose
    static body allocates the swa side independently -- an assertion failure
    against this composite's single virtual id space, and, once that is
    handled, the first path that can bind the WRONG swa pages.

Logprob parity against a non-PD unified reference is the check: the tiny
``test`` revision is undertrained, so answer quality carries no signal, but a
dropped or misaddressed component moves logprobs immediately.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=900, stage="extra-b", runner_config="2-gpu-large")

_MODEL_PATH = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")

# The unified radix tree is what merges the three components into one tree.
SERVER_ENV = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}

UNIFIED_TRI_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-unified-memory",
    # Unified requires the Triton strided page-major read/write paths.
    "--attention-backend",
    "triton",
    "--page-size",
    "128",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--swa-full-tokens-ratio",
    "0.1",
    "--mamba-full-memory-ratio",
    "0.1",
    "--mem-fraction-static",
    "0.5",
    # Inkling defaults to a FULL prefill graph, which unified rejects at boot.
    "--cuda-graph-backend-prefill",
    "disabled",
    "--revision",
    _MODEL_REVISION,
]


class TestUnifiedMemoryDisaggregationTriPool(
    PDLogprobParityMixin, PDDisaggregationServerBase
):
    """1 prefill + 1 decode, both unified, vs a non-PD unified reference."""

    model = _MODEL_PATH
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 1
    baseline_args = UNIFIED_TRI_ARGS
    extra_prefill_args = UNIFIED_TRI_ARGS
    extra_decode_args = UNIFIED_TRI_ARGS


if __name__ == "__main__":
    unittest.main()
