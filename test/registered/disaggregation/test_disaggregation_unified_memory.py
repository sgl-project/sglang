import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

KIMI_LINEAR_MODEL = "yujiepan/kimi-linear-tiny-random"
# Smallest in-tree GDN hybrid: MHA full attention + gated-delta-net linear
# layers, i.e. the unified pool's MHA sub-pool rather than the MLA one.
QWEN_GDN_MODEL = "Qwen/Qwen3.5-0.8B"
SERVER_ENV = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0"}

# --attention-backend and --enable-deterministic-inference are deliberately
# absent: bytes are backend-independent and both paths run identical shapes.
UNIFIED_MEMORY_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-unified-memory",
    # Page-major layout requires triton for both.
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
    "--max-mamba-cache-size",
    "32",
    "--max-total-tokens",
    "4096",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


class TestUnifiedMemoryDisaggregation(PDLogprobParityMixin, PDDisaggregationServerBase):
    """1 prefill + 1 decode, both with --enable-unified-memory, vs a non-PD
    unified-memory reference server."""

    model = KIMI_LINEAR_MODEL
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 1
    baseline_args = UNIFIED_MEMORY_ARGS
    extra_prefill_args = UNIFIED_MEMORY_ARGS
    extra_decode_args = UNIFIED_MEMORY_ARGS


class TestUnifiedMemoryDisaggregationChunkedPrefill(TestUnifiedMemoryDisaggregation):
    """Multi-chunk prefill (257-token prompt, 64-token chunks): each chunk's KV
    pages are translated to physical ids and shipped while later chunks still
    run, exercising the chunked send path and the prefill-side move gate
    (`chunked_req.start_send_idx > 0`). The reference server uses the same
    chunk size so any parity break isolates to the PD transfer.
    """

    _chunked_args = UNIFIED_MEMORY_ARGS + ["--chunked-prefill-size", "64"]
    baseline_args = _chunked_args
    extra_prefill_args = _chunked_args
    extra_decode_args = _chunked_args


class TestUnifiedMemoryDisaggregationMHA(TestUnifiedMemoryDisaggregation):
    """The MHA full-attention sub-pool over the wire.

    Kimi-Linear above exercises the MLA sub-pool, whose whole-envelope
    registration has always been the one PD supports. An MHA envelope is a
    different shape -- `2 * layer_num` row-blocks per page instead of
    `layer_num` -- and it reaches a different branch of
    `_send_kvcache_generic`: without `force_flat` the MHA branch halves the
    single registered region into K and V, computes `num_kv_layers = 0` and
    transfers NOTHING, which shows up as garbage decode rather than an error.
    Logprob parity against a non-PD unified reference is what catches that.
    """

    model = QWEN_GDN_MODEL


if __name__ == "__main__":
    unittest.main()
