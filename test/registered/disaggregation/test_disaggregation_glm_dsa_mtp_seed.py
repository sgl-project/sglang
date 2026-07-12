"""E2E guard for DSA indexer-topk *seed* transfer across a PD-disaggregated
MTP deployment (PR #30839).

GLM-5.2 (and the DeepSeek-V3.2/V4 DSA family) reuse the indexer top-k selected
at the last verified token (`index_share_for_mtp_iteration`, topk=1 only) as the
"seed" for every MTP draft-decode step, instead of recomputing the sparse-attn
indexer each step. In colocated serving that seed lives on `EagleDraftInput`
from the draft-extend forward straight into the draft-decode loop.

Under PD disaggregation the draft-extend runs on the *prefill* node and the
draft-decode loop runs on the *decode* node, so the seed has to be carried over
the KV-transfer wire (a new `output_dsa_topk_indices` metadata buffer). If that
transfer is missing/misaligned/zeroed, the first MTP iteration after every
prefill->decode handoff seeds the indexer wrong, which shows up as a measurable
accept-length drop (and, with a corrupt seed, degraded accuracy) — even though
nothing crashes, because the worker silently falls back to eager recompute.

This test pins both observables on the wire-transferred path:
  * GSM8K accuracy floor          -> catches a corrupt/misaligned seed.
  * accept_length floor           -> catches a dropped/zeroed seed (eager
                                      recompute or wrong indices degrade it).

The accept_length floor is the sharp, PR-specific signal: it is only reachable
when the decode side receives the same seed colocated would have computed.

Sized for the 4-GPU B200 runner (prefill TP=2 + decode TP=2), mirroring
`test_dsa_glm52_cache_layer_split.py`, which is the proven PD topology for this
checkpoint. NVFP4 is Blackwell-only (no Hopper FP4), so this cannot run on the
H200 disaggregation runners.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=1500, stage="extra-b", runner_config="4-gpu-b200")

# NVFP4 (NVIDIA Model Optimizer) checkpoint. `--quantization modelopt_fp4` is the
# canonical launch flag for every NVFP4 cell in the GLM-5.2 cookbook.
GLM_DSA_NVFP4_MODEL = "nvidia/GLM-5.2-NVFP4"

# 5-1-6 (num_steps=5, topk=1, draft_tokens=6): the low-latency MTP recipe. More
# draft-decode iterations per handoff than 1-1-2, so the reused seed is exercised
# repeatedly and a broken transfer is more visible in accept_length. topk=1 is a
# hard precondition for index_share_for_mtp_iteration.
_EAGLE_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "6",
]


class TestDisaggregationGlmDsaMtpSeed(
    SpecDecodingMixin, PDDisaggregationServerBase, GSM8KMixin
):
    model = GLM_DSA_NVFP4_MODEL

    # Accuracy: MTP verify is exact w.r.t. greedy, so a correct seed keeps the
    # NVFP4 accuracy the colocated deployment reaches; a corrupt seed corrupts
    # the indexer selection and drops it. Threshold tracks the proven NVFP4 PD
    # floor (see test_dsa_glm52_cache_layer_split.py: 0.935 on 1319 questions).
    gsm8k_accuracy_thres = 0.92
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 200
    gsm8k_num_shots = 0

    # accept_length: the PR-specific signal. A 5-step draft with a correct seed
    # sits well above ~2.5; a dropped/zeroed seed (eager recompute or wrong
    # first-iteration indices) degrades it below. Speed is a loose secondary
    # floor (PD + TP2 on a 700B NVFP4 model); tune to the runner.
    accept_length_thres = 2.5
    bs_1_speed_thres = 40

    # Prefill worker (GPUs 0-1). Produces the draft-extend indexer top-k that is
    # copied into req.output_dsa_topk_indices and shipped over the wire.
    extra_prefill_args = [
        "--tp",
        "2",
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        *_EAGLE_SPEC_ARGS,
    ]
    # Decode worker (GPUs 2-3). Consumes the transferred seed on the first MTP
    # iteration after each handoff. bs>1 (cuda-graph-max-bs 16 + GSM8K's 200
    # threads) is required to exercise the all-or-nothing seed sentinel: a batch
    # mixing seeded and seedless reqs must drop to eager, not feed a partial seed
    # to the draft graph.
    extra_decode_args = [
        "--tp",
        "2",
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--cuda-graph-max-bs",
        "16",
        "--max-running-requests",
        "16",
        "--base-gpu-id",
        "2",
        *_EAGLE_SPEC_ARGS,
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
