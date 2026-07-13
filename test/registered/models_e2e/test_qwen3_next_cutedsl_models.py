"""Qwen3-Next e2e accuracy + logprob-KL on the CuteDSL GDN prefill backend.

Lives in its own file because the CuteDSL prefill kernels are Blackwell-only:
the sibling Triton configs (test_qwen3_next_models.py) register on 4-gpu-h100,
where this class would skip unconditionally and never run.
"""

import unittest

from sglang.srt.utils import is_blackwell
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=250, stage="base-c", runner_config="4-gpu-b200")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# no_buffer (not extra_buffer_lazy) is deliberate: the CuteDSL extend returns
# h=None, so it cannot write the SSM prefix checkpoints that extra_buffer reuses
# on a prefix-cache hit -- pairing CuteDSL with extra_buffer would corrupt
# outputs on shared-prefix requests (e.g. GSM8K few-shot) — a CuteDSL
# limitation. no_buffer recomputes mamba state on cache hits, isolating the
# CuteDSL compute path. no_buffer requires page_size=1,
# --disable-overlap-schedule, and a non-trtllm_mha attention backend.
_CUTEDSL_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-scheduler-strategy",
    "no_buffer",
    "--disable-overlap-schedule",
    "--attention-backend",
    "triton",
    "--linear-attn-prefill-backend",
    "cutedsl",
    "--page-size",
    "1",
]


# Belt-and-braces: the b200 runner satisfies this, but a mis-scheduled run on
# older hardware must skip rather than silently test the Triton fallback.
@unittest.skipUnless(
    is_blackwell(),
    "CuteDSL GDN prefill requires SM100+ (Blackwell).",
)
class TestQwen3NextCuteDSLPrefill(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    """CuteDSL GDN prefill path: GSM8K accuracy + logprob-KL check.

    Same model/thresholds as the Triton configs but forces the CuteDSL GDN
    prefill kernels via --linear-attn-prefill-backend cutedsl.
    """

    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _CUTEDSL_ARGS


if __name__ == "__main__":
    unittest.main()
