import unittest

from sglang.srt.utils import is_sm100_supported, is_xpu, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import MMLUSanityMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.spec_server_kits import SpecGrammarKit, SpecLogprobKit
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=97, stage="base-b", runner_config="1-gpu-large")
register_xpu_ci(est_time=1800, suite="stage-b-test-1-gpu-xpu")

TARGET_MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"

# trtllm_mha prefill requires SM100 (Blackwell); use the Hopper-native pair elsewhere.
if is_sm100_supported():
    CUDA_ATTENTION_BACKEND = "trtllm_mha"
    CUDA_DRAFT_ATTENTION_BACKEND = "fa4"
else:
    CUDA_ATTENTION_BACKEND = "fa3"
    CUDA_DRAFT_ATTENTION_BACKEND = "fa3"


class _DSparkSanityMixin(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    MMLUSanityMixin,
    JSONConstrainedMixin,
    SpecGrammarKit,
    SpecLogprobKit,
):
    """DSpark sanity coverage shared by every backend pairing. Concrete
    subclasses add CustomTestCase, set the (attention_backend,
    draft_attention_backend) pair, and any per-backend launch overrides.
    Not a TestCase itself, so it is never collected on its own."""

    served_model_name = TARGET_MODEL
    model = TARGET_MODEL

    fwd_occupancy_threshold = 60
    fwd_occupancy_max_new_tokens = 4096
    fwd_occupancy_acc_length_threshold: float = 2.0

    mmlu_score_threshold = 0.70
    mmlu_accept_length_thres = 3.0

    # Set per concrete subclass.
    attention_backend: str = None
    draft_attention_backend: str = None
    page_size: str = "1"
    mem_fraction_static: str = "0.7"
    # None leaves the scheduler uncapped; XPU classes cap it (see below).
    max_running_requests: str = None
    extra_launch_args: list = []

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                cls.attention_backend,
                "--speculative-draft-attention-backend",
                cls.draft_attention_backend,
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                cls.mem_fraction_static,
                "--page-size",
                cls.page_size,
                "--enable-metrics",
                "--cuda-graph-backend-prefill=disabled",
                *(
                    ["--max-running-requests", cls.max_running_requests]
                    if cls.max_running_requests is not None
                    else []
                ),
                *cls.extra_launch_args,
            ],
            env={
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


@unittest.skipIf(is_xpu(), "CUDA/AMD backend pairing; XPU is covered separately")
class TestBasicSanityDSpark(_DSparkSanityMixin, CustomTestCase):
    attention_backend = CUDA_ATTENTION_BACKEND
    draft_attention_backend = CUDA_DRAFT_ATTENTION_BACKEND


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuTriton(_DSparkSanityMixin, CustomTestCase):
    attention_backend = "triton"
    draft_attention_backend = "triton"
    # gsm8k drives 128 concurrent clients; the B60 tp2 DSpark path faults under
    # the resulting large decode batches, so cap the scheduler (see intel_xpu).
    max_running_requests = "16"
    # Two B60 cards in bfloat16 (fp16 produces garbage on XPU).
    extra_launch_args = ["--tp", "2", "--dtype", "bfloat16"]


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuIntelXpu(_DSparkSanityMixin, CustomTestCase):
    attention_backend = "intel_xpu"
    draft_attention_backend = "intel_xpu"
    # intel_xpu forces page_size 64/128; the DSpark draft's block forward runs
    # through its target-verify / draft-extend paths (topk <= 1). 0.85 leaves
    # room for the draft markov head alongside the 14B target KV pool on a 24GB
    # B60 (0.7 OOMs during draft build).
    page_size = "128"
    mem_fraction_static = "0.85"
    # The intel_xpu DSpark spec kernels DEVICE_LOST when the decode batch grows
    # into the mid-20s (a batch-size limit, not KV pressure -- the pool is ~22%
    # full at that point). running-req 20 is stable; 16 leaves margin under the
    # ~24 crash onset while gsm8k's 128 client threads would otherwise blow past.
    max_running_requests = "16"
    # Two B60 cards in bfloat16 (fp16 produces garbage on XPU).
    extra_launch_args = ["--tp", "2", "--dtype", "bfloat16"]


if __name__ == "__main__":
    unittest.main()
