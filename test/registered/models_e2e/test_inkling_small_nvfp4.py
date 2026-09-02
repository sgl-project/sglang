"""Per-commit accuracy + logprob-consistency test for Inkling-Small-NVFP4.

``test_inkling.py`` boots a shrunken checkpoint, so it can only guard that the
code paths run -- an undertrained model has no answer quality to gate on. This
one serves the real NVFP4 checkpoint, which is what catches a weight-load or
FP4-kernel regression that keeps the server healthy while the outputs go wrong.

gsm8k here is few-shot completion, so it never renders a chat turn and never
reaches the reasoning path -- it gates the FP4 numerics, not answer quality
under thinking. tp=4 to match the runner; the checkpoint does not fit on fewer
cards.
"""

import os
import random
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci

# Aliased for the same reason as the single-turn helpers above.
from sglang.test.kl_multiturn_utils import (
    make_mamba_decode_assert,
)
from sglang.test.kl_multiturn_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_multiturn_decode_cache_hit,
)

# Aliased so pytest does not collect the imported `test_`-prefixed helpers as tests.
from sglang.test.kl_test_utils import (
    get_input_ids,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_logprobs_match_decode_cache_hit,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_prefill_cache_hit_helper as assert_logprobs_match_prefill_cache_hit,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=2000, stage="extra-b", runner_config="4-gpu-b200")

_MODEL_PATH = os.environ.get(
    "INKLING_SMALL_TEST_MODEL_PATH", "thinkingmachines/Inkling-Small-NVFP4"
)
_DSPARK_DRAFT_PATH = os.environ.get(
    "INKLING_SMALL_DSPARK_DRAFT_PATH", "RadixArk/Inkling-Small-DSpark"
)

# Measured 0.900 (10-shot, 200 questions, tp=4, invalid=0.000) -- completion,
# so no thinking. The floor sits ~4.5 sigma of the 200-question sampling noise
# below that: a real accuracy collapse trips it, the sample spread does not.
GSM8K_THRESHOLD = 0.80

# All three helpers measure exactly 0 on this config -- every logprob matches
# bit for bit. The floor is only there to keep a stray ulp from failing the
# run; the divergence a state-reuse bug produces lands orders of magnitude
# above it.
KL_DIV_THRESHOLD = 1e-9

# Past the 512-token sliding window, so decode carries the window through the
# handover from prompt tokens to generated ones -- where a stale conv/mamba
# checkpoint or a mis-restored prefix would surface.
KL_MAX_NEW_TOKENS = 1024

# Equal to the page size below. Out-of-window SWA slots are freed a page at a
# time, so only a checkpoint sitting on a page boundary still has a full window
# of SWA data below it -- at the default 256 half the sequence lengths land off
# that boundary and lose their decode prefix entirely.
KL_TRACK_INTERVAL = 128


class TestInklingSmallNvfp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
                "--attention-backend",
                "fa4",
                "--page-size",
                "128",
                "--fp4-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--swa-full-tokens-ratio",
                "0.1",
                "--mamba-full-memory-ratio",
                "0.1",
                "--mem-fraction-static",
                "0.85",
            ],
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k(self):
        """Answer quality on the real checkpoint: guards the modelopt_fp4 weight
        load and the FP4 GEMM/MoE kernels against changes that keep the server
        healthy but corrupt the numerics."""
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        metrics = run_few_shot_gsm8k(
            SimpleNamespace(
                num_shots=10,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{url.hostname}",
                port=int(url.port),
            )
        )
        print(f"[{self.__class__.__name__}] gsm8k: {metrics['accuracy']:.3f}")
        self.assertGreaterEqual(metrics["accuracy"], GSM8K_THRESHOLD)


class TestInklingSmallNvfp4DsparkDeterministic(CustomTestCase):
    """Prefill and decode must score a token identically once every kernel on
    the path is batch-invariant, which is what deterministic inference buys.
    Drift here is then a state-reuse bug -- a stale conv/mamba checkpoint, or a
    prefix restored from the radix cache that does not reproduce a fresh
    prefill -- rather than the float noise a loose threshold would hide.

    DSPARK drives the decode loop because the spec-side mamba/sconv save is
    otherwise unreachable: that gate lives in PrefillCudaGraphRunner, and EAGLE
    targets disable the prefill graph outright (#28386), so the MTP class in
    test_unified_radix_cache_kl_hybrid_bitexact.py never reaches it. Reverting
    #34043 reads 1.10e-01 on prefill_cache_hit here and exactly 0 without
    speculation.

    Runs its own server: the accuracy case above has to stay on the production
    numerics, so it cannot share this one.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
                "--attention-backend",
                "fa4",
                "--page-size",
                "128",
                "--fp4-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--swa-full-tokens-ratio",
                "0.1",
                "--mamba-full-memory-ratio",
                "0.1",
                # The draft weights and the speculative CUDA graphs need the
                # headroom; 0.85 OOMs mid-run on a 178 GB B200.
                "--mem-fraction-static",
                "0.80",
                "--mamba-track-interval",
                str(KL_TRACK_INTERVAL),
                "--enable-deterministic-inference",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                _DSPARK_DRAFT_PATH,
                "--speculative-draft-attention-backend",
                "fa4",
            ],
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def _run(self, helper, **kwargs):
        helper(
            self.base_url,
            {self.model: {"kl_div": KL_DIV_THRESHOLD}},
            self.model,
            max_samples=32,
            max_new_tokens=KL_MAX_NEW_TOKENS,
            trust_remote_code=True,
            **kwargs,
        )

    def test_input_output_logprobs_match(self):
        self._run(assert_logprobs_match)

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        self._run(assert_logprobs_match_prefill_cache_hit)

    def test_input_output_logprobs_match_decode_cache_hit(self):
        # Not every prompt: speculation commits up to block_size-1 tokens past
        # max_new_tokens, and those reach the radix insert but not the returned
        # output. For roughly one prompt in 32 that puts the request's only mamba
        # checkpoint past the prefix a follow-up turn can reach, and its decode
        # region is not reusable. Tighten to 0.99 once that is fixed.
        self._run(assert_logprobs_match_decode_cache_hit, min_cache_hit_ratio=0.9)


# The multi-turn branching harness, unlike the single-turn helpers above, replays
# turn N on top of turn N-1's history, so cache hits land at many different
# prefix lengths and nine interleaved branches share prefixes before diverging.
# That is what reaches the decode-side track save.
KL_HICACHE_TRACK_INTERVAL = 128


def _random_suffixes(n: int, length: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


class TestInklingSmallNvfp4HiCacheDeterministic(CustomTestCase):
    """HiCache round trip must be bit exact, since nothing about moving a state
    to host memory and back is allowed to change it.

    Every knob here is load bearing, and dropping any one of them takes the
    measurement back to zero even when the state layer is wrong:

    - hicache write_through, so an insert copies to host immediately
    - the decode CUDA graph, since the corrupted slot is only read through the
      captured decode path
    - a short ``--mamba-track-interval``, so decode actually crosses a track
      boundary within a turn
    - pools tight enough to force eviction while requests are live
    - multi-turn cache hits, because a wrong slot only surfaces once a later hit
      restores it
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
                "--attention-backend",
                "fa4",
                "--page-size",
                "128",
                "--fp4-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(KL_HICACHE_TRACK_INTERVAL),
                "--swa-full-tokens-ratio",
                "0.1",
                "--mamba-full-memory-ratio",
                "0.1",
                "--mem-fraction-static",
                "0.85",
                "--chunked-prefill-size",
                "2048",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--max-total-tokens",
                "65536",
                "--max-mamba-cache-size",
                "500",
                "--max-running-requests",
                "4",
                "--enable-deterministic-inference",
            ],
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_multiturn_decode_cache_hit_over_hicache(self):
        """Nine interleaved branches, three turns, decode hits served through the
        host tier. Also asserts every hit lands on a track boundary, which the
        logprob comparison cannot see on its own."""
        groups, branches = 3, 3
        n = groups * branches
        first_turn = []
        for g in range(groups):
            base = self.input_ids[g][:512]
            for _ in range(branches):
                first_turn.append(list(base))

        assert_multiturn_decode_cache_hit(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            first_turn,
            turn_suffixes=[
                _random_suffixes(n, 512, seed=300),
                _random_suffixes(n, 256, seed=400),
            ],
            assert_decode_cached_tokens=make_mamba_decode_assert(
                track_interval=KL_HICACHE_TRACK_INTERVAL
            ),
            branches_per_group=branches,
            max_new_tokens=KL_MAX_NEW_TOKENS,
        )


if __name__ == "__main__":
    unittest.main()
