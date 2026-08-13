"""Bit-exact KL guards for the unified radix cache on a hybrid SWA + mamba model.

The other KL tests in this directory gate on a loose threshold because their
models cannot score a token identically twice: Qwen3-Next's chunkwise prefill
scan and its decode recurrence are different algorithms and land an ulp apart, so
a tight floor there would fail on float noise. The shrunken Inkling checkpoint
reproduces every logprob exactly under deterministic inference, which turns the
same comparison into an exact one -- any nonzero KL is a state-reuse bug. It also
fits on one GPU, so these run per-commit rather than on a 4-GPU stage.

Each class below reproduces a specific merged regression when its fix is
reverted; the measured pre-fix divergence is recorded in the class docstring so a
later threshold change has to argue with a number.

These classes do not use UnifiedRadixTreeTestMixin: it bundles gsm8k and mmlu,
which an undertrained checkpoint cannot gate on, and each class here runs the
harness its regression was actually reproduced with.

The imported `test_`-prefixed helpers are aliased so pytest does not collect them
as tests.
"""

import os
import random
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    make_mamba_decode_assert,
)
from sglang.test.kl_multiturn_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_multiturn_decode_cache_hit,
)
from sglang.test.kl_test_utils import (
    get_input_ids,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_decode_cache_hit,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_prefill_cache_hit_helper as assert_prefill_cache_hit,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

_MODEL_PATH = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")

# Both classes measure exactly 0 in their fixed state -- every logprob matches bit
# for bit. The floor only keeps a stray ulp from failing the run; a state-reuse
# bug lands orders of magnitude above it. It cannot be 0.0: the comparison is a
# strict `<`, so an exact 0 would fail its own threshold.
KL_DIV_THRESHOLD = 1e-9

# Equal to the page size below. Out-of-window SWA slots are freed a page at a
# time, so only a checkpoint sitting on a page boundary still has a full window of
# SWA data below it -- at the default 256 half the sequence lengths land off that
# boundary and lose their decode prefix entirely.
TRACK_INTERVAL = 128
PAGE_SIZE = 128

# Past the 512-token sliding window, so decode carries the window through the
# handover from prompt tokens to generated ones.
MAX_NEW_TOKENS = 1024


def _random_suffixes(n: int, length: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


def _base_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--attention-backend",
        "fa4",
        "--page-size",
        str(PAGE_SIZE),
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        # 0.85 was carried over from the 4-GPU B200 test and OOMs an 80 GB card:
        # the static pool leaves ~19 GB for the prefill graphs, the fa4 workspace
        # and the chunked-prefill activations, which is what this config needs.
        "--mem-fraction-static",
        "0.6",
        "--mamba-track-interval",
        str(TRACK_INTERVAL),
        "--enable-deterministic-inference",
    ]


class TestUnifiedHybridBitExact(CustomTestCase):
    """Prefill and decode must score a token identically once every kernel on the
    path is batch-invariant, so any drift is a stale conv/mamba checkpoint or a
    prefix the cache restored wrong.

    Guards #34184 (stale track rows corrupting conv checkpoints under the prefill
    graph). Reverting that fix here measures avg_kl_div 5.58e-07 on
    test_logprobs_match and 6.22e-06 on test_prefill_cache_hit, against 0.0 with
    it in place. test_decode_cache_hit is 0.0 either way -- it guards decode-region
    state reuse in general, not that regression.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = _base_args() + [
            # Pinned, not incidental: the prefill graph derives its fixed
            # request-slot count from this (chunked_prefill_size // 512), and those
            # slots are exactly what #34184 left stale. Lowering it shrinks the
            # sentinel tail and the guard stops firing while still passing.
            "--chunked-prefill-size",
            "16384",
        ]
        if _MODEL_REVISION:
            other_args += ["--revision", _MODEL_REVISION]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def _run(self, helper):
        helper(
            self.base_url,
            {self.model: {"kl_div": KL_DIV_THRESHOLD}},
            self.model,
            max_samples=32,
            max_new_tokens=MAX_NEW_TOKENS,
            trust_remote_code=True,
        )

    def test_logprobs_match(self):
        self._run(assert_logprobs_match)

    def test_prefill_cache_hit(self):
        self._run(assert_prefill_cache_hit)

    def test_decode_cache_hit(self):
        self._run(assert_decode_cache_hit)


class TestUnifiedHybridHiCacheBitExact(CustomTestCase):
    """Same exactness bar with the host tier in the loop, over interleaved
    branches so hits land at many prefix lengths rather than one aligned one.

    Guards #29792 (decode track save picking its slot from the producer-side
    pointer). Without that fix this measures avg_kl_div 9.43e-06 and 1.16e-05 over
    two rounds, with 3 of 9 samples dirty and the rest exactly 0; with it in place
    both rounds are 0.0.

    Runs the multi-turn branching harness because the single-turn helpers above
    cannot produce a non-aligned hit length, which this regression needs.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = _base_args() + [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "4",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "direct",
            # The mamba host pool only supports page_first and page_first_direct.
            "--hicache-mem-layout",
            "page_first_direct",
            # Tight pools and a small budget so decode crosses a track boundary and
            # the host tier is actually exercised instead of everything staying
            # resident on device.
            "--chunked-prefill-size",
            "2048",
            "--max-total-tokens",
            "65536",
            "--max-mamba-cache-size",
            "500",
            "--max-running-requests",
            "4",
        ]
        if _MODEL_REVISION:
            other_args += ["--revision", _MODEL_REVISION]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=9, trust_remote_code=True
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def test_multiturn_decode_cache_hit_branching(self):
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
            # Not the default exact equality: a mamba checkpoint lands on a track
            # boundary, so the reusable prefix is floor-aligned to the interval.
            assert_decode_cached_tokens=make_mamba_decode_assert(TRACK_INTERVAL),
            branches_per_group=branches,
            max_new_tokens=512,
            sampling_temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
