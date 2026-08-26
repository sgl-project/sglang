"""Bit-exact KL guards for the unified radix cache on a hybrid SWA + mamba model.

The other KL tests in this directory gate on a loose threshold because their
models cannot score a token identically twice: Qwen3-Next's chunkwise prefill
scan and its decode recurrence are different algorithms and land an ulp apart, so
a tight floor there would fail on float noise. The shrunken Inkling checkpoint
reproduces every logprob exactly under deterministic inference, which turns the
same comparison into an exact one -- any nonzero KL is a state-reuse bug. It also
fits on one GPU, so these run per-commit rather than on a 4-GPU stage.

Reverting either #34184 or #29792 turns this file red, so a later threshold
change has to argue with a number. Which case fires is architecture-dependent
though, because prefill and decode take different fa4 kernels on SM90 and SM100.
Measured avg_kl_div:

                                             fix reverted                 fix
                                          SM100 (B200) SM90 (H200)   in place
  #34184   test_logprobs_match                5.58e-07         0.0        0.0
           test_prefill_cache_hit             6.22e-06    4.40e-06        0.0
           test_decode_cache_hit                   0.0         0.0        0.0
           multiturn branching                     0.0    2.01e-07        0.0
           lazy test_prefill_cache_hit        5.56e-06         n/a        0.0
  #29792   multiturn branching       9.43e-06/1.16e-05    5.14e-04        0.0

The lazy row was measured in the same session as the 1.18e-05 the non-lazy
`test_prefill_cache_hit` read on that GPU, so read the pair as "both fire",
not as one being weaker.

`test_prefill_cache_hit` is the only case that fires on both, so read the rest as
extra coverage rather than as the guard for one fix. CI runs `1-gpu-large`, which
is SM90.

These classes do not use UnifiedRadixTreeTestMixin: it bundles a gsm8k case an
undertrained checkpoint cannot gate on, and each class here runs the harness its
regression was actually reproduced with.

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

register_cuda_ci(est_time=1150, stage="base-b", runner_config="1-gpu-large")

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


def _base_args(mamba_strategy: str = "extra_buffer") -> list[str]:
    return [
        "--trust-remote-code",
        "--attention-backend",
        "fa4",
        "--page-size",
        str(PAGE_SIZE),
        "--mamba-radix-cache-strategy",
        mamba_strategy,
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
    graph) through test_prefill_cache_hit, which is the case that fires on both
    architectures; see the table at the top for what the other two measure.
    test_decode_cache_hit stays 0.0 even with the fix reverted, so it covers
    decode-region state reuse in general rather than that regression.
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


class TestUnifiedHybridLazyBitExact(TestUnifiedHybridBitExact):
    """Same exactness bar on the lazy extra-buffer strategy.

    Lazy is a different code path, not the same code under a flag: a request
    holds one ping-pong slot instead of two and the second is allocated only on
    the forward that crosses a track boundary, then freed again afterwards. The
    checkpoint therefore lands in a slot chosen per forward rather than one the
    request owns for its lifetime, and picking the wrong one restores another
    request's conv window.

    Admitted the same way the class it inherits from was: reverting #34184 takes
    test_prefill_cache_hit here to 5.56e-06, so this is a guard for that class of
    bug on the lazy path rather than coverage of a path nothing checks.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = _base_args("extra_buffer_lazy") + [
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


class TestUnifiedHybridHiCacheBitExact(CustomTestCase):
    """Same exactness bar with the host tier in the loop, over interleaved
    branches so hits land at many prefix lengths rather than one aligned one.

    Guards #29792 (decode track save picking its slot from the producer-side
    pointer) on both architectures. The signal is sparse rather than uniform:
    3 of 9 samples dirty and the rest exactly 0. Reverting #34184 also lands here
    on SM90, so a red run means "state reuse broke", not "#29792 broke".

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


class TestUnifiedHybridMTPBitExact(CustomTestCase):
    """Same exactness bar with MTP driving the decode loop.

    Speculative decoding advances several tokens per forward, so a track
    boundary can be crossed inside a verify step and the checkpoint is written
    from the verify path. #29792 was a wrong-slot pick in the non-spec save;
    nothing exercises the spec-side save today.

    `--speculative-num-steps 2` is load-bearing rather than incidental: it
    matches the two MTP heads this checkpoint ships, and a third step has no
    weights and the draft head refuses to start.

    Reaching the exact bar also needs the sheared relative-bias path out of the
    way, since its bias tile geometry follows the query count. Deterministic
    mode now selects the invariant path on its own, so this class carries no
    environment override; a regression there surfaces here as a nonzero KL.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = _base_args() + [
            "--speculative-algorithm",
            "EAGLE",
            "--enable-multi-layer-eagle",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
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
            env={
                **os.environ,
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            },
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

    def test_decode_cache_hit(self):
        self._run(assert_decode_cache_hit)


if __name__ == "__main__":
    unittest.main()
