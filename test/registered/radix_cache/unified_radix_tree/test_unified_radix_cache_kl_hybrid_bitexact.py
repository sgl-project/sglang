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
import re
import shutil
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _replay_and_compare_kl,
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
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_cuda_ci(est_time=2800, stage="base-b", runner_config="1-gpu-large")

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

# flush_cache waits for a fully idle scheduler, which under buffer_only means
# every staged storage write has drained. The file backend writes each page
# through its generic (non zero-copy) path at a few MB/s, so a seeding pass
# outlasts the 30s default the helpers use by a wide margin.
_FLUSH_TIMEOUT_S = 600

# The seeded prefix has to end strictly inside the measured prompt, not at its
# end: storage keys one recurrent state per checkpoint node, the hit query only
# accepts a prefix that ends on one, and a request can never ask for its own
# last token back (max_prefix_len is input_len - 1). Seeding and measuring the
# same prompt therefore reads as a total miss however much KV is stored.
#
# Sample count and generation length are sized against the file backend's write
# throughput rather than against the KL statistics: every cached token here is
# also a page it has to write before the next flush can return. The generation
# still runs past the 512-token sliding window, so decode carries the window
# through the prompt-to-generated handover.
BUFFER_ONLY_SAMPLES = 8
BUFFER_ONLY_SEED_TOKENS = 512
BUFFER_ONLY_PROMPT_TOKENS = 768
BUFFER_ONLY_MAX_NEW_TOKENS = 640


def _random_suffixes(n: int, length: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


def _base_args(
    mamba_strategy: str = "extra_buffer",
    swa_full_tokens_ratio: str = "0.1",
    *,
    mem_fraction_static: float = 0.6,
) -> list[str]:
    return [
        "--trust-remote-code",
        "--attention-backend",
        "fa4",
        "--page-size",
        str(PAGE_SIZE),
        "--mamba-radix-cache-strategy",
        mamba_strategy,
        "--swa-full-tokens-ratio",
        swa_full_tokens_ratio,
        "--mamba-full-memory-ratio",
        "0.1",
        # 0.85 was carried over from the 4-GPU B200 test and OOMs an 80 GB card:
        # the static pool leaves ~19 GB for the prefill graphs, the fa4 workspace
        # and the chunked-prefill activations, which is what this config needs.
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--mamba-track-interval",
        str(TRACK_INTERVAL),
        "--enable-deterministic-inference",
    ]


def _prefill_graph_count(base_url: str) -> float:
    metrics = requests.get(base_url + "/metrics", timeout=30).text
    matches = re.findall(
        r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"[^}]*\}'
        r"\s+([0-9.eE+-]+)$",
        metrics,
        re.MULTILINE,
    )
    return sum(map(float, matches), 0.0)


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

    tree_core_backend = "python"

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
            env=unified_radix_tree_server_env(cls.tree_core_backend),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)

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
            env=unified_radix_tree_server_env(cls.tree_core_backend),
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

    tree_core_backend = "python"

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
            env=unified_radix_tree_server_env(cls.tree_core_backend),
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=9, trust_remote_code=True
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)

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

    tree_core_backend = "python"

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Target FullCG and both MTP draft workers retain graph pools, so this
        # class needs more dynamic-memory headroom than the non-spec tests.
        other_args = _base_args(mem_fraction_static=0.58) + [
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
            "--enable-metrics",
        ]
        if _MODEL_REVISION:
            other_args += ["--revision", _MODEL_REVISION]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=unified_radix_tree_server_env(cls.tree_core_backend),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)

    def _run(self, helper):
        server_info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(server_info["cuda_graph_config"]["prefill"]["backend"], "full")
        graph_count = _prefill_graph_count(self.base_url)
        helper(
            self.base_url,
            {self.model: {"kl_div": KL_DIV_THRESHOLD}},
            self.model,
            max_samples=32,
            max_new_tokens=MAX_NEW_TOKENS,
            trust_remote_code=True,
        )
        self.assertGreater(
            _prefill_graph_count(self.base_url),
            graph_count,
            "MTP target prefill did not replay the configured Full CUDA graph.",
        )

    def test_logprobs_match(self):
        self._run(assert_logprobs_match)

    def test_decode_cache_hit(self):
        self._run(assert_decode_cache_hit)


class TestRustUnifiedHybridBitExact(TestUnifiedHybridBitExact):
    tree_core_backend = "rust"


class TestRustUnifiedHybridLazyBitExact(TestUnifiedHybridLazyBitExact):
    tree_core_backend = "rust"


class TestRustUnifiedHybridHiCacheBitExact(TestUnifiedHybridHiCacheBitExact):
    tree_core_backend = "rust"


class TestRustUnifiedHybridMTPBitExact(TestUnifiedHybridMTPBitExact):
    tree_core_backend = "rust"


class TestUnifiedHybridBufferOnlyBitExact(CustomTestCase):
    """Same exactness bar with host memory as a pure GPU<->L3 staging buffer.

    `buffer_only` retains nothing on the host tier, so after a flush the only
    thing a request can reuse is what L3 holds, spliced in at prefill
    admission. All three components have to come back off one host bounce and
    reproduce what a fresh prefill computes: the FULL KV, the trailing SWA
    window, and the recurrent Mamba state. The state is the part with no
    second chance -- it is a single slot restored into both the published tree
    node and the consuming request, and a restore into the wrong slot still
    generates fluent text.

    The flush between seeding and measurement is what makes this a storage
    test rather than a device-tree test: it drops the radix tree while leaving
    L3 intact, so every nonzero `cached_tokens` below came back through the
    buffer-mode read path.
    """

    @classmethod
    def _seed_ids(cls) -> list[list[int]]:
        return [ids[:BUFFER_ONLY_SEED_TOKENS] for ids in cls.input_ids]

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="sgl-buffer-only-kl-")
        # A span is only readable back from storage if its trailing SWA window
        # was still resident when the backup staged: the hit query folds the
        # window's boundary into the KV hit, so an SWA-evicted span reads as a
        # total miss. At the 0.1 the other classes run, the seeding set's own
        # windows evict each other and the earliest samples go cold.
        other_args = _base_args(swa_full_tokens_ratio="0.5") + [
            "--enable-hierarchical-cache",
            "--hicache-host-memory-mode",
            "buffer_only",
            "--hicache-storage-backend",
            "file",
            # Without it the backend answers every existence query with a
            # scandir over the whole storage directory, which grows with the
            # pages this class writes.
            "--hicache-storage-backend-extra-config",
            '{"enable_metadata_cache": true}',
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "direct",
            # The mamba host pool only supports page_first and page_first_direct.
            "--hicache-mem-layout",
            "page_first_direct",
            # Fetch to completion: a partial fetch is a legitimate outcome the
            # scheduler degrades on, and it would turn the cache-hit assertions
            # below into flakes rather than a signal.
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            # Same tight full-KV budget as the cache-mode class above, so
            # eviction and checkpoint rotation actually run instead of
            # everything staying resident on device.
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
            env={
                **os.environ,
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
            },
        )
        cls.input_ids = [
            ids[:BUFFER_ONLY_PROMPT_TOKENS]
            for ids in get_input_ids(
                tokenizer_path=cls.model,
                num_samples=BUFFER_ONLY_SAMPLES,
                trust_remote_code=True,
            )
        ]

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def test_prefill_cache_hit_from_storage(self):
        """Seed a prefix into L3, drop the device tree, then measure a longer
        prompt over it: the restored span and the recurrent state that ends it
        must score the generation exactly as a recomputed prefix does."""
        seed_ids = self._seed_ids()
        full_ids = self.input_ids

        # flush_cache only proceeds once the scheduler is fully idle, which in
        # buffer mode includes the in-flight storage writes, so the second
        # flush cannot race the seeding pass to L3. Those writes are what the
        # helpers' 30s default is too short for.
        _flush_cache(self.base_url, timeout_s=_FLUSH_TIMEOUT_S)
        _generate(self.base_url, seed_ids, max_new_tokens=0)
        _flush_cache(self.base_url, timeout_s=_FLUSH_TIMEOUT_S)

        results = _generate(
            self.base_url,
            full_ids,
            BUFFER_ONLY_MAX_NEW_TOKENS,
            return_logprob=True,
        )
        self.assertEqual(len(results), len(full_ids))

        cached = [r["meta_info"]["cached_tokens"] for r in results]
        print(f"buffer_only storage hits: {cached}")
        for i, hit in enumerate(cached):
            self.assertGreater(
                hit,
                0,
                f"buffer_only[{i}] took no storage hit: the device tree was "
                "flushed, so this run never exercised the buffer-mode read path",
            )
            # Storage holds one recurrent state per checkpoint node, so a hit
            # can only end where a checkpoint does, and never past the span
            # that was seeded.
            self.assertEqual(
                hit % TRACK_INTERVAL,
                0,
                f"buffer_only[{i}]: hit of {hit} tokens is off the "
                f"{TRACK_INTERVAL}-token checkpoint grid",
            )
            self.assertLessEqual(hit, BUFFER_ONLY_SEED_TOKENS)

        # Drain this pass's storage writes here: the replay's own flush uses
        # the 30s default, which they can outlast.
        _flush_cache(self.base_url, timeout_s=_FLUSH_TIMEOUT_S)
        _replay_and_compare_kl(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            [full_ids[i] + results[i]["output_ids"] for i in range(len(results))],
            [_extract_output_logprobs(r) for r in results],
            label="buffer_only_prefill_cache_hit",
            # One replay batch, one flush: the helper flushes per batch, and a
            # per-sample flush would have to drain that sample's own storage
            # writes inside its 30s default.
            batch_size=BUFFER_ONLY_SAMPLES,
            sampling_temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
