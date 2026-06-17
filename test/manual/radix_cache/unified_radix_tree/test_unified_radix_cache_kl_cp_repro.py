import os
import random
import unittest

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    get_input_ids,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

QWEN3_30B_MOE_MODEL = "Qwen/Qwen3-30B-A3B-FP8"
QWEN3_32B_DENSE_MODEL = "Qwen/Qwen3-32B"

LOADBACK_CAP = 14000
NO_LOADBACK_CAP = 60000
SUFFIX_SEED_T2 = 100
SUFFIX_SEED_T3 = 200

_SUMMARY: list[dict] = []


def _random_suffixes(n, length, seed):
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


def _build_turn_suffixes(n, num_turns):
    out = []
    for t in range(num_turns - 1):
        length = 512 if t == 0 else 256
        seed = (SUFFIX_SEED_T2 if t == 0 else SUFFIX_SEED_T3) + t
        out.append(_random_suffixes(n, length, seed=seed))
    return out


def _measure_per_sample_kl(
    base_url, model, ids, max_new_tokens, temperature, num_turns
):
    n = len(ids)
    turn_suffixes = _build_turn_suffixes(n, num_turns)

    _flush_cache(base_url)
    current = [list(x) for x in ids]
    last_outputs = None
    results = None
    for turn in range(num_turns):
        if turn > 0:
            suff = turn_suffixes[turn - 1]
            current = [current[i] + last_outputs[i] + suff[i] for i in range(n)]
        results = _generate(
            base_url,
            current,
            max_new_tokens,
            return_logprob=True,
            temperature=temperature,
        )
        last_outputs = [r["output_ids"] for r in results]

    replay_ids = [current[i] + results[i]["output_ids"] for i in range(n)]
    output_lps = [_extract_output_logprobs(r) for r in results]
    input_lps = []
    for i in range(n):
        input_lps.extend(
            _get_input_logprobs(
                base_url,
                replay_ids[i : i + 1],
                output_lps[i : i + 1],
                temperature=temperature,
            )
        )

    kls = []
    for inp, out in zip(input_lps, output_lps):
        logr = np.array(inp) - np.array(out)
        kls.append(float(np.mean((np.exp(logr) - 1) - logr)))
    return kls


def _print_summary_table():
    if not _SUMMARY:
        return
    print("\n" + "=" * 124)
    print("HiCache+CP KL repro - per-sample tail (so far)")
    print("=" * 124)
    print(
        f"{'config':<30} {'cap':>6} {'T':>4} {'reps':>4} {'mean_smpl':>9} "
        f"{'p99_smpl':>9} {'max_smpl':>9} {'max_avg':>9} {'avg_over':>9} "
        f"{'smpl_over':>10} {'verdict':>8}"
    )
    print("-" * 124)
    for row in _SUMMARY:
        verdict = "FAIL" if row["max_avg"] >= row["thr"] else "ok"
        print(
            f"{row['name']:<30} {str(row['cap']):>6} {row['temp']:>4} {row['reps']:>4} "
            f"{row['mean_smpl']:>9.5f} {row['p99_smpl']:>9.5f} {row['max_smpl']:>9.5f} "
            f"{row['max_avg']:>9.5f} {row['avg_over']:>4}/{row['reps']:<4} "
            f"{row['smpl_over']:>4}/{row['n_smpl']:<5} {verdict:>8}"
        )
    print("=" * 124 + "\n")


class _ReproBase(CustomTestCase):
    __test__ = False

    model: str = QWEN3_32B_DENSE_MODEL
    is_moe: bool = False
    enable_cp: bool = True
    max_total_tokens: int | None = LOADBACK_CAP
    temperature: float = 1.0

    hicache_io_backend = "direct"
    hicache_mem_layout = "page_first_direct"
    hicache_ratio = "2"
    max_running_requests = 32
    kl_threshold = 0.005
    max_new_tokens = 512

    @classmethod
    def _build_args(cls):
        args = [
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            os.environ.get("KL_MEM_FRACTION", "0.8"),
            "--cuda-graph-max-bs",
            "32",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--disable-piecewise-cuda-graph",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            cls.hicache_ratio,
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            cls.hicache_io_backend,
            "--hicache-mem-layout",
            cls.hicache_mem_layout,
        ]
        if cls.is_moe:
            args += ["--moe-dp-size", "1", "--ep-size", "4"]
        if cls.enable_cp:
            args += ["--attn-cp-size", "2", "--enable-prefill-context-parallel"]
        if cls.max_total_tokens is not None:
            args += ["--max-total-tokens", str(cls.max_total_tokens)]
        return args

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._build_args(),
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_kl_tail(self):
        repeats = int(os.environ.get("KL_REPEATS", "12"))
        n = int(os.environ.get("KL_NUM_SAMPLES", "4"))
        num_turns = int(os.environ.get("KL_NUM_TURNS", "3"))
        total = len(self.input_ids)
        name = self.__class__.__name__

        per_repeat_avg = []
        all_sample_kls = []
        for r in range(repeats):
            start = (r * n) % total
            ids = [self.input_ids[(start + j) % total] for j in range(n)]
            kls = _measure_per_sample_kl(
                self.base_url,
                self.model,
                ids,
                self.max_new_tokens,
                self.temperature,
                num_turns,
            )
            avg = float(np.mean(kls))
            per_repeat_avg.append(avg)
            all_sample_kls.extend(kls)
            flag = "OVER" if avg >= self.kl_threshold else "ok"
            print(
                f"[{name}] repeat {r + 1}/{repeats} avg_kl={avg:.6f} [{flag}] "
                f"per_sample={[round(x, 5) for x in kls]}"
            )

        arr = np.array(per_repeat_avg)
        flat = np.array(all_sample_kls)
        row = {
            "name": name,
            "cap": self.max_total_tokens if self.max_total_tokens else "-",
            "temp": self.temperature,
            "reps": repeats,
            "mean_smpl": float(flat.mean()),
            "p99_smpl": float(np.percentile(flat, 99)),
            "max_smpl": float(flat.max()),
            "max_avg": float(arr.max()),
            "avg_over": int((arr >= self.kl_threshold).sum()),
            "smpl_over": int((flat >= self.kl_threshold).sum()),
            "n_smpl": int(flat.size),
            "thr": self.kl_threshold,
        }
        _SUMMARY.append(row)
        print(
            f"[{name}] SUMMARY mean_smpl={row['mean_smpl']:.6f} "
            f"p99_smpl={row['p99_smpl']:.6f} max_smpl={row['max_smpl']:.6f} "
            f"max_avg={row['max_avg']:.6f} avg_over={row['avg_over']}/{repeats} "
            f"sample_over={row['smpl_over']}/{flat.size}"
        )
        _print_summary_table()


class TestDenseCpLoadback(_ReproBase):
    __test__ = True
    model = QWEN3_32B_DENSE_MODEL
    is_moe = False
    enable_cp = True
    max_total_tokens = LOADBACK_CAP
    temperature = 1.0


class TestDenseCpNoLoadback(_ReproBase):
    __test__ = True
    model = QWEN3_32B_DENSE_MODEL
    is_moe = False
    enable_cp = True
    max_total_tokens = NO_LOADBACK_CAP
    temperature = 1.0


class TestDenseTpLoadback(_ReproBase):
    __test__ = True
    model = QWEN3_32B_DENSE_MODEL
    is_moe = False
    enable_cp = False
    max_total_tokens = LOADBACK_CAP
    temperature = 1.0


class TestDenseCpLoadbackGreedy(_ReproBase):
    __test__ = True
    model = QWEN3_32B_DENSE_MODEL
    is_moe = False
    enable_cp = True
    max_total_tokens = LOADBACK_CAP
    temperature = 0.0


class TestDenseCpNoLoadbackGreedy(_ReproBase):
    __test__ = True
    model = QWEN3_32B_DENSE_MODEL
    is_moe = False
    enable_cp = True
    max_total_tokens = NO_LOADBACK_CAP
    temperature = 0.0


class TestMoeCpLoadback(_ReproBase):
    __test__ = True
    model = QWEN3_30B_MOE_MODEL
    is_moe = True
    enable_cp = True
    max_total_tokens = LOADBACK_CAP
    temperature = 1.0


class TestMoeCpNoLoadback(_ReproBase):
    __test__ = True
    model = QWEN3_30B_MOE_MODEL
    is_moe = True
    enable_cp = True
    max_total_tokens = NO_LOADBACK_CAP
    temperature = 1.0


class TestMoeTpLoadback(_ReproBase):
    __test__ = True
    model = QWEN3_30B_MOE_MODEL
    is_moe = True
    enable_cp = False
    max_total_tokens = LOADBACK_CAP
    temperature = 1.0


class TestMoeCpLoadbackGreedy(_ReproBase):
    __test__ = True
    model = QWEN3_30B_MOE_MODEL
    is_moe = True
    enable_cp = True
    max_total_tokens = LOADBACK_CAP
    temperature = 0.0


class TestMoeCpNoLoadbackGreedy(_ReproBase):
    __test__ = True
    model = QWEN3_30B_MOE_MODEL
    is_moe = True
    enable_cp = True
    max_total_tokens = NO_LOADBACK_CAP
    temperature = 0.0


if __name__ == "__main__":
    unittest.main()
