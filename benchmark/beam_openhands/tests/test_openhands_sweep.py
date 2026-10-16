"""Unit tests for the OpenHands sweep helpers; no server or network.

Run from benchmark/beam_openhands: python -m unittest discover -s tests -v
(the builder and plugin cases need the client venv; they are skipped elsewhere).
"""

import asyncio
import base64
import hashlib
import itertools
import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import summarize_rung  # noqa: E402
import sweep_tool  # noqa: E402

try:
    import build_openhands_padded_dataset as builder
    import numpy as np
except ImportError:  # outside the client venv
    builder = None
try:
    import evalscope_client
    from evalscope.perf.utils.benchmark_util import BenchmarkData
except ImportError:
    evalscope_client = None

TURNS, OSL = 3, 5
SPECIAL = ["<|start|>", "<|channel|>", "<|message|>", "<|end|>"]


def _tokenizer_dir(root):
    """A tokenizer.json with four special added tokens and one non-special one."""
    path = os.path.join(root, "tok")
    os.makedirs(path, exist_ok=True)
    added = [{"content": t, "special": True} for t in SPECIAL] + [
        {"content": "<think>", "special": False}
    ]
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump({"added_tokens": added}, f)
    return path


def _chunks(content, reasoning, cached=None, record=True):
    chunks = [
        {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        },
        {"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}}]},
        {"choices": [{"index": 0, "delta": {"content": content}}]},
    ]
    if record:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": OSL,
            "cached_tokens": cached,
            "special_token_chars_removed": 0,
        }
        chunks.append({summarize_rung.RECORD_KEY: usage})
    return chunks


def _write_db(
    path,
    conversations,
    osl=OSL,
    drop=None,
    raw_history_at=None,
    cached=None,
    record=True,
    empty_at=None,
):
    """An evalscope-shaped result table for conversations [(system, first_turn_tokens)], turns in order.

    Every generated turn streams a special-token spelling; the replayed history strips it, as the plugin does,
    except at raw_history_at=(conversation, turn), which replays the raw stream. empty_at=(conversation, turn)
    streams no text at all.
    """
    con = sqlite3.connect(path)
    with con:
        con.execute(
            "CREATE TABLE result(request TEXT, start_time REAL, inter_token_latencies TEXT, success INTEGER, "
            "response_messages TEXT, completed_time REAL, latency REAL, first_chunk_latency REAL, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, max_gpu_memory_cost REAL, time_per_output_token REAL)"
        )
        for ci, (system, first_tokens) in enumerate(conversations):
            messages, prompt = [{"role": "system", "content": system}], first_tokens
            for t in range(1, TURNS + 1):
                messages = messages + [{"role": "user", "content": f"u{ci}-{t}"}]
                content, reasoning = f"answer {ci}-{t}<|start|>tail", f"think {ci}-{t} "
                if (ci, t) == empty_at:
                    content, reasoning = "", ""
                if (ci, t) != drop:
                    con.execute(
                        "INSERT INTO result(request, success, response_messages, prompt_tokens, completion_tokens) "
                        "VALUES (?, 1, ?, ?, ?)",
                        (
                            json.dumps({"messages": messages}),
                            base64.b64encode(
                                pickle.dumps(
                                    _chunks(content, reasoning, cached, record)
                                )
                            ).decode(),
                            prompt,
                            osl,
                        ),
                    )
                # The chunks above stream reasoning first, then content.
                streamed = reasoning + content
                history = (
                    streamed
                    if (ci, t) == raw_history_at
                    else streamed.replace("<|start|>", "")
                )
                messages = messages + [{"role": "assistant", "content": history}]
                prompt += 10
    con.close()


def _stats(conversations):
    return [
        {
            "system_sha256": hashlib.sha256(s.encode()).hexdigest(),
            "first_turn_prompt_tokens": n,
        }
        for s, n in conversations
    ]


class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "benchmark_data.db")
        self.special = sweep_tool.special_token_pattern(_tokenizer_dir(self.tmp.name))
        self.convs = [("sys-a", 100), ("sys-b", 101), ("sys-c", 102), ("sys-d", 103)]

    def tearDown(self):
        self.tmp.cleanup()

    def _validate(self, ran, offset, number, **db_kwargs):
        if os.path.exists(self.db):
            os.remove(self.db)
        _write_db(self.db, ran, **db_kwargs)
        return summarize_rung.validate(
            records=summarize_rung.read_requests(self.db),
            stats=_stats(self.convs),
            offset=offset,
            number=number,
            turns=TURNS,
            osl=OSL,
            context_length=10_000,
            special=self.special,
        )

    def test_complete_run_passes(self):
        failures, details = self._validate(self.convs[:2], offset=0, number=2)
        self.assertEqual(failures, [])
        self.assertEqual(details["turn1_prompt_tokens_checked"], 2)
        self.assertEqual(details["responses_with_special_token_text"], 2 * TURNS)
        self.assertEqual(details["output_mix"]["content_and_reasoning"], 2 * TURNS)

    def test_warmup_window_is_the_rotated_tail(self):
        # evalscope rotates by --dataset-offset: offset n-2, number 2 is the last two conversations.
        self.assertEqual(self._validate(self.convs[2:], offset=2, number=2)[0], [])
        failures, _ = self._validate(self.convs[:2], offset=2, number=2)
        self.assertTrue(
            any("not in the expected dataset window" in f for f in failures), failures
        )

    def test_short_output_is_not_a_pass(self):
        failures, _ = self._validate(self.convs[:2], offset=0, number=2, osl=OSL - 1)
        self.assertTrue(
            any(f"did not produce {OSL} completion tokens" in f for f in failures),
            failures,
        )

    def test_missing_turn_is_not_a_pass(self):
        failures, _ = self._validate(self.convs[:2], offset=0, number=2, drop=(1, 2))
        self.assertTrue(any("requests recorded" in f for f in failures), failures)
        self.assertTrue(any("exactly once" in f for f in failures), failures)

    def test_history_must_be_the_normalized_streamed_text(self):
        failures, _ = self._validate(
            self.convs[:2], offset=0, number=2, raw_history_at=(0, 1)
        )
        self.assertTrue(
            any("streamed text as history" in f for f in failures), failures
        )

    def test_an_empty_stream_is_counted_not_failed(self):
        # The server sometimes streams no text for a 220-token turn; evalscope replays an empty message.
        failures, details = self._validate(
            self.convs[:2], offset=0, number=2, empty_at=(1, 2)
        )
        self.assertEqual(failures, [])
        self.assertEqual(details["empty_replayed_history_turns"], 1)
        self.assertEqual(details["output_mix"]["empty_text"], 1)

    def test_requests_outside_the_replay_plugin_are_refused(self):
        failures, _ = self._validate(self.convs[:2], offset=0, number=2, record=False)
        self.assertTrue(any("usage record" in f for f in failures), failures)

    def test_turn1_token_mismatch_flags_tokenizer_or_kwargs_drift(self):
        failures, _ = self._validate(
            [("sys-a", 100), ("sys-b", 99)], offset=0, number=2
        )
        self.assertTrue(
            any("turn-1 prompt tokens differ" in f for f in failures), failures
        )

    def test_cache_hit_needs_server_cached_tokens_on_every_request(self):
        _, details = self._validate(self.convs[:2], offset=0, number=2, cached=None)
        self.assertIsNone(details["server_cache"]["hit_pct"])
        _, details = self._validate(self.convs[:2], offset=0, number=2, cached=90)
        self.assertEqual(details["server_cache"]["hit_pct"], 90.0)

    def test_stream_decoding_matches_evalscope_concatenation(self):
        # evalscope default_api: per chunk, delta.content + delta.reasoning_content, in stream order.
        blob = base64.b64encode(pickle.dumps(_chunks("final", "reason "))).decode()
        streamed, content, reasoning, record = summarize_rung.decode_stream(blob)
        self.assertEqual(
            (streamed, content, reasoning), ("reason final", "final", "reason ")
        )
        self.assertEqual(record["completion_tokens"], OSL)

    def test_special_token_pattern_leaves_non_special_added_tokens(self):
        self.assertEqual(
            self.special.sub("", "<|start|>a<|channel|><think>b"), "a<think>b"
        )


class RowTest(unittest.TestCase):
    SUMMARY = {
        "Total Requests": 6,
        "Failed Requests": 0,
        "Output Throughput (tok/s)": 800.0,
        "Total Throughput (tok/s)": 80000.0,
        "KV Cache Hit Rate (%)": 91.9,
    }
    PCT = [
        {"Percentiles": "50%", "TPOT (ms)": 20.0, "TTFT (ms)": 300.0},
        {"Percentiles": "90%", "TPOT (ms)": 30.0, "TTFT (ms)": 600.0},
        {"Percentiles": "99%", "TPOT (ms)": 40.0, "TTFT (ms)": 900.0},
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rung = os.path.join(self.tmp.name, "c1")
        self.run_dir = os.path.join(self.rung, "ladder", "c1", "parallel_1_number_2")
        os.makedirs(self.run_dir)
        with open(os.path.join(self.rung, "validation.json"), "w") as f:
            json.dump({"ok": True, "server_cache": {"hit_pct": None}}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, status="OK", n_gpus=8):
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "summarize_rung.py"),
                "row",
                "--rung-dir",
                self.rung,
                "--cc",
                "1",
                "--status",
                status,
                "--expected-requests",
                "6",
                "--n-gpus",
                str(n_gpus),
                "--out",
                os.path.join(self.rung, "rung.json"),
            ],
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout.strip()

    def _write(self, name, obj):
        with open(os.path.join(self.run_dir, name), "w") as f:
            json.dump(obj, f)

    def test_ok_needs_the_summary_artifacts(self):
        self.assertEqual(self._row(), (1, "SUMMARY_FAIL"))
        self._write("benchmark_summary.json", self.SUMMARY)
        self.assertEqual(self._row(), (1, "SUMMARY_FAIL"))  # percentiles still missing
        self._write("benchmark_percentile.json", self.PCT)
        self.assertEqual(self._row(), (0, "OK"))
        self._write(
            "benchmark_summary.json", dict(self.SUMMARY, **{"Total Requests": 5})
        )
        self.assertEqual(self._row(), (1, "SUMMARY_FAIL"))

    def test_non_positive_gpu_count_is_refused(self):
        self.assertEqual(self._row(n_gpus=0)[0], 2)

    def test_metrics_normalization_and_cache_columns(self):
        m = summarize_rung.rung_metrics(
            self.SUMMARY, self.PCT, {"server_cache": {"hit_pct": None}}, n_gpus=8
        )
        self.assertEqual(
            (m["out_tok_s_per_gpu"], m["total_tok_s_per_gpu"]), (100.0, 10000.0)
        )
        self.assertEqual(
            (m["interactivity_p50"], m["interactivity_at_tpot_p99"]), (50.0, 25.0)
        )
        self.assertEqual(m["actual_cache_hit_pct"], "unavailable")
        self.assertEqual(
            m["evalscope_kv_hit_basis"], "client estimate, not a cache measurement"
        )


class PreflightTest(unittest.TestCase):
    def test_client_pins(self):
        class Dist:
            def __init__(self, version, commit=""):
                self.version, self._commit = version, commit

            def read_text(self, name):
                return (
                    json.dumps({"vcs_info": {"commit_id": self._commit}})
                    if self._commit
                    else None
                )

        pins = {"evalscope": sweep_tool.EVALSCOPE_COMMIT, "tokenizers": "0.22.2"}
        good = {
            "evalscope": Dist("0.0.0", sweep_tool.EVALSCOPE_COMMIT),
            "tokenizers": Dist("0.22.2"),
            "pip": Dist("24"),
        }
        self.assertEqual(sweep_tool.check_client(pins, good), [])
        problems = sweep_tool.check_client(
            pins, dict(good, tokenizers=Dist("0.23.2"), extra=Dist("1"))
        )
        self.assertTrue(
            any("tokenizers" in p for p in problems)
            and any("extra" in p for p in problems),
            problems,
        )

    def test_server_profile_concurrency_and_gpu_count(self):
        profile = {"resolved_args": {"tp_size": 8, "page_size": 64}}
        info = {
            "tp_size": 8,
            "dp_size": 1,
            "page_size": 64,
            "model_path": "/m",
            "served_model_name": "s",
            "max_running_requests": 64,
        }
        self.assertEqual(
            sweep_tool.check_server(info, profile, "/m", "s", cc=64, n_gpus=8), []
        )
        problems = sweep_tool.check_server(
            dict(info, page_size=128), profile, "/m", "s", cc=128, n_gpus=4
        )
        self.assertEqual(len(problems), 3, problems)

    def test_load_counts_include_side_queues(self):
        payload = {
            "loads": [
                {
                    "num_running_reqs": 3,
                    "num_waiting_reqs": 1,
                    "queues": {"waiting": 1, "retracted": 2},
                }
            ]
        }
        self.assertEqual(sweep_tool.load_counts(payload), (3, 3))

    def test_load_sampler_stops_when_its_sweep_is_gone(self):
        """Regression: an orphaned sampler kept polling the server and held the results-directory lock."""
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = os.path.join(tmp, "pid")
            script = (
                f"{sys.executable} {os.path.join(HERE, 'sweep_tool.py')} sample-loads "
                f"--base-url http://127.0.0.1:9 --interval-s 0.1 --out {tmp}/loads.jsonl "
                f"> /dev/null 2>&1 & echo $! > {pid_file}; sleep 0.5"
            )
            subprocess.run(["bash", "-c", script], check=True)
            with open(pid_file) as f:
                pid = int(f.read())
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, 9)
                self.fail("the sampler outlived the process that started it")

    def test_sample_indices_cover_both_ends(self):
        self.assertEqual(sweep_tool.sample_indices(1024, 2), [0, 1023])
        self.assertEqual(sweep_tool.sample_indices(3, 8), [0, 1, 2])


class ResumeIdentityTest(unittest.TestCase):
    SERVER = {
        "tp_size": 8,
        "tokenizer_worker_num": 1,
        "enable_http2": False,
        "cuda_graph_config": {"decode": {"max_bs": 512}},
        "random_seed": 1,
        "internal_states": [{"last_gen_throughput": 1.0}],
        "startup_time": {"load_weight": 60.0},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(self.pkg)
        for name in sweep_tool.IDENTITY_FILES:
            with open(os.path.join(self.pkg, name), "w") as f:
                f.write(name)
        self.results = os.path.join(self.tmp.name, "results")
        os.makedirs(self.results)

    def tearDown(self):
        self.tmp.cleanup()

    def _attempt(self, name, server):
        path = os.path.join(self.tmp.name, name)
        os.makedirs(path)
        files = {
            "client-manifest.json": {
                "requirements_sha256": "r",
                "evalscope_commit": "e",
                "python": "3.12",
                "machine": "aarch64",
            },
            "tokenizer-view.json": {"revision": "t"},
            "dataset-stats.json": {
                "sha256": "d",
                "num_conversations": 4,
                "metadata_sha256": "m",
                "metadata": {"tokenizer": {"revision": "t"}},
            },
            "server-info-start.json": server,
        }
        for f, obj in files.items():
            with open(os.path.join(path, f), "w") as out:
                json.dump(obj, out)
        return path

    def _accept(self, attempt):
        return sweep_tool.accept_run(self.results, attempt, self.pkg, {"osl": "220"})[0]

    def _snapshot(self):
        out = {}
        for name in sorted(os.listdir(self.results)):
            with open(os.path.join(self.results, name), "rb") as f:
                out[name] = f.read()
        return out

    def test_stable_server_changes_refuse_resume_and_keep_accepted_files(self):
        self.assertEqual(self._accept(self._attempt("a0", self.SERVER)), 0)
        accepted = self._snapshot()
        self.assertEqual(
            sorted(accepted), sorted(sweep_tool.RUN_FILES + ("run-identity.json",))
        )
        changes = [
            {"tokenizer_worker_num": 4},
            {"enable_http2": True},
            {"cuda_graph_config": {"decode": {"max_bs": 128}}},
            {"random_seed": 2},
        ]
        for i, change in enumerate(changes):
            self.assertEqual(
                self._accept(self._attempt(f"a{i + 1}", dict(self.SERVER, **change))),
                3,
                change,
            )
            self.assertEqual(self._snapshot(), accepted, change)

    def test_live_state_and_boot_timings_do_not_block_resume(self):
        self.assertEqual(self._accept(self._attempt("a0", self.SERVER)), 0)
        accepted = self._snapshot()
        moved = dict(
            self.SERVER,
            internal_states=[{"last_gen_throughput": 9.0}],
            startup_time={"load_weight": 70.0},
        )
        self.assertEqual(self._accept(self._attempt("a1", moved)), 0)
        self.assertEqual(self._snapshot(), accepted)


class _ToyBpe:
    """Rank-ordered pair merges: "bc" beats "ab" and "cd", so "ab" + "cd" re-encodes as a, bc, d."""

    MERGES = {("b", "c"): 0, ("a", "b"): 1, ("c", "d"): 2}

    def __init__(self):
        self.vocab = {}

    def _id(self, token):
        return self.vocab.setdefault(token, len(self.vocab))

    def encode(self, text):
        toks = list(text)
        while True:
            pairs = [
                (self.MERGES.get((a, b), 99), i)
                for i, (a, b) in enumerate(itertools.pairwise(toks))
            ]
            rank, i = min(pairs, default=(99, -1))
            if rank == 99:
                return [self._id(t) for t in toks]
            toks[i : i + 2] = [toks[i] + toks[i + 1]]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def count(self, text):
        return len(self.encode(text))

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[int(i)] for i in ids)


class _Pool:
    def __init__(self, ids):
        self.ids, self.used = np.asarray(ids, dtype=np.int64), 0

    def peek(self, n):
        return (
            self.ids[self.used : self.used + n]
            if self.used + n <= len(self.ids)
            else None
        )

    def consume(self, n):
        self.used += n


@unittest.skipIf(builder is None, "needs the client venv (numpy, tqdm)")
class PadFitTest(unittest.TestCase):
    def test_exact_fit_when_the_boundary_merge_skips_the_target(self):
        """Regression: a pad whose first token merges with the prefix end jumped 752 -> 754 and the build failed."""
        codec = _ToyBpe()
        prefix = "xxab"
        pool = _Pool([codec.encode(t)[0] for t in ["cd", "e", "f", "g", "h"] * 30])
        pads = builder.PadMaker(codec=codec, sanitize=lambda s: s, pool=pool)
        target = codec.count(prefix) + 1
        text = pads.fit(prefix, target, rng=None)
        self.assertEqual(codec.count(text), target)
        self.assertEqual(pads.boundary_adjustments, 1)
        before = pool.used
        pads.fit("", 7, rng=None)
        self.assertGreater(
            pool.used, before
        )  # later pads start after every consumed token

    def test_truncate_to_never_exceeds_the_limit(self):
        codec = _ToyBpe()
        self.assertLessEqual(
            codec.count(builder.truncate_to("abcdabcdabcd", 5, codec)), 5
        )


@unittest.skipIf(evalscope_client is None, "needs the client venv (evalscope)")
class ReplayPluginTest(unittest.TestCase):
    def test_history_drops_special_tokens_and_keeps_usage(self):
        """The Beam template rejects replayed Harmony tags with HTTP 400; timing and counts must stay untouched."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        plugin = object.__new__(evalscope_client.OpenHandsReplayPlugin)
        plugin.special_tokens = sweep_tool.special_token_pattern(_tokenizer_dir(tmp))

        async def fake_process(self, *args):
            return BenchmarkData(
                success=True,
                generated_text="think<|start|>assistant<|channel|>final<|message|>hi",
                prompt_tokens=10,
                completion_tokens=OSL,
                query_latency=1.5,
                first_chunk_latency=0.5,
            )

        base = evalscope_client.OpenaiPlugin
        original, base.process_request = base.process_request, fake_process
        self.addCleanup(setattr, base, "process_request", original)
        out = asyncio.run(plugin.process_request(None, "url", {}, {}))
        self.assertEqual(out.generated_text, "thinkassistantfinalhi")
        self.assertEqual(
            (out.completion_tokens, out.query_latency, out.first_chunk_latency),
            (OSL, 1.5, 0.5),
        )
        record = out.response_messages[-1][evalscope_client.RECORD_KEY]
        self.assertEqual(
            (record["completion_tokens"], record["cached_tokens"]), (OSL, None)
        )


if __name__ == "__main__":
    unittest.main()
