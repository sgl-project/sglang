"""Small-model CUDA smoke test; run directly with PYTHONPATH=python.

Diagnostic hooks synchronize tensors to prove actual mixed batches and completed
KV checks. This is a correctness test, not a performance benchmark. Dense
STANDALONE coverage does not establish EAGLE3, HiSparse, or distributed behavior.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


def install_probes():
    # Executed at module scope in the server bootstrap, including spawned ranks.
    import torch

    from sglang.srt.managers.scheduler_components.invariant_checker import (
        SchedulerInvariantChecker,
    )
    from sglang.srt.speculative import eagle_utils, eagle_worker_common, eagle_worker_v2

    def record(kind, **values):
        with open(os.environ["OPT_OUT_TRACE"], "a") as f:
            f.write(json.dumps(dict(kind=kind, **values)) + "\n")

    sample = eagle_utils.eagle_sample

    def sample_probe(verify, batch, logits, *args, **kwargs):
        result = sample(verify, batch, logits, *args, **kwargs)
        info = batch.sampling_info
        if getattr(info, "has_spec_opt_out", False):
            assert not torch.cuda.is_current_stream_capturing()
            flags = info.spec_opt_out_mask.tolist()
            lengths = result[1].tolist()
            counts = (result[2] >= 0).sum(1).tolist()
            assert all(n == c for n, c in zip(lengths, counts))
            assert all(n == 1 for flag, n in zip(flags, lengths) if flag)
            record("verify", flags=flags, lengths=lengths, seq=batch.seq_lens.tolist())
        return result

    eagle_utils.eagle_sample = eagle_worker_common.eagle_sample = sample_probe
    extend = eagle_worker_v2.EagleDraftWorker._draft_extend_for_decode

    def extend_probe(self, batch, result, *args, **kwargs):
        out = extend(self, batch, result, *args, **kwargs)
        if getattr(batch.sampling_info, "has_spec_opt_out", False):
            torch.cuda.synchronize()
            record(
                "draft_extend",
                lengths=result.accept_lens.tolist(),
                verify_cuda_graph=bool(result.can_run_cuda_graph),
            )
        return out

    eagle_worker_v2.EagleDraftWorker._draft_extend_for_decode = extend_probe
    check = SchedulerInvariantChecker._check_kv_page_invariants

    def page_probe(self):
        check(self)
        record("kv_page_check", page_size=self.page_size)

    SchedulerInvariantChecker._check_kv_page_invariants = page_probe
    idle = SchedulerInvariantChecker._check_all_pools

    def idle_probe(self, *args, **kwargs):
        result = idle(self, *args, **kwargs)
        assert not result[0], result[1]
        record("idle_pool_check")
        return result

    SchedulerInvariantChecker._check_all_pools = idle_probe


PROMPTS = [
    "Continue the sequence with numbers separated by commas: 1, 2, 3, 4,",
    "Explain why the sky is blue in simple terms. The explanation is",
    "Write a Python function to compute Fibonacci numbers.\n```python\n",
]


def run():
    root = Path(os.environ.get("OPT_OUT_RESULTS", "/tmp/opt-out-e2e"))
    root.mkdir(parents=True, exist_ok=True)
    bootstrap = root / "serve_probe.py"
    bootstrap.write_text(
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_per_request_spec_opt_out_e2e import install_probes\n"
        "install_probes()\n"
        "if __name__ == '__main__':\n"
        "    from sglang.launch_server import run_server\n"
        "    from sglang.srt.server_args import prepare_server_args\n"
        "    run_server(prepare_server_args(sys.argv[1:]))\n"
    )
    url = "http://127.0.0.1:30000"
    baseline = None
    summary = {}
    for mode in ("target", "eager", "graph"):
        trace = root / f"{mode}.jsonl"
        trace.write_text("")
        env = dict(os.environ)
        env.update(
            OPT_OUT_TRACE=str(trace),
            SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY="1",
            SGLANG_CHECK_KV_PAGE_INVARIANTS="1",
            SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="1",
            SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH="1",
        )
        args = [
            sys.executable,
            str(bootstrap),
            "--model-path",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--attention-backend",
            "triton",
            "--mem-fraction-static",
            "0.65",
            "--max-total-tokens",
            "4096",
            "--context-length",
            "1024",
            "--page-size",
            "16",
            "--max-running-requests",
            "8",
            "--disable-prefill-cuda-graph",
        ]
        if mode != "target":
            args += [
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ]
        args += (
            ["--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "4"]
            if mode == "graph"
            else ["--disable-decode-cuda-graph"]
        )
        log = open(root / f"{mode}.log", "w")
        process = subprocess.Popen(
            args, env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                assert process.poll() is None, f"{mode} server exited; see {log.name}"
                try:
                    if requests.get(url + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"{mode} server launch timed out")

            def generate(flags, limits=(96, 96, 96)):
                response = requests.post(
                    url + "/generate",
                    json={
                        "text": PROMPTS,
                        "sampling_params": [
                            dict(
                                temperature=0,
                                max_new_tokens=n,
                                ignore_eos=True,
                                disable_speculative_decoding=flag,
                            )
                            for flag, n in zip(flags, limits)
                        ],
                        "return_logprob": True,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                rows = response.json()
                assert len(rows) == 3, rows
                tokens = []
                for row, flag, n in zip(rows, flags, limits):
                    meta = row["meta_info"]
                    ids = [item[1] for item in meta["output_token_logprobs"]]
                    assert len(ids) == n, (len(ids), n, row)
                    if flag and mode != "target":
                        assert meta["spec_num_correct_drafts"] == 0, meta
                        assert meta["spec_verify_ct"] > 0, meta
                    tokens.append(ids)
                return tokens

            if mode == "target":
                baseline = generate([False] * 3)
            else:
                # Reuse prefixes and freed rows, then flush and reallocate. The
                # asymmetric wave also filters finished rows during live decode.
                for flags, limits in [
                    ([False, True, False], (96, 96, 96)),
                    ([True] * 3, (96, 96, 96)),
                    ([True, False, True], (17, 96, 65)),
                    ([False, True, False], (96, 96, 96)),
                ]:
                    actual = generate(flags, limits)
                    for i, flag in enumerate(flags):
                        if flag:
                            assert actual[i] == baseline[i][: limits[i]], (mode, i)
                response = requests.get(url + "/flush_cache", timeout=30)
                response.raise_for_status()
                assert generate([True] * 3) == baseline
            # Probe idle checks after all requests have released their rows.
            time.sleep(2)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            counts = {
                k: sum(r["kind"] == k for r in rows)
                for k in ("verify", "draft_extend", "kv_page_check", "idle_pool_check")
            }
            assert counts["kv_page_check"] > 0 and counts["idle_pool_check"] > 0, counts
            if mode != "target":
                verifies = [r for r in rows if r["kind"] == "verify"]
                assert any(any(r["flags"]) and not all(r["flags"]) for r in verifies)
                assert any(all(r["flags"]) for r in verifies)
                assert any(any(n > 1 for n in r["lengths"]) for r in verifies)
                assert counts["draft_extend"] > 0, counts
                assert any(max(r["seq"]) > 64 for r in verifies)
                graph_rounds = sum(r.get("verify_cuda_graph", False) for r in rows)
                assert (graph_rounds > 0) == (mode == "graph"), graph_rounds
                counts["verify_cuda_graph_rounds"] = graph_rounds
            summary[mode] = counts
            print(mode, counts, flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("MODEL_E2E_AND_KV_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    run()
