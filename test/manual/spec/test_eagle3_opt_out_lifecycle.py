"""Bounded EAGLE3 lifecycle diagnostic, run directly on CUDA.

Checks actual mixed acceptance, draft committed-prefix preservation, and slot
reuse after streaming/EOS/abort. Synchronizing probes are not benchmarks.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from test_per_request_spec_opt_out_e2e import install_probes as base_probes
from test_per_request_spec_opt_out_e2e import response_token_ids

TARGET = "Qwen/Qwen3-1.7B"
DRAFT = "AngelSlim/Qwen3-1.7B_eagle3"


def check_stream(previous, current):
    assert current[: len(previous)] == previous, "stream revised emitted tokens"
    assert len(current) >= len(previous), "stream shortened"


def install_probes():
    import torch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative import eagle_worker_v2

    base_probes()

    def record(kind, **kw):
        with open(os.environ["OPT_OUT_TRACE"], "a") as f:
            f.write(json.dumps(dict(kind=kind, **kw)) + "\n")

    release = ReqToTokenPool.free

    def free(self, req):
        row = req.kv.req_pool_idx
        locs = self.req_to_token[row, : req.kv.kv_allocated_len].tolist()
        record(
            "release", rid=req.rid, row=row, locs=locs, finish=str(req.finished_reason)
        )
        return release(self, req)

    ReqToTokenPool.free = free

    original = eagle_worker_v2.EagleDraftWorker._draft_extend_for_decode
    retained = {}

    def extend(self, batch, result, **kw):
        if not batch.reqs:
            return original(self, batch, result, **kw)
        torch.cuda.synchronize()
        pool = self.draft_runner.token_to_kv_pool
        table = self.draft_runner.req_to_token_pool.req_to_token
        layers = list(range(pool.start_layer, pool.start_layer + pool.layer_num))
        assert layers, "no draft layers inspected"
        old_lens = batch.seq_lens.tolist()
        new_lens = result.new_seq_lens.tolist()
        lengths = result.accept_lens.tolist()
        rows = batch.req_pool_indices.tolist()
        flags = [r.sampling_params.disable_speculative_decoding for r in batch.reqs]
        snapshots = []
        for req, row, old, new, count in zip(
            batch.reqs, rows, old_lens, new_lens, lengths
        ):
            assert new == old + count
            locs = table[row, :old].long()
            buffers = [
                (pool.get_key_buffer(layer), pool.get_value_buffer(layer))
                for layer in layers
            ]
            values = [(k[locs].clone(), v[locs].clone()) for k, v in buffers]
            if req.rid in retained:
                prev_locs, prev_values = retained[req.rid]
                assert torch.equal(
                    locs[: len(prev_locs)], prev_locs
                ), "committed draft mapping changed"
                for (k, v), (pk, pv) in zip(values, prev_values):
                    assert torch.equal(
                        k[: len(prev_locs)], pk
                    ), "committed draft K changed"
                    assert torch.equal(
                        v[: len(prev_locs)], pv
                    ), "committed draft V changed"
            snapshots.append((req.rid, row, old, new, locs, values, buffers))
        out = original(self, batch, result, **kw)
        torch.cuda.synchronize()
        tokens = result.next_token_ids.reshape(len(rows), -1)
        for i, (rid, row, old, new, locs, values, buffers) in enumerate(snapshots):
            now_locs = table[row, :new].long().clone()
            assert len(now_locs.unique()) == new, "aliased committed draft slots"
            after = []
            for (k, v), (pk, pv) in zip(buffers, values):
                assert torch.equal(k[locs], pk), "extend overwrote draft K prefix"
                assert torch.equal(v[locs], pv), "extend overwrote draft V prefix"
                nk, nv = k[now_locs].clone(), v[now_locs].clone()
                assert torch.isfinite(nk).all() and torch.isfinite(nv).all()
                after.append((nk, nv))
            retained[rid] = (now_locs, after)
            record(
                "draft_state",
                rid=rid,
                row=row,
                old=old,
                new=new,
                accept=lengths[i],
                opt_out=flags[i],
                layers=len(layers),
                retained_ids=tokens[i, : lengths[i]].tolist(),
                locs=now_locs.tolist(),
            )
        # Keep only the current batch's snapshots to bound diagnostic memory.
        live = {r.rid for r in batch.reqs}
        for rid in list(retained):
            if rid not in live:
                del retained[rid]
        return out

    eagle_worker_v2.EagleDraftWorker._draft_extend_for_decode = extend


def run():
    from transformers import AutoTokenizer

    root = Path(os.environ.get("LIFECYCLE_RESULTS", "/workspace/eagle3-lifecycle"))
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET)
    prompts = [
        tok.apply_chat_template(
            [dict(role="user", content=p)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in (
            "Count from 1 to 1000, separated by commas. Output only numbers.",
            "Write a Python function that returns Fibonacci numbers.",
            "Reply with exactly the word hello.",
        )
    ]
    bootstrap = root / "serve_lifecycle.py"
    bootstrap.write_text(
        f"import sys; sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})\nfrom test_eagle3_opt_out_lifecycle import install_probes\ninstall_probes()\nif __name__ == '__main__':\n from sglang.launch_server import run_server\n from sglang.srt.server_args import prepare_server_args\n run_server(prepare_server_args(sys.argv[1:]))\n"
    )
    trace = root / "trace.jsonl"
    trace.write_text("")
    env = dict(
        os.environ,
        OPT_OUT_TRACE=str(trace),
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY="1",
        SGLANG_CHECK_KV_PAGE_INVARIANTS="1",
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="1",
    )
    args = [
        sys.executable,
        str(bootstrap),
        "--model-path",
        TARGET,
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DRAFT,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--attention-backend",
        "triton",
        "--page-size",
        "16",
        "--mem-fraction-static",
        "0.65",
        "--max-total-tokens",
        "4096",
        "--context-length",
        "2048",
        "--max-running-requests",
        "8",
        "--random-seed",
        "0",
        "--disable-radix-cache",
        "--disable-prefill-cuda-graph",
        "--disable-decode-cuda-graph",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
    ]
    (root / "args.json").write_text(json.dumps(args))
    url = "http://127.0.0.1:30000"
    summary = {}

    def save(name, data):
        (root / (name + ".json")).write_text(json.dumps(data, indent=2))

    def post(name, body):
        r = requests.post(url + "/generate", json=body, timeout=180)
        save(name, dict(request=body, status=r.status_code, response=r.text))
        assert r.ok, r.text
        return r.json()

    def body(rid, prompt, flag=True, n=64, **params):
        return dict(
            rid=rid,
            text=prompt,
            return_logprob=True,
            sampling_params=dict(
                temperature=0,
                max_new_tokens=n,
                disable_speculative_decoding=flag,
                **params,
            ),
        )

    def flush():
        for _ in range(100):
            r = requests.get(url + "/flush_cache", timeout=5)
            if r.ok:
                return
            time.sleep(0.1)
        raise AssertionError("server did not drain")

    def stream(name, payload, abort_after=None, disconnect=False):
        payload = dict(payload, stream=True)
        events = []
        previous = []
        aborted = False
        with requests.post(
            url + "/generate", json=payload, stream=True, timeout=180
        ) as r:
            assert r.ok, r.text
            for line in r.iter_lines(chunk_size=1):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                row = json.loads(data)
                events.append(row)
                ids = row.get("output_ids", [])
                check_stream(previous, ids)
                previous = ids
                if abort_after and len(ids) >= abort_after and not aborted:
                    if disconnect:
                        break
                    a = requests.post(
                        url + "/abort_request",
                        json=dict(rid=payload["rid"]),
                        timeout=10,
                    )
                    assert a.ok
                    aborted = True
        save(
            name,
            dict(
                request=payload,
                events=events,
                abort_sent=aborted,
                disconnect=disconnect,
            ),
        )
        assert events and previous
        return events[-1]

    with open(root / "server.log", "w") as log:
        proc = subprocess.Popen(
            args, env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            for _ in range(420):
                assert proc.poll() is None, "server exited"
                try:
                    if requests.get(url + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("server startup")
            isolated = post(
                "isolated", body("isolated", prompts[0], n=80, ignore_eos=True)
            )
            baseline = response_token_ids(isolated, 80)
            flush()
            for wave, flags, limits in [
                ("mixed", [True, False], [80, 40]),
                ("all-out", [True, True], [64, 32]),
                ("mixed-again", [False, True], [40, 80]),
            ]:
                b = dict(
                    rid=[wave + "-0", wave + "-1"],
                    text=prompts[:2],
                    return_logprob=True,
                    sampling_params=[
                        dict(
                            temperature=0,
                            max_new_tokens=n,
                            ignore_eos=True,
                            disable_speculative_decoding=f,
                        )
                        for f, n in zip(flags, limits)
                    ],
                )
                result = post(wave, b)
                ids = [response_token_ids(r, n) for r, n in zip(result, limits)]
                summary[wave] = dict(
                    lengths=list(map(len, ids)),
                    isolated_match=ids[0] == baseline if wave == "mixed" else None,
                )
                flush()
            s = stream(
                "stream-normal",
                body("stream-normal", prompts[0], n=80, ignore_eos=True),
            )
            assert (
                response_token_ids(s, 80) == baseline
            ), "stream output differs from isolated"
            assert s["meta_info"]["finish_reason"]["type"] == "length"
            flush()
            eos = stream("eos", body("eos", prompts[2], n=96))
            assert eos["meta_info"]["finish_reason"]["type"] == "stop", eos
            assert eos["meta_info"]["finish_reason"].get("matched") in (
                [tok.eos_token_id]
                if isinstance(tok.eos_token_id, int)
                else tok.eos_token_id
            ), eos
            flush()
            aborted = stream(
                "abort",
                body("abort", prompts[0], n=1500, ignore_eos=True),
                abort_after=8,
            )
            assert aborted["meta_info"]["finish_reason"]["type"] == "abort", aborted
            flush()
            stream(
                "disconnect",
                body("disconnect", prompts[0], n=1500, ignore_eos=True),
                abort_after=8,
                disconnect=True,
            )
            flush()
            for case in ("post-release-0", "post-release-1"):
                r = post(case, body(case, prompts[0], n=80, ignore_eos=True))
                assert (
                    response_token_ids(r, 80) == baseline
                ), "slot reuse changed output"
                flush()
            rows = [json.loads(x) for x in trace.read_text().splitlines()]
            ver = [x for x in rows if x["kind"] == "verify"]
            assert any(
                any(x["flags"]) and not all(x["flags"]) for x in ver
            ), "no mixed verify"
            assert any(
                any(n > 1 for f, n in zip(x["flags"], x["lengths"]) if not f)
                for x in ver
            ), "no enabled acceptance"
            states = [x for x in rows if x["kind"] == "draft_state"]
            assert states
            releases = [x for x in rows if x["kind"] == "release"]
            reuse = {}
            for rid in ("stream-normal", "eos", "abort", "disconnect"):
                freed = next(x for x in releases if x["rid"] == rid)
                later = [
                    x
                    for x in rows[rows.index(freed) + 1 :]
                    if x["kind"] == "draft_state" and x["rid"] != rid
                ]
                reuse[rid] = any(
                    x["row"] == freed["row"] and set(x["locs"]) & set(freed["locs"])
                    for x in later
                )
                assert reuse[rid], f"no observed row+KV slot reuse after {rid}"
            summary.update(
                mixed_rounds=sum(any(x["flags"]) and not all(x["flags"]) for x in ver),
                draft_checks=len(states),
                reuse=reuse,
                eos=eos["meta_info"]["finish_reason"],
                abort=aborted["meta_info"]["finish_reason"],
            )
            save("summary", summary)
            print("EAGLE3_LIFECYCLE_PASS", json.dumps(summary), flush=True)
        finally:
            save("partial-summary", summary)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    run()
