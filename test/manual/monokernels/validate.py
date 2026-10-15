# SPDX-License-Identifier: Apache-2.0
"""Validate a model's monokernels on one node and print PASS or FAIL for each step (RFC #43949 section 11).

Steps, in order (--steps picks a subset):
  cpu         the plugin and spec CPU tests (test/registered/monokernels/unit)
  kernels     each span against the normal path at every production width, TP2 and TP4, with graph replay
  guard       a stalled wait at every named site stops every rank, with the reason readable
  refuse      an unsupported deployment refuses to start, with every reason at once
  serve, ab   the cookbook command (DSpark verify steps) with the switch on, then off: the plans, the captured
              widths, GSM8K within --gsm8k-gap points, NIAH 1.0 at 64k and 128k tokens, KL-consistency with token
              agreement, and the DSpark accept length (a drop of more than 3 % fails)
  serve_decode, ab_decode
              the same command without speculative decoding (decode steps): decode at 48 rows, above and back;
              GSM8K and KL-consistency on and off
  fault       a mid-step wait stalled on one rank during a streamed request stops the server on every rank; the
              request does not finish
  fault_last  the last layer's epoch wait stalled on one rank (decode steps, one request): the server stops, the
              request does not finish, and every token sent matches the request's answer without the fault

Servers run the cookbook's agentic command (docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx, section 3.6,
concurrency 8) with its environment, the Engram host table (per_rank) included; --host-table 0 opts out for a
separate test. The gates keep at most 8 requests in flight, so every verify step stays within the monokernels'
48 rows.

Example (DeepSeek-V4.1-Flash, 4 GPUs):
  python3 test/manual/monokernels/validate.py --model /model --tp 4
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
KERNELS = os.path.join(ROOT, "test", "registered", "monokernels", "kernels")
UNIT = os.path.join(ROOT, "test", "registered", "monokernels", "unit")

# the cookbook's agentic command at concurrency 8 (its table: 16 running requests, 512 prefix tails, interval 192,
# lpm, 0.70 of memory at TP4)
COOKBOOK_ENV = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV41_REASONING_EFFORT": "high",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "0",
    "TORCH_BLAS_PREFER_HIPBLASLT": "1",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "TRITON_HIP_USE_ASYNC_COPY": "0",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
    "SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL": "true",
    "SGLANG_OPT_HIP_FLYDSL_MXFP8": "1",
    "HSA_NO_SCRATCH_RECLAIM": "0",
    "GPU_MAX_HW_QUEUES": "2",
}
SPECULATIVE = "--speculative-algorithm DSPARK --speculative-dspark-block-size 5"
COOKBOOK_ARGS = (
    f"--trust-remote-code --ep-size 1 {SPECULATIVE} --enforce-shared-experts-fusion --fp8-gemm-backend aiter "
    "--enable-decoder-swa-bounded-replay --cuda-graph-backend-prefill disabled --mem-fraction-static 0.70 "
    "--chunked-prefill-size 16384 --prefill-decode-interval 192 --schedule-policy lpm --swa-prefix-tails 512 "
    "--max-running-requests 16 --cuda-graph-max-bs-decode 64 --reasoning-parser auto --tool-call-parser auto "
    "--watchdog-timeout 3600"
)
# decode steps: the same without speculative decoding, with room for decode steps of 48 rows and above
DECODE_ARGS = COOKBOOK_ARGS.replace(f"{SPECULATIVE} ", "").replace(
    "--max-running-requests 16", "--max-running-requests 64"
)
# requests in flight for the gates: with DSpark's 6 verify rows a request, 8 requests keep every verify step within
# the 48 rows the monokernels serve (more requests send the steps to the model path)
IN_FLIGHT = 8
DEADLINE_S = 60  # half the cookbook's 3600 s watchdog, at most 60 s
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def run(cmd, timeout, env=None, cwd=None):
    t0 = time.time()
    p = subprocess.run(
        cmd, shell=True, cwd=cwd, env={**os.environ, **(env or {})}, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    return p.returncode, p.stdout, time.time() - t0


class Server:
    def __init__(self, args, mono: bool, env=None, extra="", decode=False):
        self.args, self.port = args, args.port
        recipe = DECODE_ARGS if decode else COOKBOOK_ARGS
        cmd = (
            f"{sys.executable} -m sglang.launch_server --model-path {args.model} --tp {args.tp} "
            f"--port {self.port} {recipe} {'--enable-monokernel' if mono else ''} {extra}"
        )
        name = f"server_{'on' if mono else 'off'}{'_decode' if decode else ''}_{int(time.time())}.log"
        self.log = open(os.path.join(args.out, name), "w")
        cookbook = dict(COOKBOOK_ENV)
        if not args.host_table:
            cookbook["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"] = "0"
        self.p = subprocess.Popen(
            cmd, shell=True, env={**os.environ, **cookbook, **(env or {})}, stdout=self.log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip

    def wait_ready(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                return False
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=5
                )
                return True
            except Exception:
                time.sleep(10)
        return False

    def text(self):
        self.log.flush()
        return open(self.log.name, errors="replace").read()

    def stop(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def complete(port, prompt, n, timeout=1800):
    body = {"prompt": prompt, "max_tokens": n, "temperature": 0}
    return post(port, "/v1/completions", body, timeout)["choices"][0]["text"]


def gsm8k(port, n):
    rc, out, _ = run(
        f"{sys.executable} -m sglang.test.few_shot_gsm8k --host 127.0.0.1 --port {port} --num-questions {n} "
        f"--num-shots 5 --parallel {IN_FLIGHT} --max-new-tokens 512", timeout=7200, cwd="/tmp",
    )  # fmt: skip
    m = re.findall(r"Accuracy: ([0-9.]+)", out)
    return float(m[-1]) if m else None


FILLER = (
    "The river ran low that summer, and the ferry kept to the deep channel near the eastern bank.",
    "Merchants in the old quarter opened their shutters at dawn and closed them again at noon.",
    "A committee met twice a month to argue about the price of grain and the state of the roads.",
    "The lighthouse keeper wrote the weather in a ledger, one line a day, for forty-one years.",
    "Children walked to the school on the hill, where the bell was rung by whoever came first.",
    "In winter the lake froze from the north shore, and the fishermen cut holes along its edge.",
    "The library kept maps of every parish, though few of them agreed on where the borders lay.",
    "A traveling choir sang in the square each spring and left before the first market day.",
)


def niah(port, model, lengths=(65536, 131072), depths=(0.1, 0.5, 0.9)):
    """Needle in a haystack: one number at each depth of a prompt of each length; the share of answers found."""
    import random

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    per = sum(len(tok.encode(" " + s, add_special_tokens=False)) for s in FILLER) / len(
        FILLER
    )
    rng = random.Random(0)
    found, notes = 0, []
    for n in lengths:
        for d in depths:
            num = rng.randrange(1_000_000, 9_999_999)
            body = [rng.choice(FILLER) for _ in range(int((n - 200) / per))]
            body.insert(
                int(d * len(body)),
                f"The special magic number for the harbor town is {num}.",
            )
            prompt = (
                " ".join(body)
                + "\n\nQuestion: What is the special magic number for the harbor town?\n"
                "Answer: The special magic number for the harbor town is"
            )
            got = complete(port, prompt, 24)
            ok = str(num) in got
            found += ok
            notes.append(f"{n // 1024}k@{d}:{'ok' if ok else repr(got[:30])}")
    return found / (len(lengths) * len(depths)), notes


def questions(model, n=48):
    from transformers import AutoTokenizer

    from sglang.utils import download_and_cache_file, read_jsonl

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rows = list(read_jsonl(download_and_cache_file(GSM8K_URL)))[:n]
    return [tok.encode(f"Question: {r['question']}\nAnswer:") for r in rows]


def decode_logprobs(port, ids, chunk=IN_FLIGHT):
    """Greedy tokens and their decode-time logprobs, in batches of ``chunk``. Prompt i decodes 32 * (1 + i % 8)
    tokens, so a batch of 8 shrinks from 8 requests to 1 and every width down to one request is served. Decode-time
    logprobs only: the recipe's decoder SWA bounded replay refuses prompt-token logprobs."""
    out = []
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        res = post(
            port,
            "/generate",
            {
                "input_ids": batch,
                "sampling_params": [
                    {
                        "temperature": 0.0,
                        "max_new_tokens": 32 * (1 + (i + j) % 8),
                        "ignore_eos": True,
                    }
                    for j in range(len(batch))
                ],
                "return_logprob": True,
                "return_text_in_logprobs": False,
            },
        )
        out += [
            (r["output_ids"], [x[0] for x in r["meta_info"]["output_token_logprobs"]])
            for r in res
        ]
    return out


def kl_between(a, b):
    """Two greedy runs of the same prompts: the mean over prompts of the per-token KL estimate
    (sglang.test.kl_test_utils' measure) over each prompt's common token prefix, the mean prefix length, and each
    prompt's first differing position (None when the tokens agree to the shorter end)."""
    kls, lens, first = [], [], []
    for (ta, la), (tb, lb) in zip(a, b):
        n, end = 0, min(len(ta), len(tb))
        while n < end and ta[n] == tb[n]:
            n += 1
        logr = [x - y for x, y in zip(la[:n], lb[:n])]
        kls.append(sum(math.exp(v) - 1 - v for v in logr) / max(n, 1))
        lens.append(n)
        first.append(n if n < end else None)
    return sum(kls) / len(kls), sum(lens) / len(lens), first


def kl_gate(on, off, off_chunked):
    """KL-consistency: switch on against off (both in batches of 8) within twice the switch-off server's own KL
    (batches of 4 against batches of 8) plus 0.001; token agreement: the mean common prefix on against off at least
    half the switch-off pair's, and the prompts that diverge before token 16 with the switch on no more than 10 % of
    the prompts above the switch-off pair's count. Token agreement is relative to the switch-off pair because the
    first token comes from the prefill, which the monokernels do not run."""
    kl, n, first = kl_between(on, off)
    noise, n0, first0 = kl_between(off_chunked, off)
    bound = 2 * noise + 1e-3
    early_on = sum(1 for f in first if f is not None and f < 16)
    early_off = sum(1 for f in first0 if f is not None and f < 16)
    margin = 0.1 * len(first)
    ok = kl <= bound and n >= 0.5 * n0 and (early_on - early_off) <= margin
    note = (
        f"KL on vs off {kl:.3g} (bound {bound:.3g} = 2 x switch-off {noise:.3g} + 0.001); mean common prefix "
        f"{n:.1f} tokens (switch-off pair {n0:.1f}); prompts diverging before token 16: on {early_on}, "
        f"switch-off pair {early_off} (allowed {early_off} + {margin:.1f} = 10 % of {len(first)})"
    )
    return ok, note


def accept_length(text, in_flight=IN_FLIGHT):
    """The mean DSpark accept length of the server's decode log lines with ``in_flight`` running requests."""
    vals = [
        float(a)
        for r, a in re.findall(
            r"#running-req: (\d+),.*?accept len: ([0-9.]+)", text, flags=re.M
        )
        if int(r) == in_flight
    ]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def gates(args, srv, tag, decode):
    """GSM8K, NIAH (verify steps) and greedy decode logprobs on one server, kept in args.state[tag]. A gate that
    raises is recorded as its error; the others still run."""
    out, ids = {}, None
    port = srv.port
    jobs = [("gsm8k", lambda: gsm8k(port, args.gsm8k))]
    if args.niah and not decode:
        jobs.append(("niah", lambda: niah(port, args.model)))
    if args.kl:
        jobs.append(("logprobs", lambda: decode_logprobs(port, ids)))
        if tag.startswith("off"):
            jobs.append(
                ("logprobs_chunked", lambda: decode_logprobs(port, ids, IN_FLIGHT // 2))
            )
    if args.kl:
        ids = questions(args.model)
    for name, fn in jobs:
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["niah"], out["niah_notes"] = out.get("niah") or (None, [])
    if not decode:
        out["accept"] = accept_length(srv.text())
    args.state[tag] = out
    return out


def shown(g):
    """A gates() result without the logprob lists."""
    return {
        k: v
        for k, v in g.items()
        if not k.startswith("logprobs") or k.endswith("error")
    }


PROMPTS = [f"Question {i}: what is {i} times {i + 3}? Answer:" for i in range(64)]


def crossing(srv):
    """Decode steps of 48 rows (monokernels), then 56 (the model path), then one request again: every answer
    non-empty, the one-request answer the same before and after, and both kinds of step in the log."""
    port = srv.port
    first = complete(port, PROMPTS[0], 64)
    empty = 0
    for n in (48, 56):
        with ThreadPoolExecutor(n) as ex:
            got = list(ex.map(lambda p: complete(port, p, 64), PROMPTS[:n]))
        empty += sum(not g.strip() for g in got)
    again = complete(port, PROMPTS[0], 64)
    log = srv.text()
    ran48 = bool(re.search(r"Decode batch, #running-req: 48,", log))
    ran56 = bool(re.search(r"Decode batch, #running-req: (49|5\d|6[0-4]),", log))
    model_path = re.findall(r"decode steps of (\d+) rows take the model path", log)
    ok = not empty and first == again and ran48 and ran56 and bool(model_path)
    note = (
        f"crossing: 48-row steps served {ran48}, steps above 48 served {ran56}, model-path widths {model_path}, "
        f"empty answers {empty}, one-request answer unchanged {first == again}"
    )
    return ok, note


def step_cpu(args):
    rc, out, s = run(
        f"{sys.executable} -m pytest -q -p no:cacheprovider {UNIT}",
        timeout=1800,
        cwd=ROOT,
    )
    return rc == 0, out.strip().splitlines()[-1] if out.strip() else "", s


def step_kernels(args):
    notes, ok, total = [], True, 0.0
    for test in ("test_dsv41_ffn.py", "test_dsv41_k1.py"):
        # each file relaunches itself under torchrun at TP2 and TP4
        rc, out, s = run(f"{sys.executable} {test}", timeout=7200, cwd=KERNELS)
        total += s
        good = rc == 0
        ok &= good
        notes.append(f"{test} {'PASS' if good else 'FAIL'}")
        with open(os.path.join(args.out, f"kernels_{test[:-3]}.log"), "w") as f:
            f.write(out)
    return ok, "; ".join(notes), total


def step_guard(args):
    rc, out, s = run(
        f"{sys.executable} test_guard.py --tp {args.tp}", timeout=7200, cwd=KERNELS
    )
    rc2, out2, s2 = run(
        f"{sys.executable} test_guard.py --tp 2 --sites xrdy --after 0,3",
        timeout=1800,
        cwd=KERNELS,
    )
    # a step that ends with a TP collective: at the last exchange no rank finishes it (RFC section 8)
    rc3, out3, s3 = run(
        f"{sys.executable} test_guard.py --tp {args.tp} --sites ffn,epoch --after 0,3 --mode step",
        timeout=1800,
        cwd=KERNELS,
    )
    with open(os.path.join(args.out, "guard.log"), "w") as f:
        f.write(out + out2 + out3)
    lines = [
        ln for ln in (out + out2 + out3).splitlines() if ln.startswith(("PASS", "FAIL"))
    ]
    return (
        rc == 0 and rc2 == 0 and rc3 == 0,
        f"{sum(ln.startswith('PASS') for ln in lines)} of {len(lines)} cases",
        s + s2 + s3,
    )


def step_refuse(args):
    # three features the spec does not support, each accepted by SGLang itself: every reason at once, before loading
    t0 = time.time()
    extra = "--disable-decode-cuda-graph --enable-return-hidden-states --enable-memory-saver"
    srv = Server(args, mono=True, extra=extra)
    srv.p.wait(timeout=900)
    text = srv.text()
    reasons = sorted(set(re.findall(r"^\s+- (.+)$", text, flags=re.M)))
    ok = (
        srv.p.returncode != 0
        and len(reasons) >= 3
        and "refused" in text
        and "Load weight" not in text
    )
    return ok, f"exit {srv.p.returncode}; reasons {reasons}", time.time() - t0


def step_serve(args, decode=False):
    tag = "on_decode" if decode else "on"
    srv = Server(args, mono=True, decode=decode)
    t0 = time.time()
    try:
        if not srv.wait_ready(args.load_timeout):
            return False, "server did not start", time.time() - t0
        ready = time.time() - t0
        answer = complete(args.port, "The capital of France is", 16)
        if decode:
            args.state["fault_ref"] = stream_generate(args.port, FAULT_PROMPT, 512)
        g = gates(args, srv, tag, decode)
        cross_ok, cross_note = crossing(srv) if decode else (True, "")
        text = srv.text()
        plans = re.findall(r"(\w+) plan \w+: (\d+) layers, spans (\{[^}]*\})", text)
        captured = len(re.findall(r"captured \w+ \d+ rows on monokernels", text))
        ok = (
            bool(plans)
            and captured > 0
            and "Paris" in answer
            and g["gsm8k"] is not None
            and (decode or g["niah"] == 1.0 or not args.niah)
            and (g.get("logprobs") is not None or not args.kl)
            and cross_ok
            and srv.p.poll() is None
        )
        return (
            ok,
            f"ready {ready:.0f} s; plans {plans}; captured {captured}; answer {answer!r}; {shown(g)}; {cross_note}",
            time.time() - t0,
        )
    finally:
        srv.stop()


def step_ab(args, decode=False):
    on_tag, off_tag = ("on_decode", "off_decode") if decode else ("on", "off")
    srv = Server(args, mono=False, decode=decode)
    t0 = time.time()
    try:
        if not srv.wait_ready(args.load_timeout):
            return False, "server did not start", time.time() - t0
        off = gates(args, srv, off_tag, decode)
        on = args.state.get(on_tag) or {}
        acc_on, acc_off = on.get("gsm8k"), off["gsm8k"]
        ok = (
            acc_on is not None
            and acc_off is not None
            and abs(acc_on - acc_off) * 100 <= args.gsm8k_gap
        )
        note = f"GSM8K on {acc_on} off {acc_off} (gap limit {args.gsm8k_gap} points)"
        if args.niah and not decode:
            ok &= on.get("niah") == 1.0 and off["niah"] == 1.0
            note += f"; NIAH on {on.get('niah')} {on.get('niah_notes')} off {off['niah']} {off['niah_notes']}"
        if args.kl:
            if (
                on.get("logprobs")
                and off.get("logprobs")
                and off.get("logprobs_chunked")
            ):
                kl_ok, kl_note = kl_gate(
                    on["logprobs"], off["logprobs"], off["logprobs_chunked"]
                )
                ok &= kl_ok
                note += f"; {kl_note}"
            else:
                ok = False
                note += f"; KL missing: on {shown(on)} off {shown(off)}"
        if not decode:
            (a_on, n_on), (a_off, n_off) = on.get("accept", (None, 0)), off["accept"]
            drop = None if not (a_on and a_off) else 100 * (a_off - a_on) / a_off
            ok &= drop is not None and drop <= 3.0
            note += (
                f"; DSpark accept length at {IN_FLIGHT} requests: on {a_on} ({n_on} log lines), off {a_off} "
                f"({n_off}); drop {drop if drop is None else round(drop, 2)} % (limit 3 %)"
            )
        return ok, note, time.time() - t0
    finally:
        srv.stop()


def stream(port, prompt, n, timeout=900):
    """Stream a completion: (text chunks received, time of the last one, finished, client error)."""
    body = json.dumps(
        {"prompt": prompt, "max_tokens": n, "temperature": 0, "stream": True}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        body,
        {"Content-Type": "application/json"},
    )
    chunks, last, finished, err = 0, time.time(), False, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    finished = True
                    break
                choice = json.loads(data)["choices"][0]
                if choice.get("text"):
                    chunks, last = chunks + 1, time.time()
                finished |= bool(choice.get("finish_reason"))
    except Exception as e:
        err = type(e).__name__
    return chunks, last, finished, err


FAULT_PROMPT = "Count from 1 to 400, one number a line:"


def stream_generate(port, text, n, timeout=900):
    """Stream /generate greedily: (the token ids received, time of the last, finished, client error)."""
    body = json.dumps(
        {
            "text": text,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": n,
                "ignore_eos": True,
            },
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", body, {"Content-Type": "application/json"}
    )
    ids, last, finished, err = [], time.time(), False, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                got = list(obj.get("output_ids") or [])
                if got:
                    # cumulative or incremental chunks
                    ids = got if got[: len(ids)] == ids else ids + got
                    last = time.time()
                finished |= obj.get("meta_info", {}).get("finish_reason") is not None
    except Exception as e:
        err = type(e).__name__
    return ids, last, finished, err


def step_fault(args):
    # stall rank 1's 'pmix' wait (mid-step: no rank can finish that step) well after capture, during a streamed
    # request: every rank must stop within the deadline plus a margin and the request must not finish
    srv = Server(
        args,
        mono=True,
        env={"SGLANG_TEST_MONO_FAULT_INJECT": f"1:pmix:{args.fault_after}"},
    )
    t0 = time.time()
    try:
        if not srv.wait_ready(args.load_timeout):
            return False, "server did not start", time.time() - t0
        chunks, last, finished, err = stream(args.port, FAULT_PROMPT, 2048)
        try:
            srv.p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            pass
        stopped = srv.p.poll() is not None
        gone = time.time() - last
        ok = stopped and not finished and chunks > 0 and gone <= DEADLINE_S + 90 + 300
        return (
            ok,
            f"server stopped {stopped}; {chunks} chunks streamed, then nothing; last chunk to server exit "
            f"{gone:.0f} s (deadline {DEADLINE_S} s); request finished {finished}; client {err}",
            time.time() - t0,
        )
    finally:
        srv.stop()


def step_fault_last(args):
    # decode steps, one request: every forward on monokernels runs the ffn span on all 40 layers, so the FFN
    # launch 40 m + 39 is the last layer of a forward whatever the warm-up count. Rank 1 stalls at that launch's
    # epoch wait, after its last send: its peers finish the launch, but the step's logits gather needs rank 1, so
    # no token of the step can be sent (RFC section 8)
    ref = args.state.get("fault_ref")
    after = 40 * args.fault_forwards + 39
    srv = Server(
        args,
        mono=True,
        env={"SGLANG_TEST_MONO_FAULT_INJECT": f"1:epoch:{after}"},
        decode=True,
    )
    t0 = time.time()
    try:
        if not srv.wait_ready(args.load_timeout):
            return False, "server did not start", time.time() - t0
        ids, last, finished, err = stream_generate(args.port, FAULT_PROMPT, 512)
        try:
            srv.p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            pass
        stopped = srv.p.poll() is not None
        gone = time.time() - last
        ref_ids = ref[0] if ref else None
        matches = ref_ids is not None and ids == ref_ids[: len(ids)]
        ok = (
            stopped
            and not finished
            and 0 < len(ids) < 512
            and matches
            and gone <= DEADLINE_S + 90 + 300
        )
        return (
            ok,
            f"stall at FFN launch {after} (last layer of a forward); server stopped {stopped}; {len(ids)} tokens "
            f"sent, all equal to the answer without the fault: {matches}; last token to server exit {gone:.0f} s; "
            f"request finished {finished}; client {err}",
            time.time() - t0,
        )
    finally:
        srv.stop()


STEPS = {
    "cpu": step_cpu,
    "kernels": step_kernels,
    "guard": step_guard,
    "refuse": step_refuse,
    "serve": step_serve,
    "ab": step_ab,
    "serve_decode": lambda a: step_serve(a, decode=True),
    "ab_decode": lambda a: step_ab(a, decode=True),
    "fault": step_fault,
    "fault_last": step_fault_last,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--port", type=int, default=31170)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--gsm8k", type=int, default=1319)
    ap.add_argument("--gsm8k-gap", type=float, default=1.0)
    ap.add_argument("--niah", type=int, default=1)
    ap.add_argument("--kl", type=int, default=1)
    ap.add_argument("--host-table", type=int, default=1)
    ap.add_argument("--fault-after", type=int, default=4000)
    # forwards on monokernels before the last-layer stall: past the capture warm-ups and the server's warm-up
    ap.add_argument("--fault-forwards", type=int, default=150)
    ap.add_argument("--load-timeout", type=int, default=3600)
    ap.add_argument("--out", default="validate_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.state = {}
    results = []
    for name in args.steps.split(","):
        try:
            ok, note, secs = STEPS[name](args)
        except Exception as e:
            ok, note, secs = False, f"{type(e).__name__}: {e}", 0.0
        results.append((name, ok, note, secs))
        print(f"{'PASS' if ok else 'FAIL'} {name} ({secs:.0f} s): {note}", flush=True)
    print(
        f"{sum(ok for _, ok, _, _ in results)} of {len(results)} steps passed",
        flush=True,
    )
    sys.exit(0 if all(ok for _, ok, _, _ in results) else 1)


if __name__ == "__main__":
    main()
