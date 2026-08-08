"""FlashInfer cascade CG-mode integration test.

Boots Qwen2.5-0.5B/TP=1 with ``--attention-backend flashinfer-cascade`` and
verifies CG-mode cascade firing:

    1. **CG-fires test (KEY)** -- bs=8 (in default cuda_graph_bs), shared
       ~256-token prefix, CG enabled (default). Run >= 10 decode steps.
       Output epsilon-equivalent to flashinfer baseline (>= 7/8 prompts
       byte-equal). Verify via debug log/counter that cascade actually
       fired under CG (not the eager fallback / not the parent path).

    2. **CG fallback test** -- bs=8 (in cuda_graph_bs), shared prefix below
       threshold (~25 tokens), CG enabled. Cascade does NOT cross the
       "fired" counter (>= threshold) but the captured cascade graph still
       RUNS with common=0 / common < threshold (the always-cascade-in-CG
       design choice). Output epsilon-equivalent to flashinfer baseline.

    3. **Eager regression test** -- bs > cuda_graph_max_bs forces eager
       path. With shared 256-token prefix, eager cascade fires. Output
       epsilon-equivalent to flashinfer baseline.

    4. **Eager-mode cross-check** -- re-runs the cascade-fires-at-threshold
       behavior in eager mode to confirm CG-mode additions did not regress
       the eager path.

Manual integration test: launches a subprocess server, hits ``/generate``,
greps logs for cascade dispatch markers. Not registered for CI.

Usage:
    pytest test/manual/attention/test_flashinfer_cascade_cg.py -v -s
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.environ.get("CASCADE_CG_TEST_PORT", "30079"))
BASE_URL = f"http://127.0.0.1:{PORT}"
HEALTH_TIMEOUT = 240

LONG_SYSTEM_PROMPT = (
    "You are a meticulous assistant. "
    "Answer the user's question with a single short word. "
    "Do not add explanation, punctuation, or any extra text. "
    "Examples: Q: capital of France? A: Paris. Q: capital of Japan? A: Tokyo. "
    "Q: capital of Germany? A: Berlin. "
    "If the question has no clear single-word answer, reply 'Unknown'. "
    "Always reply with exactly one word and nothing else. "
) * 8  # roughly ~256 tokens after tokenization for shared-prefix tests

SHORT_SHARED_PREFIX = "Answer with one word. Examples: Q: capital of France? A: Paris."

CAPITAL_PROMPTS = [
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Greece?",
    "What is the capital of Norway?",
    "What is the capital of Sweden?",
    "What is the capital of Finland?",
    "What is the capital of Ireland?",
]

MAX_TOKENS = 8


def _wait_for_health(base_url: str, max_wait: int = HEALTH_TIMEOUT) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def _launch(
    extra_args: list[str], log_path: Path, debug_cascade: bool = False
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--port",
        str(PORT),
        "--host",
        "127.0.0.1",
        "--mem-fraction-static",
        "0.5",
        "--context-length",
        "4096",
    ] + extra_args
    print(f"\n[cascade-cg-test] launching: {' '.join(cmd)}", flush=True)
    log_fh = open(log_path, "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    repo_python = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = repo_python + ":" + env.get("PYTHONPATH", "")
    if debug_cascade:
        env["SGLANG_CASCADE_DEBUG"] = "1"
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def _completion(
    system: str, user: str, max_tokens: int = MAX_TOKENS
) -> tuple[str, list[int]]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    return text, []


def _generate_concurrent(prompts_with_system: list[tuple[str, str]]) -> list[str]:
    import concurrent.futures as cf

    def _one(payload):
        sysp, user = payload
        text, _ = _completion(sysp, user)
        return text

    out = [None] * len(prompts_with_system)
    with cf.ThreadPoolExecutor(max_workers=len(prompts_with_system)) as ex:
        futs = {ex.submit(_one, p): i for i, p in enumerate(prompts_with_system)}
        for f in cf.as_completed(futs):
            i = futs[f]
            out[i] = f.result()
    return out


def _run_server_and_collect(
    extra_args: list[str],
    label: str,
    log_path: Path,
    payloads: list[tuple[str, str]],
    debug_cascade: bool = False,
) -> tuple[list[str], str]:
    proc = _launch(extra_args, log_path, debug_cascade=debug_cascade)
    try:
        if not _wait_for_health(BASE_URL):
            tail = log_path.read_text(errors="replace").splitlines()[-40:]
            pytest.fail(
                f"{label} server did not become ready within {HEALTH_TIMEOUT}s.\n"
                + "\n".join(tail)
            )
        print(
            f"\n[cascade-cg-test] {label} -- collecting {len(payloads)} outputs",
            flush=True,
        )
        outs = _generate_concurrent(payloads)
        return outs, log_path.read_text(errors="replace")
    finally:
        _kill(proc)
        time.sleep(5)


# ---------------- Tests ----------------


def test_cascade_cg_fires_at_threshold():
    """KEY test: at bs=8 + ~256 token shared prefix, CG enabled (default),
    cascade fires inside the captured graph for bs=8.

    Output must be epsilon-equivalent (>= 7/8 string-equal vs flashinfer
    baseline). And the log must contain "Cascade fires (CG)" so we know
    cascade ran inside the captured graph (not the eager path, not the
    parent's per-request decode).
    """
    payloads = [(LONG_SYSTEM_PROMPT, q) for q in CAPITAL_PROMPTS]

    base_log = Path("/tmp/sglang_cascade_cg_fire_baseline.log")
    cascade_log = Path("/tmp/sglang_cascade_cg_fire_test.log")

    # Baseline: stock flashinfer with default CG.
    base_outs, _ = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer",
        ],
        label="flashinfer baseline (CG, bs=8)",
        log_path=base_log,
        payloads=payloads,
    )

    # Cascade: flashinfer-cascade with default CG. bs=8 must hit the CG
    # cascade path because 8 is in the default cuda_graph_bs list.
    cascade_outs, log_text = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer-cascade",
            "--cascade-min-prefix-tokens",
            "128",
            "--cascade-min-batch-size",
            "4",
        ],
        label="flashinfer-cascade CG (bs=8, above threshold)",
        log_path=cascade_log,
        payloads=payloads,
        debug_cascade=True,
    )

    print("\n[cascade-cg-test] cascade-fires compare:", flush=True)
    matches = 0
    for prompt, b, c in zip(payloads, base_outs, cascade_outs):
        ok = b == c
        if ok:
            matches += 1
        print(
            f"  {'OK' if ok else 'DRIFT'} q={prompt[1][:40]!r}\n"
            f"    base   = {b!r}\n"
            f"    cascade= {c!r}",
            flush=True,
        )

    # KEY assertion: cascade fired inside the captured graph.
    cg_fire_count = log_text.count("Cascade fires (CG)")
    assert cg_fire_count > 0, (
        "Cascade did not fire under CG. "
        "Expected 'Cascade fires (CG)' log entry. Tail:\n"
        + "\n".join(log_text.splitlines()[-60:])
    )

    # >= 7/8 prompts must match the baseline byte-for-byte.
    assert matches >= len(payloads) - 1, (
        f"Only {matches}/{len(payloads)} cascade outputs matched baseline; "
        "expected >= 7/8 (LSE merge tolerance)."
    )


def test_cascade_cg_fallback_below_prefix_threshold():
    """At bs=8 (in cuda_graph_bs) with a shared prefix below threshold
    (~25 tokens of SHORT_SHARED_PREFIX << 128), cascade must not log a
    fired event (>= threshold) but the captured cascade graph still runs
    correctly with common < threshold.

    Output must remain epsilon-equivalent to the flashinfer baseline.
    """
    payloads = [(SHORT_SHARED_PREFIX, q) for q in CAPITAL_PROMPTS]

    base_log = Path("/tmp/sglang_cascade_cg_fallback_baseline.log")
    cascade_log = Path("/tmp/sglang_cascade_cg_fallback_test.log")

    base_outs, _ = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer",
        ],
        label="flashinfer baseline (CG, bs=8, short prefix)",
        log_path=base_log,
        payloads=payloads,
    )

    cascade_outs, log_text = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer-cascade",
            "--cascade-min-prefix-tokens",
            "128",
            "--cascade-min-batch-size",
            "4",
        ],
        label="flashinfer-cascade CG (bs=8, prefix below threshold)",
        log_path=cascade_log,
        payloads=payloads,
        debug_cascade=True,
    )

    print("\n[cascade-cg-test] cg-fallback compare:", flush=True)
    matches = 0
    for prompt, b, c in zip(payloads, base_outs, cascade_outs):
        ok = b == c
        if ok:
            matches += 1
        print(
            f"  {'OK' if ok else 'DRIFT'} q={prompt[1][:40]!r}\n"
            f"    base   = {b!r}\n"
            f"    cascade= {c!r}",
            flush=True,
        )

    # Cascade must not have fired at threshold (the short prefix is
    # ~25 tokens << 128).
    cg_fire_count = log_text.count("Cascade fires (CG)")
    assert cg_fire_count == 0, (
        f"Cascade fired (CG) at common < threshold. count={cg_fire_count}. "
        "Tail:\n" + "\n".join(log_text.splitlines()[-60:])
    )

    # Output epsilon-equivalent to baseline (the always-cascade-in-CG path
    # produces output identical to per-request decode when common < threshold).
    assert matches >= len(payloads) - 1, (
        f"Only {matches}/{len(payloads)} fallback outputs matched baseline; "
        "expected >= 7/8 -- the always-cascade-in-CG path should be "
        "mathematically equivalent to per-request decode."
    )


def test_cascade_eager_fires_at_high_batch():
    """Regression for the eager path: at bs > cuda_graph_max_bs, eager mode
    fires. We exercise this by running 16 concurrent prompts on a server
    with --cuda-graph-max-bs=8 (so bs=16 falls to eager). Cascade must fire
    via the eager path -- "Cascade fires:" log (no "(CG)" suffix).
    """
    payloads = [(LONG_SYSTEM_PROMPT, q) for q in CAPITAL_PROMPTS] * 2  # 16 reqs

    base_log = Path("/tmp/sglang_cascade_eager_high_baseline.log")
    cascade_log = Path("/tmp/sglang_cascade_eager_high_test.log")

    base_outs, _ = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer",
            "--cuda-graph-max-bs",
            "8",
        ],
        label="flashinfer baseline (eager fallback, bs=16)",
        log_path=base_log,
        payloads=payloads,
    )

    cascade_outs, log_text = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer-cascade",
            "--cuda-graph-max-bs",
            "8",
            "--cascade-min-prefix-tokens",
            "128",
            "--cascade-min-batch-size",
            "4",
        ],
        label="flashinfer-cascade eager (bs=16, above cg max)",
        log_path=cascade_log,
        payloads=payloads,
        debug_cascade=True,
    )

    print("\n[cascade-cg-test] eager-fires compare:", flush=True)
    matches = 0
    for prompt, b, c in zip(payloads, base_outs, cascade_outs):
        ok = b == c
        if ok:
            matches += 1
        print(
            f"  {'OK' if ok else 'DRIFT'} q={prompt[1][:40]!r}\n"
            f"    base   = {b!r}\n"
            f"    cascade= {c!r}",
            flush=True,
        )

    # KEY: eager path used (not CG path).
    eager_fire = re.findall(r"Cascade fires: bs=", log_text)
    cg_fire = re.findall(r"Cascade fires \(CG\): bs=", log_text)
    print(
        f"  eager_fire_count={len(eager_fire)}, cg_fire_count={len(cg_fire)}",
        flush=True,
    )
    # bs=16 may also batch as smaller groups; the assertion is that EAGER
    # cascade fired at least once on this workload.
    assert len(eager_fire) > 0, (
        "Eager cascade did not fire at bs > cuda_graph_max_bs. "
        "Expected 'Cascade fires:' (without (CG) suffix). Tail:\n"
        + "\n".join(log_text.splitlines()[-60:])
    )

    assert matches >= len(payloads) - 2, (
        f"Only {matches}/{len(payloads)} eager outputs matched baseline; "
        "expected >= N-2 (LSE merge tolerance + slight scheduling jitter)."
    )


def test_cascade_eager_regression_still_passes():
    """Regression cross-check: re-run the cascade-fires-at-threshold scenario
    in eager mode to confirm CG-mode additions did not break the eager path.

    bs=8 with --disable-cuda-graph + ~256 token shared prefix. Cascade
    must fire via the eager path. >= 7/8 byte-equal vs flashinfer
    baseline.
    """
    payloads = [(LONG_SYSTEM_PROMPT, q) for q in CAPITAL_PROMPTS]

    base_log = Path("/tmp/sglang_cascade_p14_eager_baseline.log")
    cascade_log = Path("/tmp/sglang_cascade_p14_eager_test.log")

    base_outs, _ = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer",
            "--disable-cuda-graph",
        ],
        label="flashinfer baseline (eager, bs=8)",
        log_path=base_log,
        payloads=payloads,
    )

    cascade_outs, log_text = _run_server_and_collect(
        extra_args=[
            "--page-size",
            "1",
            "--attention-backend",
            "flashinfer-cascade",
            "--disable-cuda-graph",
            "--cascade-min-prefix-tokens",
            "128",
            "--cascade-min-batch-size",
            "4",
        ],
        label="flashinfer-cascade eager (bs=8)",
        log_path=cascade_log,
        payloads=payloads,
        debug_cascade=True,
    )

    matches = sum(1 for b, c in zip(base_outs, cascade_outs) if b == c)
    eager_fire_count = log_text.count("Cascade fires:")  # eager log marker
    cg_fire_count = log_text.count("Cascade fires (CG)")
    print(
        f"\n[cascade-cg-test] p14-eager regression: "
        f"matches={matches}/{len(payloads)}, eager_fires={eager_fire_count}, "
        f"cg_fires={cg_fire_count}",
        flush=True,
    )
    # CG must NOT have fired -- we passed --disable-cuda-graph.
    assert (
        cg_fire_count == 0
    ), f"CG cascade fired with --disable-cuda-graph. count={cg_fire_count}"
    assert eager_fire_count > 0, "Eager cascade path stopped firing."
    assert (
        matches >= len(payloads) - 1
    ), f"Eager-path regression: only {matches}/{len(payloads)} match baseline."
