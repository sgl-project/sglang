"""GPU E2E for an externally installed LMCacheUnifiedConnector.

Run inside the SGLang/LMCache environment documented in the PR evidence.
This script starts and stops only its own server process groups. It requires
two otherwise unused CUDA devices for the default TP=1,2 matrix.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


class Validation:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.evidence_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.url = f"http://127.0.0.1:{args.port}"
        self.lm_url = f"http://127.0.0.1:{args.lm_http_port}"
        self.processes = []
        self.process_names = {}
        self.stops = []
        self.results = []
        self.commands = []
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        text = (
            "The research laboratory records each observation carefully. "
            "Water freezes at zero degrees Celsius at standard pressure. "
            "A good experiment compares its result with a control group.\n"
        ) * 200
        self.tokens = tokenizer.encode(text)[:2049]
        assert len(self.tokens) == 2049
        self.extended_tokens = (
            self.tokens
            + tokenizer.encode(
                "A different observation must be computed after the cached prefix. "
                * 40
            )[:256]
        )
        self.write("prompt.json", {"input_ids": self.tokens})

    def write(self, name, value):
        (self.out / name).write_text(json.dumps(value, indent=2) + "\n")

    def start(self, name, command, url):
        log = (self.out / f"{name}.log").open("w")
        self.commands.append({"name": name, "argv": command})
        self.write("commands.json", self.commands)
        env = dict(os.environ, LMCACHE_LOG_LEVEL="DEBUG", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        log.close()
        self.processes.append(proc)
        self.process_names[proc.pid] = name
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited {proc.returncode}; inspect {name}.log"
                )
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    print(f"READY {name}", flush=True)
                    return proc
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"{name} startup exceeded 1200 seconds")

    def stop(self, proc, check=False):
        forced = False
        if proc.poll() is None:
            # Let the frontend dispatch ShutdownReq to scheduler workers so
            # release_host_resources() can unregister CUDA IPC allocations.
            proc.terminate()
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                forced = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        self.processes.remove(proc)
        self.stops.append(
            {
                "name": self.process_names[proc.pid],
                "returncode": proc.returncode,
                "forced_kill": forced,
            }
        )
        self.write("process-exits.json", self.stops)
        if check:
            assert not forced and proc.returncode == 0, self.stops[-1]

    def backend_idle(self, label, registered=None):
        deadline = time.monotonic() + 10
        while True:
            response = requests.get(self.lm_url + "/status", timeout=10)
            response.raise_for_status()
            status = response.json()
            l1 = status["storage_manager"]["l1_manager"]
            idle = (
                status["active_sessions"] == 0
                and status["active_prefetch_jobs"] == 0
                and l1["write_locked_count"] == 0
                and l1["read_locked_count"] == 0
                and l1["temporary_count"] == 0
            )
            if registered is not None:
                idle = idle and len(status["registered_gpu_ids"]) == registered
            if idle or time.monotonic() >= deadline:
                self.write(f"{label}-backend-status.json", status)
                assert idle, status
                return
            time.sleep(0.1)

    def sglang(self, tp, external, suffix=""):
        name = f"tp{tp}-{'external' if external else 'baseline'}{suffix}"
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.args.model,
            "--revision",
            self.args.revision,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.port),
            "--tp-size",
            str(tp),
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "flashinfer",
            "--page-size",
            "16",
            "--chunked-prefill-size",
            "512",
            "--max-total-tokens",
            "16384",
            "--context-length",
            "8192",
            "--max-running-requests",
            "8",
            "--cuda-graph-max-bs-decode",
            "8",
            "--mem-fraction-static",
            "0.25",
            "--random-seed",
            "42",
            "--enable-metrics",
        ]
        if external:
            command += [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "LMCacheUnifiedConnector",
                        "kv_connector_module_path": (
                            "lmcache.integration.sglang.external_kv_connector"
                        ),
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "lmcache.mp.host": "tcp://127.0.0.1",
                            "lmcache.mp.port": self.args.lm_port,
                        },
                    }
                ),
            ]
        else:
            command.append("--disable-radix-cache")
        return self.start(name, command, self.url + "/health")

    def flush(self, label):
        response = requests.post(self.url + "/flush_cache?timeout=30", timeout=40)
        (self.out / f"{label}-flush.txt").write_text(response.text)
        response.raise_for_status()

    def clear(self, label):
        self.flush(label)
        response = requests.post(
            self.url + "/hicache/storage-backend/clear", timeout=60
        )
        (self.out / f"{label}-clear.txt").write_text(response.text)
        response.raise_for_status()

    def payload(self, label, salt, input_ids=None):
        return {
            "rid": label,
            "input_ids": self.tokens if input_ids is None else input_ids,
            "cache_salt": salt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 32,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        }

    def generate(self, label, salt, baseline=None, cache=None, input_ids=None):
        payload = self.payload(label, salt, input_ids)
        self.write(f"{label}-request.json", payload)
        started = time.monotonic()
        response = requests.post(self.url + "/generate", json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        self.write(f"{label}-response.json", result)
        meta = result["meta_info"]
        details = meta.get("cached_tokens_details") or {}
        tokens = result["output_ids"]
        assert len(tokens) == 32, result
        assert meta["completion_tokens"] == 32, meta
        assert meta["finish_reason"]["type"] == "length", meta
        logprob_diff = None
        if baseline is not None:
            assert tokens == baseline["output_ids"], f"Output token mismatch: {label}"
            assert result["text"] == baseline["text"], f"Output text mismatch: {label}"
            expected = baseline["meta_info"]["output_token_logprobs"]
            actual = meta["output_token_logprobs"]
            assert len(expected) == len(actual) == 32
            logprob_diff = max(
                abs(x[0] - y[0]) for x, y in zip(expected, actual, strict=True)
            )
            # Allow small BF16 batching differences while checking more than
            # argmax token identity. This is an absolute log-probability bound.
            assert logprob_diff <= 1e-3, (label, logprob_diff)
        storage = details.get("storage", 0)
        device = details.get("device", 0)
        if cache == "cold":
            assert meta["cached_tokens"] == 0 and storage == 0, meta
        elif cache == "external":
            assert storage == 2048 and device == 0, meta
            assert details["storage_backend"] == "LMCacheUnifiedConnector", meta
        elif cache == "local":
            assert device > 0 and storage == 0, meta
        row = {
            "case": label,
            "passed": True,
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": meta["completion_tokens"],
            "cached_tokens": meta["cached_tokens"],
            "device": device,
            "storage": storage,
            "output_matches_baseline": baseline is not None,
            "max_output_logprob_abs_diff": logprob_diff,
            "output_ids_sha256": hashlib.sha256(
                json.dumps(tokens).encode()
            ).hexdigest(),
            "elapsed_seconds": time.monotonic() - started,
        }
        self.results.append(row)
        print(json.dumps(row), flush=True)
        return result

    def cancel(self, label, salt):
        self.flush(label)
        payload = self.payload(label, salt)
        payload["stream"] = True
        payload["sampling_params"]["max_new_tokens"] = 2048
        self.write(f"{label}-request.json", payload)
        with requests.post(
            self.url + "/generate", json=payload, stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            events = response.iter_lines()
            for line in events:
                if line.startswith(b"data: {"):
                    first = json.loads(line[6:])
                    self.write(f"{label}-first-event.json", first)
                    assert first["meta_info"]["completion_tokens"] < 2048
                    break
            else:
                raise AssertionError("Cancellation request produced no data")
            abort = requests.post(
                self.url + "/abort_request", json={"rid": label}, timeout=30
            )
            abort.raise_for_status()
            (self.out / f"{label}-abort.txt").write_text(abort.text)
            tail = [line.decode() for line in events]
            (self.out / f"{label}-stream-tail.txt").write_text("\n".join(tail))
            terminal = [
                json.loads(line[6:]) for line in tail if line.startswith("data: {")
            ]
            assert terminal[-1]["meta_info"]["finish_reason"]["type"] == "abort"
        self.flush(label + "-after")
        self.backend_idle(label + "-after")
        self.results.append(
            {"case": label, "passed": True, "drained_after_abort": True}
        )

    def run(self):
        try:
            self.start(
                "lmcache",
                [
                    "lmcache",
                    "server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.args.lm_port),
                    "--http-host",
                    "127.0.0.1",
                    "--http-port",
                    str(self.args.lm_http_port),
                    "--chunk-size",
                    "256",
                    "--l1-size-gb",
                    "4",
                    "--eviction-policy",
                    "LRU",
                    "--max-gpu-workers",
                    "2",
                    "--max-cpu-workers",
                    "4",
                ],
                self.lm_url + "/healthcheck",
            )
            for tp in self.args.tp:
                salt = f"external-connector-e2e-tp{tp}"
                prefix = f"tp{tp}"
                baseline_proc = self.sglang(tp, False)
                baseline = self.generate(prefix + "-baseline", salt, cache="cold")
                short_baseline = self.generate(
                    prefix + "-short-baseline",
                    salt,
                    cache="cold",
                    input_ids=self.tokens[:129],
                )
                extended_baseline = self.generate(
                    prefix + "-extended-baseline",
                    salt,
                    cache="cold",
                    input_ids=self.extended_tokens,
                )
                self.stop(baseline_proc, check=True)
                proc = self.sglang(tp, True)
                self.clear(prefix + "-initial")
                self.generate(prefix + "-cold", salt, baseline, "cold")
                self.generate(prefix + "-local-warm", salt, baseline, "local")
                self.flush(prefix + "-external-warm")
                self.generate(prefix + "-external-warm", salt, baseline, "external")
                self.flush(prefix + "-partial-hit")
                self.generate(
                    prefix + "-partial-hit",
                    salt,
                    extended_baseline,
                    "external",
                    self.extended_tokens,
                )
                self.flush(prefix + "-short-prompt")
                self.generate(
                    prefix + "-short-prompt",
                    salt,
                    short_baseline,
                    "cold",
                    self.tokens[:129],
                )
                self.flush(prefix + "-salt-isolation")
                self.generate(
                    prefix + "-salt-isolation", salt + "-isolated", baseline, "cold"
                )
                self.flush(prefix + "-concurrent")
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(
                            self.generate, f"{prefix}-concurrent-{i}", salt, baseline
                        )
                        for i in range(4)
                    ]
                    for future in futures:
                        future.result()
                rows = [
                    r
                    for r in self.results
                    if r["case"].startswith(prefix + "-concurrent-")
                ]
                assert sum(r["storage"] for r in rows) > 0
                self.cancel(prefix + "-cancel", salt)
                self.generate(prefix + "-after-cancel", salt, baseline, "external")
                self.flush(prefix + "-before-restart")
                self.stop(proc, check=True)
                self.backend_idle(prefix + "-before-restart", registered=0)
                proc = self.sglang(tp, True, "-restarted")
                self.generate(prefix + "-after-restart", salt, baseline, "external")
                self.clear(prefix + "-external-clear")
                self.generate(prefix + "-after-clear", salt, baseline, "cold")
                self.flush(prefix + "-final")
                self.backend_idle(prefix + "-final", registered=tp)
                status = requests.get(self.lm_url + "/status", timeout=30)
                status.raise_for_status()
                self.write(f"{prefix}-lmcache-status.json", status.json())
                for name, url in (("sglang", self.url), ("lmcache", self.lm_url)):
                    response = requests.get(url + "/metrics", timeout=30)
                    response.raise_for_status()
                    (self.out / f"{prefix}-{name}-metrics.txt").write_text(
                        response.text
                    )
                self.stop(proc, check=True)
                self.backend_idle(prefix + "-shutdown", registered=0)
                self.results.append({"case": prefix + "-shutdown", "passed": True})
            self.write("verdict.json", {"passed": True, "results": self.results})
        except BaseException as exc:
            self.write(
                "verdict.json",
                {"passed": False, "error": repr(exc), "results": self.results},
            )
            raise
        finally:
            for proc in list(reversed(self.processes)):
                self.stop(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--port", type=int, default=13085)
    parser.add_argument("--lm-port", type=int, default=15585)
    parser.add_argument("--lm-http-port", type=int, default=18085)
    Validation(parser.parse_args()).run()
