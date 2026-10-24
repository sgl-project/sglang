# SPDX-License-Identifier: Apache-2.0
"""Run native-load versus daemon-reshard acceptance tests on eight CUDA GPUs.

All child processes use this interpreter. Checkpoints are local paths; target
daemons only use their config/tokenizer files. Logs and results stay in output-dir.
Every case compares state; five representative layouts also run IPC inference.
"""

import argparse
import gc
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import psutil
import pynvml


@dataclass(frozen=True)
class Layout:
    tp: int
    pp: int = 1
    ep: int = 1
    adp: int = 1

    @property
    def size(self):
        return self.tp * self.pp

    def args(self):
        return [
            "--tp-size",
            str(self.tp),
            "--pp-size",
            str(self.pp),
            "--ep-size",
            str(self.ep),
            "--attn-dp-size",
            str(self.adp),
        ]


def cases():
    result = []
    for model, pairs in (
        ("qwen-small", ((1, 2), (2, 1), (2, 4), (4, 2))),
        # Untied embedding coverage; the small model covers the other TP pairs.
        ("qwen-large", ((1, 2), (2, 1))),
    ):
        for source, target in pairs:
            result.append(
                (
                    f"{model}-tp{source}-to{target}",
                    model,
                    Layout(source),
                    Layout(target),
                    model == "qwen-large" and target == 2,
                )
            )
    for source, target in (
        (Layout(2), Layout(1, pp=2)),
        (Layout(1, pp=2), Layout(2)),
        (Layout(2), Layout(2, adp=2)),
        (Layout(2, adp=2), Layout(2)),
    ):
        suffix = f"{source.tp}p{source.pp}a{source.adp}-to{target.tp}p{target.pp}a{target.adp}"
        result.append(
            (
                f"qwen-small-{suffix}",
                "qwen-small",
                source,
                target,
                target.pp == 2 or target.adp == 2,
            )
        )
    for source, target in (
        (Layout(2), Layout(4)),
        (Layout(4), Layout(2)),
        (Layout(4), Layout(4, ep=2)),
        (Layout(4, ep=2), Layout(4)),
        (Layout(2), Layout(2, pp=2)),
        (Layout(2, pp=2), Layout(2)),
    ):
        suffix = (
            f"{source.tp}p{source.pp}e{source.ep}-to{target.tp}p{target.pp}e{target.ep}"
        )
        result.append(
            (
                f"deepseek-{suffix}",
                "deepseek",
                source,
                target,
                target.ep == 2 or target.pp == 2,
            )
        )
    return result


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def live_group_pids(pgid):
    pids = []
    for process in psutil.process_iter(["pid", "status"]):
        try:
            if (
                process.info["status"] != psutil.STATUS_ZOMBIE
                and os.getpgid(process.pid) == pgid
            ):
                pids.append(process.pid)
        except ProcessLookupError:
            pass
    return pids


def stop(process):
    if process is None:
        return
    # A launcher can exit before its workers. Reap it and wait for the whole
    # process group, escalating only this test's group if workers do not exit.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 15
        while True:
            process.poll()
            remaining = live_group_pids(process.pid)
            if not remaining:
                process.wait(timeout=15)
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    raise TimeoutError(f"test workers did not exit: {remaining}")


def gpu_free_memory(uuids):
    pynvml.nvmlInit()
    try:
        return {
            uuid: pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByUUID(uuid)
            ).free
            for uuid in uuids
        }
    finally:
        pynvml.nvmlShutdown()


def wait_for_gpu_memory(baseline, timeout):
    deadline = time.monotonic() + timeout
    while True:
        current = gpu_free_memory(baseline)
        pending = {
            uuid: baseline[uuid] - free
            for uuid, free in current.items()
            if free < baseline[uuid]
        }
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"source GPU memory was not released (bytes): {pending}")
        time.sleep(0.2)


def start(command, gpus, path, env):
    env = dict(env, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    with path.open("w") as log:
        print("START", path.name, " ".join(command), flush=True)
        return subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_ready(process, uuids, env, timeout):
    from sglang.srt.weight_cache.protocol import _read_ready_pid

    paths = [
        Path(env["SGLANG_WEIGHT_CACHE_READY_TEMPLATE"].format(device_uuid=u))
        for u in uuids
    ]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"daemon exited with status {process.returncode}; inspect its log"
            )
        try:
            pids = [_read_ready_pid(str(p)) for p in paths]
            if all(pid is not None and os.getpgid(pid) == process.pid for pid in pids):
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("daemon readiness timed out")


def generate(model, layout, gpus, directory, env, timeout):
    port = free_port()
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        *layout.args(),
        "--weight-cache-mode",
        "client",
        "--attention-backend",
        "triton",
        "--moe-runner-backend",
        "triton",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.82",
        "--context-length",
        "512",
        "--max-running-requests",
        "2",
        "--max-total-tokens",
        "1024",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    process = start(command, gpus, directory, env)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"engine exited with status {process.returncode}: {directory}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError(f"engine readiness timed out: {directory}")
        responses = []
        for prompt in ("The capital of France is", "Write the next number: 1, 2, 3,"):
            body = json.dumps(
                {
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                }
            ).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                responses.append(json.load(response))
        return responses
    finally:
        stop(process)


def run_case(name, model, source_layout, target_layout, inference, args, uuids):
    from sglang.test.weight_cache_state_compare import compare_rank

    directory = Path(args.output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wa-", dir="/tmp") as sockets:
        env = dict(
            os.environ,
            NCCL_IB_DISABLE="1",
            SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED="0",
            MOONCAKE_PROTOCOL="tcp",
            PYTHONUNBUFFERED="1",
            SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE=sockets + "/{device_uuid}.sock",
            SGLANG_WEIGHT_CACHE_READY_TEMPLATE=sockets + "/{device_uuid}.ready",
        )
        # The state comparator uses physical GPU ordinals in this parent process.
        for key in (
            "SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE",
            "SGLANG_WEIGHT_CACHE_READY_TEMPLATE",
        ):
            os.environ[key] = env[key]
        source_gpus = list(range(source_layout.size))
        target_gpus = list(
            range(source_layout.size, source_layout.size + target_layout.size)
        )
        reference_gpus = [i for i in range(len(uuids)) if i not in target_gpus][
            : target_layout.size
        ]
        if (
            len(target_gpus) + source_layout.size > len(uuids)
            or len(reference_gpus) != target_layout.size
        ):
            raise ValueError(f"not enough GPUs for {name}")
        port = free_port()
        base = [
            sys.executable,
            "-m",
            "sglang.srt.weight_cache.daemon",
            "--model-path",
            model,
            "--attention-backend",
            "triton",
            "--moe-runner-backend",
            "triton",
            "--timeout",
            str(args.timeout),
            "--weight-cache-timeout",
            str(args.timeout),
        ]
        source_memory = gpu_free_memory([uuids[i] for i in source_gpus])
        source = target = reference = None
        try:
            source = start(
                [
                    *base,
                    *source_layout.args(),
                    # Each local group gets its own store; no ephemeral TCP
                    # port can be claimed between selection and worker startup.
                    "--dist-init-method",
                    f"file://{sockets}/source.dist",
                    "--weight-heterogeneous-transfer-mode",
                    "source",
                    "--weight-heterogeneous-transfer-host",
                    "127.0.0.1",
                    "--weight-heterogeneous-transfer-port",
                    str(port),
                ],
                source_gpus,
                directory / "source.log",
                env,
            )
            wait_ready(source, [uuids[i] for i in source_gpus], env, args.timeout)
            target = start(
                [
                    *base,
                    *target_layout.args(),
                    "--dist-init-method",
                    f"file://{sockets}/target.dist",
                    "--weight-heterogeneous-transfer-mode",
                    "target",
                    "--weight-heterogeneous-transfer-host",
                    "127.0.0.1",
                    "--weight-heterogeneous-transfer-port",
                    str(port),
                ],
                target_gpus,
                directory / "target.log",
                env,
            )
            wait_ready(target, [uuids[i] for i in target_gpus], env, args.timeout)
            stop(source)
            source = None
            wait_for_gpu_memory(source_memory, args.timeout)
            reference = start(
                [
                    *base,
                    *target_layout.args(),
                    "--dist-init-method",
                    f"file://{sockets}/reference.dist",
                ],
                reference_gpus,
                directory / "reference.log",
                env,
            )
            wait_ready(reference, [uuids[i] for i in reference_gpus], env, args.timeout)
            entries = nbytes = 0
            for rank, (left, right) in enumerate(zip(target_gpus, reference_gpus)):
                count, size = compare_rank(left, right, rank)
                entries += count
                nbytes += size
            result = {
                "case": name,
                "state_entries": entries,
                "compared_bytes": nbytes,
                "status": "passed",
                "inference": "skipped",
            }
            if inference and not args.skip_inference:
                import torch

                # Comparison copies reserve differently sized allocator blocks
                # on different GPUs across cases. Release those temporary blocks
                # before the engine checks TP free-memory balance.
                gc.collect()
                for gpu in set(target_gpus + reference_gpus):
                    with torch.cuda.device(gpu):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                target_output = generate(
                    model,
                    target_layout,
                    target_gpus,
                    directory / "target-engine.log",
                    env,
                    args.timeout,
                )
                reference_output = generate(
                    model,
                    target_layout,
                    reference_gpus,
                    directory / "reference-engine.log",
                    env,
                    args.timeout,
                )
                for left, right in zip(target_output, reference_output):
                    if left["text"] != right["text"]:
                        raise AssertionError(
                            "greedy generation differs from native-load reference"
                        )
                    lhs = left["meta_info"]["output_token_logprobs"]
                    rhs = right["meta_info"]["output_token_logprobs"]
                    if [p[1] for p in lhs] != [p[1] for p in rhs]:
                        raise AssertionError("generated token IDs differ")
                    torch.testing.assert_close(
                        torch.tensor([p[0] for p in lhs]),
                        torch.tensor([p[0] for p in rhs]),
                        rtol=1e-3,
                        atol=1e-3,
                    )
                (directory / "generation.json").write_text(
                    json.dumps(
                        {"target": target_output, "reference": reference_output},
                        indent=2,
                    )
                )
                result["inference"] = "passed"
            return result
        finally:
            for process in (reference, target, source):
                stop(process)


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qwen-small", "qwen-large", "deepseek"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case", action="append", choices=[c[0] for c in cases()])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip even the representative IPC inference smoke cases.",
    )
    args = parser.parse_args()
    import torch
    from sglang.srt.platforms import current_platform

    uuids = [
        current_platform.get_device_uuid(i) for i in range(torch.cuda.device_count())
    ]
    models = {
        name: getattr(args, name.replace("-", "_"))
        for name in ("qwen-small", "qwen-large", "deepseek")
    }
    results = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = {case[0]: case for case in cases()}
    for name, model, source, target, inference in (
        selected[n] for n in (args.case or selected)
    ):
        print("RUN", name, flush=True)
        try:
            result = run_case(
                name, models[model], source, target, inference, args, uuids
            )
        except Exception as error:
            result = {"case": name, "status": "failed", "error": repr(error)}
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
