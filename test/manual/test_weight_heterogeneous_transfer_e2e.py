# SPDX-License-Identifier: Apache-2.0
"""Reshard real daemon weights and compare every exported tensor with disk.

Requires source_tp * source_pp + 2 * target_tp * target_pp free GPUs and a
local checkpoint. Uses the installed Mooncake package, including its native TE.

Example:
    MOONCAKE_PROTOCOL=rdma MOONCAKE_DEVICE=erdma_0 python \
        test/manual/test_weight_heterogeneous_transfer_e2e.py \
        --model-path /models/Qwen3.5-0.8B --source-tp 1 --target-tp 2
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INFERENCE_PROGRAM = """
import json
import sys
from pathlib import Path
from sglang import Engine

engine = Engine(
    model_path=sys.argv[1], base_gpu_id=int(sys.argv[2]),
    tp_size=int(sys.argv[3]), pp_size=int(sys.argv[4]), ep_size=int(sys.argv[5]),
    port=int(sys.argv[7]), weight_cache_mode="client", dtype="bfloat16",
    skip_tokenizer_init=True, attention_backend="triton", disable_cuda_graph=True,
    mem_fraction_static=0.3, context_length=2048, max_total_tokens=2048,
    max_running_requests=4, random_seed=42, log_level="info",
)
try:
    outputs = engine.generate(
        input_ids=[[1, 2, 3, 4], [1, 5, 9, 10]],
        sampling_params={"temperature": 0, "max_new_tokens": 16},
    )
    Path(sys.argv[6]).write_text(json.dumps([item["output_ids"] for item in outputs]))
finally:
    engine.shutdown()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-tp", type=int, default=1)
    parser.add_argument("--source-pp", type=int, default=1)
    parser.add_argument("--source-ep", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--target-pp", type=int, default=1)
    parser.add_argument("--target-ep", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-inference", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_count = args.source_tp * args.source_pp
    target_count = args.target_tp * args.target_pp
    import torch

    from sglang.srt.platforms import current_platform

    gpu_uuids = [
        current_platform.get_device_uuid(index)
        for index in range(torch.cuda.device_count())
    ]
    if len(gpu_uuids) < source_count + 2 * target_count:
        parser.error("not enough GPUs for source, target, and disk reference")

    processes = []
    logs = []

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with tempfile.TemporaryDirectory(prefix="weight-reshard-e2e-") as sockets:
        env = dict(os.environ)
        env.update(
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE=f"{sockets}/{{device_uuid}}.sock",
            SGLANG_WEIGHT_CACHE_READY_TEMPLATE=f"{sockets}/{{device_uuid}}.ready",
        )
        registry_port = free_port()

        def launch(name, base_gpu, tp, pp, ep, mode=None):
            command = [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                args.model_path,
                "--tp-size",
                str(tp),
                "--pp-size",
                str(pp),
                "--ep-size",
                str(ep),
                "--base-gpu-id",
                str(base_gpu),
                "--dtype",
                "bfloat16",
                "--disable-cuda-graph",
                "--attention-backend",
                "triton",
                "--timeout",
                str(args.timeout),
                "--dist-init-method",
                f"tcp://127.0.0.1:{free_port()}",
            ]
            if mode:
                command += [
                    "--weight-heterogeneous-transfer-mode",
                    mode,
                    "--weight-heterogeneous-transfer-host",
                    "127.0.0.1",
                    "--weight-heterogeneous-transfer-port",
                    str(registry_port),
                ]
            log = (args.output_dir / f"{name}.log").open("w")
            logs.append(log)
            print(f"Starting {name}: {command}", flush=True)
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            deadline = time.monotonic() + args.timeout
            ready = [
                Path(sockets) / f"{uuid}.ready"
                for uuid in gpu_uuids[base_gpu : base_gpu + tp * pp]
            ]
            while not all(path.exists() for path in ready):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited: see {log.name}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} not ready: see {log.name}")
                time.sleep(0.2)
            print(f"{name} ready", flush=True)

        try:
            launch(
                "source", 0, args.source_tp, args.source_pp, args.source_ep, "source"
            )
            launch(
                "target",
                source_count,
                args.target_tp,
                args.target_pp,
                args.target_ep,
                "target",
            )
            launch(
                "reference",
                source_count + target_count,
                args.target_tp,
                args.target_pp,
                args.target_ep,
            )
            comparison = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("test_weight_cache_state_compare.py")),
                    "--left-base-gpu-id",
                    str(source_count),
                    "--right-base-gpu-id",
                    str(source_count + target_count),
                    "--num-ranks",
                    str(target_count),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            (args.output_dir / "comparison.log").write_text(comparison.stdout)
            print(comparison.stdout, flush=True)
            comparison.check_returncode()
            if args.check_inference:
                generated = []
                for name, base_gpu in (
                    ("target", source_count),
                    ("reference", source_count + target_count),
                ):
                    output = args.output_dir / f"{name}-tokens.json"
                    log = (args.output_dir / f"{name}-inference.log").open("w")
                    logs.append(log)
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            INFERENCE_PROGRAM,
                            args.model_path,
                            str(base_gpu),
                            str(args.target_tp),
                            str(args.target_pp),
                            str(args.target_ep),
                            str(output),
                            str(free_port()),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append(process)
                    if process.wait(timeout=args.timeout) != 0:
                        raise RuntimeError(f"{name} inference failed: see {log.name}")
                    if "Loaded model via IPC" not in Path(log.name).read_text():
                        raise AssertionError(f"{name} inference did not load via IPC")
                    generated.append(json.loads(output.read_text()))
                if generated[0] != generated[1]:
                    raise AssertionError("transferred and disk-reference tokens differ")
                print(
                    "PASS: IPC clients generated identical greedy token IDs", flush=True
                )
            (args.output_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "model": args.model_path,
                        "source": [args.source_tp, args.source_pp, args.source_ep],
                        "target": [args.target_tp, args.target_pp, args.target_ep],
                        "protocol": env.get("MOONCAKE_PROTOCOL", "rdma"),
                        "comparison": comparison.stdout,
                        "inference_checked": args.check_inference,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
