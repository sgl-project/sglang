"""Run the NVFP4 PD acceptance on two Slurm segments with four GPUs each."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from bench_glm52_nvfp4_layersplit import PHASES, _read_allocations, compare_results

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def launch(command, output, name):
    write_json(output / f"{name}.command.json", command)
    environment = {
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
        "WITH_NVIDIA_PEERMEM": os.environ.get("WITH_NVIDIA_PEERMEM", "1"),
        "MOONCAKE_PROTOCOL": os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
    }
    for key in ("MC_GID_INDEX", "MC_CUSTOM_TOPO_JSON", "SGLANG_WAIT_PORT_TIMEOUT"):
        if key in os.environ:
            environment[key] = os.environ[key]
    write_json(output / f"{name}.environment.json", environment)
    with (
        (output / f"{name}.stdout.log").open("w", buffering=1) as stdout,
        (output / f"{name}.stderr.log").open("w", buffering=1) as stderr,
    ):
        return subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={**os.environ, **environment},
        )


def check_peer(control):
    for path in control.glob("*.failure.json"):
        raise RuntimeError(f"Peer failed: {path.read_text()}")


def wait_control(path, control, process=None, timeout=10800):
    deadline = time.monotonic() + timeout
    while not path.exists():
        check_peer(control)
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(1)


def wait_health(url, process, control, timeout=3600):
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_peer(control)
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}: {url}")
        try:
            if requests.get(url + "/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready: {url}")


def model_args():
    path = (
        ROOT / "test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit_nvfp4.py"
    )
    spec = importlib.util.spec_from_file_location("glm52_nvfp4_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TestGLM52NVFP4KVLayerSplit


def server_command(args, base, side, phase, address):
    extra = list(getattr(base, f"extra_{side}_args"))
    if side == "prefill" and phase == "baseline":
        extra.remove("--enable-dsa-cache-layer-split")
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--trust-remote-code",
        "--host",
        address,
        "--port",
        "7000",
        "--tp-size",
        "4",
        "--base-gpu-id",
        "0",
        "--nccl-port",
        "7101",
        "--disaggregation-mode",
        side,
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        "8998",
        "--max-total-tokens",
        str(args.max_total_tokens),
        *extra,
    ]


def run_decode(args, base, control, address):
    for phase in PHASES:
        wait_control(control / f"{phase}.start.json", control)
        process = launch(
            server_command(args, base, "decode", phase, address),
            args.output_dir,
            f"{phase}.decode",
        )
        try:
            wait_health(f"http://{address}:7000", process, control)
            write_json(control / f"{phase}.decode.ready.json", {"ready": True})
            wait_control(control / f"{phase}.stop.json", control, process)
        finally:
            stop_process(process)
        write_json(control / f"{phase}.decode.stopped.json", {"stopped": True})


def run_prefill(args, base, control, addresses, report):
    import requests

    if not args.performance_only:
        from sglang.test.kits.eval_accuracy_kit import prepare_full_gsm8k_data
        from sglang.test.run_eval import run_eval

        dataset = prepare_full_gsm8k_data(args.output_dir / "data/full_gsm8k.jsonl")
        report["dataset"] = {
            "sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            "train_demonstrations": 20,
            "test_examples": 1319,
        }
    prefill_url, decode_url = (f"http://{address}:7000" for address in addresses)
    router_url = f"http://{addresses[0]}:7001"
    for phase in PHASES:
        report["stage"] = phase
        report["runs"][phase] = []
        write_json(args.output_dir / "results.json", report)
        write_json(control / f"{phase}.start.json", {"phase": phase})
        prefill = launch(
            server_command(args, base, "prefill", phase, addresses[0]),
            args.output_dir,
            f"{phase}.prefill",
        )
        router = None
        try:
            wait_health(prefill_url, prefill, control)
            wait_control(control / f"{phase}.decode.ready.json", control)
            router = launch(
                [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    prefill_url,
                    "--decode",
                    decode_url,
                    "--host",
                    addresses[0],
                    "--port",
                    "7001",
                ],
                args.output_dir,
                f"{phase}.router",
            )
            wait_health(router_url, router, control, timeout=120)
            if args.performance_only:
                from bench_glm52_nvfp4_layersplit_perf import run_synthetic_phase

                run_synthetic_phase(
                    args, phase, router_url, (prefill_url, decode_url), control, report
                )
            for round_index in range(0 if args.performance_only else 3):
                check_peer(control)
                for url in (prefill_url, decode_url):
                    requests.post(
                        url + "/flush_cache", params={"timeout": 60}, timeout=65
                    ).raise_for_status()
                metrics = run_eval(
                    SimpleNamespace(
                        base_url=router_url,
                        model=args.model_path,
                        eval_name="gsm8k",
                        num_examples=1319,
                        num_threads=200,
                        api="completion",
                        max_tokens=512,
                        num_shots=20,
                        temperature=0,
                        gsm8k_data_path=dataset,
                    )
                )
                report["runs"][phase].append(metrics)
                stem = "gsm8k_" + args.model_path.replace("/", "_")
                for suffix in ("html", "json"):
                    shutil.copy2(
                        Path("/tmp") / f"{stem}.{suffix}",
                        args.output_dir / f"{phase}.run{round_index + 1}.{suffix}",
                    )
                write_json(args.output_dir / "results.json", report)
                if metrics["score"] < 0.935:
                    raise RuntimeError(f"{phase} accuracy below 93.5%: {metrics}")
        finally:
            stop_process(router)
            stop_process(prefill)
            write_json(control / f"{phase}.stop.json", {"stop": True})
        wait_control(control / f"{phase}.decode.stopped.json", control)
        report["allocations"][phase] = _read_allocations(args.output_dir, phase)
        write_json(args.output_dir / "results.json", report)
    if args.performance_only:
        from bench_glm52_nvfp4_layersplit_perf import summarize_performance

        report["comparison"] = summarize_performance(
            report["synthetic_runs"], report["allocations"]
        )
        report["status"] = "completed"
    else:
        report["comparison"] = compare_results(report["runs"], report["allocations"])
        report["status"] = "passed" if report["comparison"]["passed"] else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-total-tokens", type=int, default=131072)
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run fixed synthetic requests without dataset accuracy evaluation",
    )
    args = parser.parse_args()
    if args.max_total_tokens <= 0 or args.max_total_tokens % 64:
        parser.error("--max-total-tokens must be a positive multiple of 64")
    if os.environ.get("SGLANG_ACCEPTANCE_NODES"):
        nodes = os.environ["SGLANG_ACCEPTANCE_NODES"].split(",")
    else:
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            text=True,
        ).splitlines()
    if len(nodes) != 2 or int(os.environ["SLURM_NTASKS"]) != 2:
        parser.error("Run one task on each of two four-GPU Slurm segments")
    rank = int(os.environ["SLURM_PROCID"])
    side = "prefill" if rank == 0 else "decode"
    control = args.output_dir / "control"
    control.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "stage": "preflight",
        "model": args.model_path,
        "max_total_tokens": args.max_total_tokens,
        "topology": {"nodes": nodes, "gpus_per_node": 4, "prefill": nodes[0]},
        "runs": {},
        "allocations": {},
        "smoke_test_run": False,
        "accuracy_evaluated": not args.performance_only,
    }
    report["source_sha256"] = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in (
            "python/sglang/srt/arg_groups/overrides.py",
            "python/sglang/srt/mem_cache/memory_pool.py",
            "python/sglang/srt/mem_cache/dsa_cache_layer_split.py",
            "python/sglang/srt/model_executor/pool_configurator.py",
            "python/sglang/test/kits/eval_accuracy_kit.py",
            "scripts/bench_glm52_nvfp4_layersplit.py",
            "scripts/bench_glm52_nvfp4_layersplit_slurm.py",
            "scripts/bench_glm52_nvfp4_layersplit_perf.py",
            "test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit_nvfp4.py",
        )
    }
    if os.environ.get("GLM52_CAMPAIGN_ROOT"):
        deployment = Path(os.environ["GLM52_CAMPAIGN_ROOT"])
        report["deployment"] = {}
        for name in (
            "source-freeze.current.json",
            "model.identity.json",
            "prepare.versions.json",
            "native.sha256.json",
        ):
            path = deployment / name
            if path.exists():
                report["deployment"][name] = json.loads(path.read_text())
        report["deployment"]["container_sha256"] = (
            (deployment / "container.sqsh.sha256").read_text().split()[0]
        )
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import torch
        from sgl_kernel import flashmla_ops  # noqa: F401

        if torch.cuda.device_count() != 4 or any(
            "GB300" not in torch.cuda.get_device_name(i) for i in range(4)
        ):
            raise RuntimeError("Each segment must expose exactly four GB300 GPUs")
        base = model_args()
        configured_command = server_command(args, base, side, PHASES[0], "127.0.0.1")
        kv_dtype = configured_command[
            configured_command.index("--kv-cache-dtype") + 1
        ]
        if kv_dtype == "nvfp4":
            from sgl_kernel import flashmla_nvfp4_ops  # noqa: F401

            if not hasattr(torch.ops.sgl_kernel, "sparse_decode_fwd_nvfp4"):
                raise RuntimeError(
                    "Rebuild flashmla_nvfp4_ops with sparse_decode_fwd_nvfp4"
                )
        write_json(
            args.output_dir / f"{side}.versions.json",
            {
                "host": socket.gethostname(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(4)],
                "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            },
        )
        addresses = [socket.gethostbyname(node) for node in nodes]
        no_proxy = ",".join(
            [
                os.environ.get("NO_PROXY", ""),
                "127.0.0.1",
                "localhost",
                *nodes,
                *addresses,
            ]
        )
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy
        if rank == 1:
            run_decode(args, base, control, addresses[1])
        else:
            run_prefill(args, base, control, addresses, report)
    except BaseException as error:
        write_json(control / f"{side}.failure.json", {"error": repr(error)})
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        if rank == 0:
            write_json(args.output_dir / "results.json", report)
    if rank == 0 and report["status"] not in ("passed", "completed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
