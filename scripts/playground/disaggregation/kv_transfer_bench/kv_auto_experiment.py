#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


SRC_IPS = {
    0: "fd03:4514:80:6240::1",
    1: "fd03:4514:80:6241::1",
    2: "fd03:4514:80:6242::1",
    3: "fd03:4514:80:6243::1",
}
TGT_IPS = {
    0: "fd03:4514:80:5f00::1",
    1: "fd03:4514:80:5f01::1",
    2: "fd03:4514:80:5f02::1",
    3: "fd03:4514:80:5f03::1",
}

DENSE_SIZES_1 = (
    "1MB,2MB,4MB,8MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,"
    "192MB,256MB,384MB,512MB,768MB,1GB,1.25GB,1.5GB,1.75GB,2GB"
)
DENSE_SIZES_2 = (
    "512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,"
    "96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB"
)
DENSE_SIZES_4 = (
    "256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,"
    "48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB"
)


Endpoint = namedtuple("Endpoint", "lane gpu_id src_host tgt_host ib_device")
CompetitionCase = namedtuple(
    "CompetitionCase",
    "run flow_sizes protocol ib_device lane_cap_gbps mode",
)

PHYSICAL_LANE_GBPS = 200.0


class MatrixRun:
    def __init__(
        self,
        run: str,
        lanes: Iterable[Endpoint],
        shards: int,
        max_bytes: str,
        sizes: str,
        protocol: str,
        lane_cap_gbps: float,
        bg_rate_gbps: float,
        fg_rate_gbps: Optional[float],
        capfill_lanes: Iterable[Endpoint] = (),
        capfill_rate_gbps: float = 0.0,
    ):
        self.run = run
        self.lanes = tuple(lanes)
        self.shards = shards
        self.max_bytes = max_bytes
        self.sizes = sizes
        self.protocol = protocol
        self.lane_cap_gbps = lane_cap_gbps
        self.bg_rate_gbps = bg_rate_gbps
        self.fg_rate_gbps = fg_rate_gbps
        self.capfill_lanes = tuple(capfill_lanes)
        self.capfill_rate_gbps = capfill_rate_gbps


def q(value: object) -> str:
    return shlex.quote(str(value))


def _time_ns() -> int:
    if hasattr(time, "time_ns"):
        return time.time_ns()
    return int(time.time() * 1_000_000_000)


def parse_size(raw: str) -> int:
    raw = raw.strip().lower()
    units = [
        ("gib", 1024**3),
        ("gb", 1024**3),
        ("g", 1024**3),
        ("mib", 1024**2),
        ("mb", 1024**2),
        ("m", 1024**2),
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
        ("b", 1),
    ]
    for suffix, multiplier in units:
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * multiplier)
    return int(float(raw))


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.6f}"


def percentile(values: List[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.local_children: List[subprocess.Popen] = []

    def local_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("SGLANG_IMAGE", self.args.image)
        return env

    def run_local(self, cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
        print("+ " + " ".join(q(x) for x in cmd), flush=True)
        if self.args.dry_run:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(cmd, check=check, universal_newlines=True, env=self.local_env())

    def run_local_capture(self, cmd: List[str], *, check: bool = True) -> str:
        print("+ " + " ".join(q(x) for x in cmd), flush=True)
        if self.args.dry_run:
            return ""
        result = subprocess.run(
            cmd,
            check=check,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.local_env(),
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.stdout

    def remote_cmd(self) -> List[str]:
        cmd = ["ssh"]
        if self.args.ssh_key:
            cmd.extend(["-i", os.path.expanduser(self.args.ssh_key)])
        cmd.extend(["-o", "BatchMode=yes", self.args.target_ssh_host, "bash", "-s"])
        return cmd

    def run_remote(self, script: str, *, check: bool = True) -> subprocess.CompletedProcess:
        print("+ ssh " + self.args.target_ssh_host + " bash -s <<'REMOTE'", flush=True)
        print(script.rstrip(), flush=True)
        print("REMOTE", flush=True)
        if self.args.dry_run:
            return subprocess.CompletedProcess(self.remote_cmd(), 0, "", "")
        return subprocess.run(self.remote_cmd(), input=script, universal_newlines=True, check=check)

    def run_remote_capture(self, script: str, *, check: bool = True) -> str:
        print("+ ssh " + self.args.target_ssh_host + " bash -s <<'REMOTE'", flush=True)
        print(script.rstrip(), flush=True)
        print("REMOTE", flush=True)
        if self.args.dry_run:
            return ""
        result = subprocess.run(
            self.remote_cmd(),
            input=script,
            universal_newlines=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.stdout

    def docker_cmd(
        self,
        *,
        name: str,
        env: Dict[str, str],
        out_dir: Path,
        inner_cmd: str,
    ) -> List[str]:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc",
            "host",
            "--privileged",
            "--ulimit",
            "memlock=-1:-1",
        ]
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(
            [
                "-v",
                f"{self.args.bench_dir}:/workspace/kv_transfer_bench:ro",
                "-v",
                f"{out_dir}:/tmp/kv-transfer-bench",
            ]
        )
        if Path("/dev/infiniband").exists():
            cmd.extend(["-v", "/dev/infiniband:/dev/infiniband"])
        cmd.extend(
            [
                self.args.image,
                "bash",
                "-lc",
                "cd /workspace/kv_transfer_bench && " + inner_cmd,
            ]
        )
        return cmd

    def start_local_docker(
        self,
        *,
        name: str,
        env: Dict[str, str],
        out_dir: Path,
        inner_cmd: str,
        log_path: Path,
    ) -> subprocess.Popen:
        cmd = self.docker_cmd(name=name, env=env, out_dir=out_dir, inner_cmd=inner_cmd)
        print("+ " + " ".join(q(x) for x in cmd) + f" > {log_path} 2>&1", flush=True)
        if self.args.dry_run:
            proc = subprocess.Popen(["true"])
            self.local_children.append(proc)
            return proc
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.local_env())
        self.local_children.append(proc)
        return proc

    def cleanup_run(self, run: str, out_dir: Path) -> None:
        local_script = (
            f"docker ps -a --format '{{{{.Names}}}}' | "
            f"grep '^kv_{run}_' | xargs -r docker rm -f\n"
        )
        self.run_local(["bash", "-lc", local_script], check=False)
        remote_script = f"""
set +e
docker ps -a --format '{{{{.Names}}}}' | grep '^kv_{run}_' | xargs -r docker rm -f
if [ -s {q(str(out_dir / "raw" / "rdma-monitor.pid"))} ]; then
  kill "$(cat {q(str(out_dir / "raw" / "rdma-monitor.pid"))})" 2>/dev/null || true
fi
"""
        self.run_remote(remote_script, check=False)

    def remote_target_script(self, run: Union[MatrixRun, CompetitionCase], targets: List[Endpoint], out_dir: Path, max_bytes: str) -> str:
        lines = [
            "set -euo pipefail",
            f"mkdir -p {q(str(out_dir / 'raw'))}",
        ]
        target_specs = []
        if isinstance(run, MatrixRun):
            for endpoint in targets:
                target_specs.append((endpoint, f"target-bond{endpoint.lane}"))
                if run.bg_rate_gbps > 0:
                    target_specs.append((endpoint, f"target-bg-bond{endpoint.lane}"))
            for endpoint in run.capfill_lanes:
                target_specs.append((endpoint, f"target-cap-bond{endpoint.lane}"))
        else:
            for endpoint in targets:
                target_specs.append((endpoint, f"target-flow{endpoint.lane}"))

        for endpoint, suffix in target_specs:
            name = f"kv_{run.run}_{suffix.replace('-', '_')}"
            info_file = f"/tmp/kv-transfer-bench/{suffix}.json"
            log_file = out_dir / "raw" / f"{suffix}.log"
            env = [
                "-e MOONCAKE_PROTOCOL=" + q(run.protocol),
                "-e IB_DEVICE=" + q(endpoint.ib_device),
            ]
            if run.protocol == "rdma":
                env.extend(["-e MC_USE_IPV6=1", "-e MC_GID_INDEX=3"])
            if run.protocol == "tcp":
                env.append("-e MC_FORCE_TCP=1")
            lines.append(
                "nohup docker run --rm "
                f"--name {q(name)} --gpus all --network host --ipc host --privileged "
                "--ulimit memlock=-1:-1 "
                + " ".join(env)
                + f" -v {q(self.args.bench_dir)}:/workspace/kv_transfer_bench:ro"
                + f" -v {q(str(out_dir / 'raw'))}:/tmp/kv-transfer-bench"
                + " -v /dev/infiniband:/dev/infiniband"
                + f" {q(self.args.image)} bash -lc "
                + q(
                    "cd /workspace/kv_transfer_bench && "
                    f"python3 kv_transfer_latency.py --role target "
                    f"--host {endpoint.tgt_host} --gpu-id {endpoint.gpu_id} "
                    f"--ib-device {endpoint.ib_device} --protocol {run.protocol} "
                    f"--max-bytes {max_bytes} --target-info-file {info_file}"
                )
                + f" > {q(str(log_file))} 2>&1 &"
            )

        for _, suffix in target_specs:
            log_file = out_dir / "raw" / f"{suffix}.log"
            lines.append(f"until grep -q 'target_ready=true' {q(str(log_file))}; do sleep 1; done")
        lines.append(f"grep '^TARGET_INFO_JSON=' {q(str(out_dir / 'raw'))}/target*.log")
        return "\n".join(lines) + "\n"

    def start_remote_monitor(self, run: Union[MatrixRun, CompetitionCase], out_dir: Path, endpoints: Iterable[Endpoint]) -> None:
        devices = " ".join(monitor_ib_devices(endpoints)) if run.protocol == "rdma" else ""
        if not devices:
            return
        raw_dir = out_dir / "raw"
        monitor_csv = raw_dir / "rdma-rcv-monitor.csv"
        monitor_err = raw_dir / "rdma-rcv-monitor.err"
        monitor_pid = raw_dir / "rdma-monitor.pid"
        monitor_inner = f"""
declare -A last
for dev in {devices}; do
  last[$dev]=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
done
tlast=$(date +%s)
while true; do
  sleep 2
  now=$(date +%s)
  ts=$(date -Iseconds)
  dt=$((now - tlast))
  for dev in {devices}; do
    cur=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
    prev=${{last[$dev]}}
    gbps=$(awk -v cur="$cur" -v prev="$prev" -v dt="$dt" 'BEGIN{{printf "%.3f",(cur-prev)*32/dt/1e9}}')
    echo "$ts,$dev,$gbps"
    last[$dev]=$cur
  done
  tlast=$now
done
"""
        script = f"""
set -euo pipefail
mkdir -p {q(str(raw_dir))}
echo "ts,dev,rcv_Gbps" > {q(str(monitor_csv))}
nohup bash -lc {q(monitor_inner)} </dev/null >> {q(str(monitor_csv))} 2>> {q(str(monitor_err))} &
echo $! > {q(str(monitor_pid))}
"""
        self.run_remote(script)

    def fetch_target_jsons(self, out_dir: Path, pattern: str) -> Dict[int, str]:
        script = f"set -euo pipefail\ngrep '^TARGET_INFO_JSON=' {q(str(out_dir / 'raw'))}/{pattern}\n"
        output = self.run_remote_capture(script)
        result: Dict[int, str] = {}
        for line in output.splitlines():
            if "TARGET_INFO_JSON=" not in line:
                continue
            path, raw = line.split("TARGET_INFO_JSON=", 1)
            json_data = json.loads(raw)
            flow_match = re.search(r"target-flow(\d+)\.log", path)
            key = int(flow_match.group(1)) if flow_match else int(json_data["gpu_id"])
            result[key] = raw
        return result

    def fake_target_json(self, endpoint: Endpoint, max_bytes: str) -> str:
        return json.dumps(
            {
                "bytes": parse_size(max_bytes),
                "gpu_id": endpoint.gpu_id,
                "host": endpoint.tgt_host,
                "ib_device": endpoint.ib_device,
                "protocol": "dry-run",
                "ptr": 0,
                "session_id": f"[{endpoint.tgt_host}]:0",
            }
        )

    def run_matrix(self, spec: MatrixRun) -> None:
        out_dir = Path(self.args.out_root) / spec.run
        raw = out_dir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        print(f"\n=== RUN {spec.run} ===", flush=True)
        self.cleanup_run(spec.run, out_dir)
        self.run_remote(self.remote_target_script(spec, list(spec.lanes), out_dir, spec.max_bytes))
        monitor_endpoints = list(spec.lanes) + list(spec.capfill_lanes)
        self.start_remote_monitor(spec, out_dir, monitor_endpoints)
        if self.args.dry_run:
            fg_jsons = {endpoint.gpu_id: self.fake_target_json(endpoint, spec.max_bytes) for endpoint in spec.lanes}
            bg_jsons = {}
            if spec.bg_rate_gbps > 0:
                bg_jsons = {
                    endpoint.gpu_id: self.fake_target_json(endpoint, spec.max_bytes)
                    for endpoint in spec.lanes
                }
            capfill_jsons = {
                endpoint.gpu_id: self.fake_target_json(endpoint, spec.max_bytes)
                for endpoint in spec.capfill_lanes
            }
        else:
            fg_jsons = self.fetch_target_jsons(out_dir, "target-bond*.log")
            bg_jsons = {}
            if spec.bg_rate_gbps > 0:
                bg_jsons = self.fetch_target_jsons(out_dir, "target-bg-bond*.log")
            capfill_jsons = {}
            if spec.capfill_lanes:
                capfill_jsons = self.fetch_target_jsons(out_dir, "target-cap-bond*.log")

        capfill_procs = []
        if spec.capfill_rate_gbps > 0:
            for endpoint in spec.capfill_lanes:
                target_json = capfill_jsons[endpoint.gpu_id]
                env = self.container_env(spec.protocol, endpoint.ib_device, target_json)
                env["FLOW_ID"] = f"capfill-bond{endpoint.lane}"
                inner = (
                    f"python3 kv_transfer_latency.py --role initiator --host {endpoint.src_host} "
                    f"--gpu-id {endpoint.gpu_id} --ib-device {endpoint.ib_device} "
                    f"--protocol {spec.protocol} --sizes {spec.max_bytes} "
                    f"--background-bytes {spec.max_bytes} "
                    f"--background-duration-seconds {self.args.bg_duration_seconds} "
                    f"--rate-limit-gbps {spec.capfill_rate_gbps} --chunk-size {self.args.chunk_size} "
                    f"--flow-id capfill-bond{endpoint.lane} "
                    f"--summary-csv /tmp/kv-transfer-bench/capfill-bond{endpoint.lane}-summary.csv "
                    f"--samples-jsonl /tmp/kv-transfer-bench/capfill-bond{endpoint.lane}-samples.jsonl"
                )
                capfill_procs.append(
                    self.start_local_docker(
                        name=f"kv_{spec.run}_capfill_init_bond{endpoint.lane}",
                        env=env,
                        out_dir=raw,
                        inner_cmd=inner,
                        log_path=raw / f"capfill-init-bond{endpoint.lane}.log",
                    )
                )

        bg_procs = []
        if spec.bg_rate_gbps > 0:
            for endpoint in spec.lanes:
                target_json = bg_jsons[endpoint.gpu_id]
                env = self.container_env(spec.protocol, endpoint.ib_device, target_json)
                env["FLOW_ID"] = f"bg-bond{endpoint.lane}"
                inner = (
                    f"python3 kv_transfer_latency.py --role initiator --host {endpoint.src_host} "
                    f"--gpu-id {endpoint.gpu_id} --ib-device {endpoint.ib_device} "
                    f"--protocol {spec.protocol} --sizes {spec.max_bytes} "
                    f"--background-bytes {spec.max_bytes} "
                    f"--background-duration-seconds {self.args.bg_duration_seconds} "
                    f"--rate-limit-gbps {spec.bg_rate_gbps} --chunk-size {self.args.chunk_size} "
                    f"--flow-id bg-bond{endpoint.lane} "
                    f"--summary-csv /tmp/kv-transfer-bench/bgmoon-bond{endpoint.lane}-summary.csv "
                    f"--samples-jsonl /tmp/kv-transfer-bench/bgmoon-bond{endpoint.lane}-samples.jsonl"
                )
                bg_procs.append(
                    self.start_local_docker(
                        name=f"kv_{spec.run}_bgmoon_init_bond{endpoint.lane}",
                        env=env,
                        out_dir=raw,
                        inner_cmd=inner,
                        log_path=raw / f"bgmoon-init-bond{endpoint.lane}.log",
                    )
                )

        if not self.args.dry_run and (bg_procs or capfill_procs):
            time.sleep(self.args.bg_warmup_seconds)

        fg_procs = []
        for endpoint in spec.lanes:
            target_json = fg_jsons[endpoint.gpu_id]
            env = self.container_env(spec.protocol, endpoint.ib_device, target_json)
            env["FLOW_ID"] = f"fg-bond{endpoint.lane}"
            inner = (
                f"python3 kv_transfer_latency.py --role initiator --host {endpoint.src_host} "
                f"--gpu-id {endpoint.gpu_id} --ib-device {endpoint.ib_device} "
                f"--protocol {spec.protocol} --sizes {q(spec.sizes)} "
                f"--warmup {self.args.warmup} --repeat {self.args.repeat} "
                f"{rate_limit_arg(spec.fg_rate_gbps)}--chunk-size {self.args.chunk_size} "
                f"--flow-id fg-bond{endpoint.lane} "
                f"--summary-csv /tmp/kv-transfer-bench/shard-bond{endpoint.lane}-dense-summary.csv "
                f"--samples-jsonl /tmp/kv-transfer-bench/shard-bond{endpoint.lane}-dense-samples.jsonl"
            )
            fg_procs.append(
                self.start_local_docker(
                    name=f"kv_{spec.run}_init_bond{endpoint.lane}_dense",
                    env=env,
                    out_dir=raw,
                    inner_cmd=inner,
                    log_path=raw / f"init-bond{endpoint.lane}-dense.log",
                )
            )

        self.wait_all(fg_procs, "foreground")
        self.aggregate_matrix(out_dir, spec.shards)
        self.cleanup_run(spec.run, out_dir)
        self.wait_terminate(bg_procs + capfill_procs)

    def container_env(self, protocol: str, ib_device: str, target_json: str) -> Dict[str, str]:
        env = {
            "MOONCAKE_PROTOCOL": protocol,
            "IB_DEVICE": ib_device,
            "TARGET_INFO_JSON": target_json,
        }
        if protocol == "rdma":
            env["MC_USE_IPV6"] = "1"
            env["MC_GID_INDEX"] = "3"
        if protocol == "tcp":
            env["MC_FORCE_TCP"] = "1"
        return env

    def wait_all(self, procs: List[subprocess.Popen], label: str) -> None:
        if self.args.dry_run:
            return
        errors = []
        for proc in procs:
            rc = proc.wait()
            if rc != 0:
                errors.append(rc)
        if errors:
            raise RuntimeError(f"{label} docker process failed: {errors}")

    def wait_terminate(self, procs: List[subprocess.Popen]) -> None:
        if self.args.dry_run:
            return
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def aggregate_matrix(self, out_dir: Path, shards: int) -> None:
        raw = out_dir / "raw"
        groups: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        errors: Dict[int, int] = defaultdict(int)
        for path in sorted(raw.glob("shard-bond*-dense-samples.jsonl")):
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    logical_bytes = int(row["bytes"]) * shards
                    key = (logical_bytes, int(row["iteration"]))
                    if int(row.get("ret", 0)) == 0:
                        groups[key].append(float(row["latency_ms"]))
                    else:
                        errors[logical_bytes] += 1

        by_size: Dict[int, List[float]] = defaultdict(list)
        for (logical_bytes, _), vals in groups.items():
            if len(vals) == shards:
                by_size[logical_bytes].append(max(vals))
            else:
                errors[logical_bytes] += 1

        rows = []
        for logical_bytes in sorted(by_size):
            vals = by_size[logical_bytes]
            mean = sum(vals) / len(vals)
            p50 = percentile(vals, 0.50)
            rows.append(
                {
                    "bytes": logical_bytes,
                    "human_bytes": human_bytes(logical_bytes),
                    "shard_count": shards,
                    "repeat_count": len(vals),
                    "error_count": errors[logical_bytes],
                    "latency_ms_mean": mean,
                    "latency_ms_p50": p50,
                    "latency_ms_p90": percentile(vals, 0.90),
                    "latency_ms_p99": percentile(vals, 0.99),
                    "latency_ms_min": min(vals),
                    "latency_ms_max": max(vals),
                    "bandwidth_GBps_p50": (logical_bytes / 1024**3) / (p50 / 1000),
                    "bandwidth_GBps_mean": (logical_bytes / 1024**3) / (mean / 1000),
                }
            )
        if not rows and not self.args.dry_run:
            raise RuntimeError(f"no complete shard groups found under {raw}")
        write_csv(out_dir / "aggregated-summary.csv", rows)
        print(f"aggregated {len(rows)} rows -> {out_dir / 'aggregated-summary.csv'}")

    def run_competition(self, case: CompetitionCase) -> None:
        out_dir = Path(self.args.out_root) / "competition" / case.run
        raw = out_dir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        print(f"\n=== COMPETITION {case.run} ===", flush=True)
        self.cleanup_run(case.run, out_dir)
        targets = [
            Endpoint(i, 0, SRC_IPS[0], TGT_IPS[0], case.ib_device)
            for i, _ in enumerate(case.flow_sizes)
        ]
        max_bytes = human_bytes(max(parse_size(size) for size in case.flow_sizes))
        self.run_remote(self.remote_target_script(case, targets, out_dir, max_bytes))
        self.start_remote_monitor(case, out_dir, [Endpoint(0, 0, SRC_IPS[0], TGT_IPS[0], case.ib_device)])
        if self.args.dry_run:
            target_jsons = {endpoint.lane: self.fake_target_json(endpoint, max_bytes) for endpoint in targets}
        else:
            target_jsons = self.fetch_target_jsons(out_dir, "target-flow*.log")
        start_at = _time_ns() + int(self.args.competition_start_delay_seconds * 1_000_000_000)
        procs = []
        for flow_index, size in enumerate(case.flow_sizes):
            rate = None
            if case.mode == "fair":
                rate = case.lane_cap_gbps / len(case.flow_sizes)
            target_json = target_jsons[flow_index]
            env = self.container_env(case.protocol, case.ib_device, target_json)
            env["FLOW_ID"] = f"flow{flow_index}"
            inner = (
                f"python3 kv_transfer_latency.py --role initiator --host {SRC_IPS[0]} "
                f"--gpu-id 0 --ib-device {case.ib_device} --protocol {case.protocol} "
                f"--sizes {size} --warmup 0 --repeat 1 --chunk-size {self.args.chunk_size} "
                f"--flow-id flow{flow_index} --start-at-unix-ns {start_at} "
                f"--summary-csv /tmp/kv-transfer-bench/competition-flow{flow_index}-summary.csv "
                f"--samples-jsonl /tmp/kv-transfer-bench/competition-flow{flow_index}-samples.jsonl"
            )
            if rate is not None:
                inner += f" --rate-limit-gbps {rate}"
            procs.append(
                self.start_local_docker(
                    name=f"kv_{case.run}_flow{flow_index}",
                    env=env,
                    out_dir=raw,
                    inner_cmd=inner,
                    log_path=raw / f"competition-flow{flow_index}.log",
                )
            )
        self.wait_all(procs, "competition")
        self.aggregate_competition(out_dir)
        self.cleanup_run(case.run, out_dir)

    def aggregate_competition(self, out_dir: Path) -> None:
        rows = []
        events_path = out_dir / "competition-events.jsonl"
        with events_path.open("w") as events:
            for path in sorted((out_dir / "raw").glob("competition-flow*-samples.jsonl")):
                with path.open() as f:
                    for line in f:
                        row = json.loads(line)
                        events.write(json.dumps(row, sort_keys=True) + "\n")
                        latency_ms = float(row["latency_ms"])
                        bytes_value = int(row["bytes"])
                        rows.append(
                            {
                                "flow_id": row.get("flow_id", ""),
                                "bytes": bytes_value,
                                "human_bytes": human_bytes(bytes_value),
                                "ret": row.get("ret"),
                                "start_time_unix_ns": row.get("start_time_unix_ns"),
                                "end_time_unix_ns": row.get("end_time_unix_ns"),
                                "latency_ms": latency_ms,
                                "bandwidth_GBps": (bytes_value / 1024**3) / (latency_ms / 1000),
                                "rate_limit_gbps": row.get("rate_limit_gbps"),
                            }
                        )
        write_csv(out_dir / "competition-summary.csv", rows)
        print(f"competition events -> {events_path}")
        print(f"competition summary -> {out_dir / 'competition-summary.csv'}")


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024
    raise AssertionError("unreachable")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bond_endpoint(lane: int, ib_device: Optional[str] = None) -> Endpoint:
    return Endpoint(
        lane=lane,
        gpu_id=lane,
        src_host=SRC_IPS[lane],
        tgt_host=TGT_IPS[lane],
        ib_device=ib_device or f"mlx5_bond_{lane}",
    )


def one_lane_endpoint(src_host: str, tgt_host: str, ib_device: str) -> Endpoint:
    return Endpoint(lane=0, gpu_id=0, src_host=src_host, tgt_host=tgt_host, ib_device=ib_device)


def rate_limit_arg(rate_gbps: Optional[float]) -> str:
    if rate_gbps is None:
        return ""
    return f"--rate-limit-gbps {rate_gbps} "


def monitor_ib_devices(endpoints: Iterable[Endpoint]) -> List[str]:
    devices: List[str] = []
    seen = set()
    for endpoint in endpoints:
        for raw_device in endpoint.ib_device.split(","):
            device = raw_device.strip()
            if not device or device in seen:
                continue
            devices.append(device)
            seen.add(device)
    return devices


def multi_hca_endpoint(lanes: Iterable[int]) -> Endpoint:
    lane_list = tuple(lanes)
    ib_device = ",".join(f"mlx5_bond_{lane}" for lane in lane_list)
    return Endpoint(
        lane=0,
        gpu_id=0,
        src_host=SRC_IPS[lane_list[0]],
        tgt_host=TGT_IPS[lane_list[0]],
        ib_device=ib_device,
    )


def capfill_rate_for_lane_cap(lane_cap_gbps: float) -> float:
    return max(0.0, PHYSICAL_LANE_GBPS - lane_cap_gbps)


def capfill_endpoints(lanes: Iterable[int], lane_cap_gbps: float) -> Tuple[Endpoint, ...]:
    if capfill_rate_for_lane_cap(lane_cap_gbps) <= 0:
        return ()
    return tuple(bond_endpoint(lane) for lane in lanes)


def multi_hca_run(
    *,
    prefix: str,
    lanes: Iterable[int],
    lane_cap_gbps: float,
    bg_percent: int,
    suffix: str = "multi_hca_portcap_moonbg",
) -> MatrixRun:
    lane_list = tuple(lanes)
    total_cap_gbps = lane_cap_gbps * len(lane_list)
    bg_rate = total_cap_gbps * bg_percent / 100
    fg_rate = total_cap_gbps - bg_rate
    capfill_rate = capfill_rate_for_lane_cap(lane_cap_gbps)
    return MatrixRun(
        run=f"{prefix}_bg{bg_percent}_{suffix}",
        lanes=(multi_hca_endpoint(lane_list),),
        shards=1,
        max_bytes="2GB",
        sizes=DENSE_SIZES_1,
        protocol="rdma",
        lane_cap_gbps=lane_cap_gbps,
        bg_rate_gbps=bg_rate,
        fg_rate_gbps=fg_rate,
        capfill_lanes=capfill_endpoints(lane_list, lane_cap_gbps),
        capfill_rate_gbps=capfill_rate,
    )


def multi_hca_bgcap_only_run(
    *,
    prefix: str,
    lanes: Iterable[int],
    total_cap_gbps: float,
    bg_percent: int,
) -> MatrixRun:
    lane_list = tuple(lanes)
    bg_rate = total_cap_gbps * bg_percent / 100
    return MatrixRun(
        run=f"{prefix}_bg{bg_percent}_multi_hca_bgcap_only",
        lanes=(multi_hca_endpoint(lane_list),),
        shards=1,
        max_bytes="2GB",
        sizes=DENSE_SIZES_1,
        protocol="rdma",
        lane_cap_gbps=total_cap_gbps / len(lane_list),
        bg_rate_gbps=bg_rate,
        fg_rate_gbps=None,
        capfill_lanes=(),
        capfill_rate_gbps=0.0,
    )


def split_4x100_run(bg_percent: int) -> MatrixRun:
    bg_rate = 100 * bg_percent / 100
    fg_rate = 100 - bg_rate
    return MatrixRun(
        run=f"400_4x100_bg{bg_percent}_cap100_moonbg_split",
        lanes=tuple(bond_endpoint(lane) for lane in (0, 1, 2, 3)),
        shards=4,
        max_bytes="512MB",
        sizes=DENSE_SIZES_4,
        protocol="rdma",
        lane_cap_gbps=100,
        bg_rate_gbps=bg_rate,
        fg_rate_gbps=fg_rate,
    )


def matrix_runs(args: Optional[argparse.Namespace] = None) -> List[MatrixRun]:
    runs: List[MatrixRun] = []
    for bg in (1, 10, 50, 90):
        bg_rate = 200 * bg / 100
        fg_rate = 200 - bg_rate
        runs.append(
            MatrixRun(
                run=f"800_4x200_bg{bg}_cap200_moonbg_fixed",
                lanes=tuple(bond_endpoint(lane) for lane in (0, 1, 2, 3)),
                shards=4,
                max_bytes="512MB",
                sizes=DENSE_SIZES_4,
                protocol="rdma",
                lane_cap_gbps=200,
                bg_rate_gbps=bg_rate,
                fg_rate_gbps=fg_rate,
            )
        )

    single_nic_endpoint = one_lane_endpoint(
        getattr(args, "single_nic_src_host", SRC_IPS[0]),
        getattr(args, "single_nic_tgt_host", TGT_IPS[0]),
        getattr(args, "single_nic_ib_device", "mlx5_0"),
    )
    head_rdma_endpoint = one_lane_endpoint(
        getattr(args, "head_rdma_src_host", SRC_IPS[0]),
        getattr(args, "head_rdma_tgt_host", TGT_IPS[0]),
        getattr(args, "head_rdma_ib_device", "mlx5_bond_0"),
    )
    head_tcp_endpoint = one_lane_endpoint(
        getattr(args, "head_tcp_src_host", "192.168.0.39"),
        getattr(args, "head_tcp_tgt_host", "192.168.0.41"),
        getattr(args, "head_tcp_ib_device", "mlx5_bond_0"),
    )
    one_lane_variants = [
        ("tail_single_nic", "rdma", single_nic_endpoint),
        ("head_rdma", "rdma", head_rdma_endpoint),
        ("head_tcp", "tcp", head_tcp_endpoint),
    ]
    for variant, protocol, endpoint in one_lane_variants:
        for bg in (1, 10, 50, 90):
            bg_rate = 200 * bg / 100
            fg_rate = 200 - bg_rate
            runs.append(
                MatrixRun(
                    run=f"200_1x200_{variant}_bg{bg}_cap200_moonbg_fixed",
                    lanes=(endpoint,),
                    shards=1,
                    max_bytes="2GB",
                    sizes=DENSE_SIZES_1,
                    protocol=protocol,
                    lane_cap_gbps=200,
                    bg_rate_gbps=bg_rate,
                    fg_rate_gbps=fg_rate,
                )
            )
    return runs


def multi_hca_matrix_runs() -> List[MatrixRun]:
    runs: List[MatrixRun] = []
    specs = [
        ("200_2x100", (0, 1), 100),
        ("200_4x50", (0, 1, 2, 3), 50),
        ("400_4x100", (0, 1, 2, 3), 100),
        ("400_2x200", (0, 1), 200),
    ]
    for prefix, lanes, lane_cap_gbps in specs:
        for bg_percent in (1, 10, 50, 90):
            runs.append(
                multi_hca_run(
                    prefix=prefix,
                    lanes=lanes,
                    lane_cap_gbps=lane_cap_gbps,
                    bg_percent=bg_percent,
                )
            )
    return runs


def multi_hca_compare_4x100_runs() -> List[MatrixRun]:
    runs: List[MatrixRun] = []
    for bg_percent in (1, 10, 50, 90):
        runs.append(split_4x100_run(bg_percent))
        runs.append(
            multi_hca_run(
                prefix="400_4x100",
                lanes=(0, 1, 2, 3),
                lane_cap_gbps=100,
                bg_percent=bg_percent,
            )
        )
    return runs


def multi_hca_bgcap_only_runs() -> List[MatrixRun]:
    return [
        multi_hca_bgcap_only_run(
            prefix="400_4x100",
            lanes=(0, 1, 2, 3),
            total_cap_gbps=400,
            bg_percent=bg_percent,
        )
        for bg_percent in (1, 10, 50, 90)
    ]


def split_empty_sizes_and_max_bytes(shards: int) -> Tuple[str, str]:
    if shards == 1:
        return DENSE_SIZES_1, "2GB"
    if shards == 2:
        return DENSE_SIZES_2, "1GB"
    if shards == 4:
        return DENSE_SIZES_4, "512MB"
    raise ValueError(f"unsupported split shard count: {shards}")


def ratelimit_empty_split_run(
    *,
    prefix: str,
    lanes: Iterable[int],
    per_lane_rate_gbps: float,
) -> MatrixRun:
    lane_list = tuple(lanes)
    sizes, max_bytes = split_empty_sizes_and_max_bytes(len(lane_list))
    return MatrixRun(
        run=f"{prefix}_bg0_ratelimit_split",
        lanes=tuple(bond_endpoint(lane) for lane in lane_list),
        shards=len(lane_list),
        max_bytes=max_bytes,
        sizes=sizes,
        protocol="rdma",
        lane_cap_gbps=per_lane_rate_gbps,
        bg_rate_gbps=0.0,
        fg_rate_gbps=per_lane_rate_gbps,
        capfill_lanes=(),
        capfill_rate_gbps=0.0,
    )


def ratelimit_empty_multihca_run(
    *,
    prefix: str,
    lanes: Iterable[int],
    total_rate_gbps: float,
) -> MatrixRun:
    lane_list = tuple(lanes)
    return MatrixRun(
        run=f"{prefix}_bg0_ratelimit_multihca",
        lanes=(multi_hca_endpoint(lane_list),),
        shards=1,
        max_bytes="2GB",
        sizes=DENSE_SIZES_1,
        protocol="rdma",
        lane_cap_gbps=total_rate_gbps / len(lane_list),
        bg_rate_gbps=0.0,
        fg_rate_gbps=total_rate_gbps,
        capfill_lanes=(),
        capfill_rate_gbps=0.0,
    )


def ratelimit_empty_runs() -> List[MatrixRun]:
    runs: List[MatrixRun] = []
    specs = [
        ("800_4x200", (0, 1, 2, 3), 200.0, 800.0),
        ("400_4x100", (0, 1, 2, 3), 100.0, 400.0),
        ("400_2x200", (0, 1), 200.0, 400.0),
        ("200_1x200", (0,), 200.0, 200.0),
    ]
    for prefix, lanes, per_lane_rate_gbps, total_rate_gbps in specs:
        runs.append(
            ratelimit_empty_split_run(
                prefix=prefix,
                lanes=lanes,
                per_lane_rate_gbps=per_lane_rate_gbps,
            )
        )
        runs.append(
            ratelimit_empty_multihca_run(
                prefix=prefix,
                lanes=lanes,
                total_rate_gbps=total_rate_gbps,
            )
        )
    return runs


def competition_cases() -> List[CompetitionCase]:
    cases: List[CompetitionCase] = []
    two_flow_other_sizes = ("256MB", "512MB", "1GB", "2GB", "4GB")
    for mode in ("fair", "uncapped"):
        for other in two_flow_other_sizes:
            cases.append(
                CompetitionCase(
                    run=f"comp_2flows_2GB_vs_{other}_{mode}",
                    flow_sizes=("2GB", other),
                    protocol="rdma",
                    ib_device="mlx5_bond_0",
                    lane_cap_gbps=200,
                    mode=mode,
                )
            )
        for count in (1, 2, 4, 8):
            cases.append(
                CompetitionCase(
                    run=f"comp_{count}flows_2GB_equal_{mode}",
                    flow_sizes=tuple("2GB" for _ in range(count)),
                    protocol="rdma",
                    ib_device="mlx5_bond_0",
                    lane_cap_gbps=200,
                    mode=mode,
                )
            )
    return cases


def selected_matrix_runs(args: argparse.Namespace) -> List[MatrixRun]:
    runs = matrix_runs(args)
    if args.only:
        requested = set(args.only)
        runs = [run for run in runs if run.run in requested]
    return runs


def selected_multi_hca_matrix_runs(args: argparse.Namespace) -> List[MatrixRun]:
    runs = multi_hca_matrix_runs()
    if args.only:
        requested = set(args.only)
        runs = [run for run in runs if run.run in requested]
    return runs


def selected_multi_hca_compare_4x100_runs(args: argparse.Namespace) -> List[MatrixRun]:
    runs = multi_hca_compare_4x100_runs()
    if args.only:
        requested = set(args.only)
        runs = [run for run in runs if run.run in requested]
    return runs


def selected_multi_hca_bgcap_only_runs(args: argparse.Namespace) -> List[MatrixRun]:
    runs = multi_hca_bgcap_only_runs()
    if args.only:
        requested = set(args.only)
        runs = [run for run in runs if run.run in requested]
    return runs


def selected_ratelimit_empty_runs(args: argparse.Namespace) -> List[MatrixRun]:
    runs = ratelimit_empty_runs()
    if args.only:
        requested = set(args.only)
        runs = [run for run in runs if run.run in requested]
    return runs


def selected_competition_cases(args: argparse.Namespace) -> List[CompetitionCase]:
    cases = competition_cases()
    if args.only:
        requested = set(args.only)
        cases = [case for case in cases if case.run in requested]
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run KV transfer fixed-background and Mooncake competition experiments from node 099."
    )
    parser.add_argument(
        "--suite",
        choices=(
            "fixed-missing",
            "multi-hca-bg",
            "multi-hca-portcap-bg",
            "multi-hca-bgcap-only",
            "multi-hca-compare-4x100",
            "ratelimit-empty",
            "competition",
            "all",
            "list",
        ),
        default="list",
    )
    parser.add_argument("--only", action="append", help="Run only this run name. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--target-ssh-host", default=os.environ.get("TARGET_SSH_HOST", "root@192.168.0.41"))
    parser.add_argument("--ssh-key", default=os.environ.get("TARGET_SSH_KEY", "~/.ssh/kvbench_102"))
    parser.add_argument("--bench-dir", default=os.environ.get("KV_BENCH_DIR", "/root/kv_transfer_bench"))
    parser.add_argument("--image", default=os.environ.get("SGLANG_IMAGE", "sglang-pd-switch:tianciJ"))
    parser.add_argument("--out-root", default=os.environ.get("KV_OUT_ROOT", "/tmp/kv-transfer-bench/auto"))
    parser.add_argument("--chunk-size", default=os.environ.get("KV_CHUNK_SIZE", "16MB"))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("KV_WARMUP", "3")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("KV_REPEAT", "20")))
    parser.add_argument("--bg-duration-seconds", type=float, default=float(os.environ.get("KV_BG_DURATION", "1200")))
    parser.add_argument("--bg-warmup-seconds", type=float, default=float(os.environ.get("KV_BG_WARMUP", "12")))
    parser.add_argument("--single-nic-src-host", default=os.environ.get("KV_SINGLE_NIC_SRC_HOST", SRC_IPS[0]))
    parser.add_argument("--single-nic-tgt-host", default=os.environ.get("KV_SINGLE_NIC_TGT_HOST", TGT_IPS[0]))
    parser.add_argument("--single-nic-ib-device", default=os.environ.get("KV_SINGLE_NIC_IB_DEVICE", "mlx5_0"))
    parser.add_argument("--head-rdma-src-host", default=os.environ.get("KV_HEAD_RDMA_SRC_HOST", SRC_IPS[0]))
    parser.add_argument("--head-rdma-tgt-host", default=os.environ.get("KV_HEAD_RDMA_TGT_HOST", TGT_IPS[0]))
    parser.add_argument("--head-rdma-ib-device", default=os.environ.get("KV_HEAD_RDMA_IB_DEVICE", "mlx5_bond_0"))
    parser.add_argument("--head-tcp-src-host", default=os.environ.get("KV_HEAD_TCP_SRC_HOST", "192.168.0.39"))
    parser.add_argument("--head-tcp-tgt-host", default=os.environ.get("KV_HEAD_TCP_TGT_HOST", "192.168.0.41"))
    parser.add_argument("--head-tcp-ib-device", default=os.environ.get("KV_HEAD_TCP_IB_DEVICE", "mlx5_bond_0"))
    parser.add_argument(
        "--competition-start-delay-seconds",
        type=float,
        default=float(os.environ.get("KV_COMP_START_DELAY", "8")),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner(args)
    listed = set()

    def print_list_item(name: str) -> None:
        if name in listed:
            return
        listed.add(name)
        print(name)

    if args.suite in ("list", "fixed-missing", "all"):
        for run in selected_matrix_runs(args):
            if args.suite == "list":
                print_list_item(run.run)
            else:
                runner.run_matrix(run)

    if args.suite in ("list", "multi-hca-bg", "multi-hca-portcap-bg"):
        for run in selected_multi_hca_matrix_runs(args):
            if args.suite == "list":
                print_list_item(run.run)
            else:
                runner.run_matrix(run)

    if args.suite in ("list", "multi-hca-compare-4x100"):
        for run in selected_multi_hca_compare_4x100_runs(args):
            if args.suite == "list":
                print_list_item(run.run)
            else:
                runner.run_matrix(run)

    if args.suite in ("list", "multi-hca-bgcap-only"):
        for run in selected_multi_hca_bgcap_only_runs(args):
            if args.suite == "list":
                print_list_item(run.run)
            else:
                runner.run_matrix(run)

    if args.suite in ("list", "ratelimit-empty"):
        for run in selected_ratelimit_empty_runs(args):
            if args.suite == "list":
                print_list_item(run.run)
            else:
                runner.run_matrix(run)

    if args.suite in ("list", "competition", "all"):
        for case in selected_competition_cases(args):
            if args.suite == "list":
                print_list_item(case.run)
            else:
                runner.run_competition(case)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
