# SPDX-License-Identifier: Apache-2.0
"""Weight-cache parity harness: disk-load vs daemon-load state_dict diff.

Per rank, snapshots a disk-loaded reference model and a daemon-served
IPC-loaded model, then diffs names/shape/dtype/stride/kind/persistence/
byte-hash; any difference fails. A clean run is the evidence required to
extend IPC_QUANT_ALLOWLIST (add the candidate entry first — both sides gate
on it). Phases run sequentially as one subprocess per rank so two full
weight copies never coexist on a GPU.

--gsm8k N additionally greedy-decodes N few-shot GSM8K questions on a
disk-loaded and a daemon-loaded engine and gates on the accuracy delta.

Usage:
    python -m sglang.srt.weight_cache.parity --model-path M --tp-size 2 [--gsm8k 20]
"""

import argparse
import hashlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List

import msgspec

from .protocol import compute_local_gpu_id, get_ready_path

logger = logging.getLogger(__name__)

# How many offending tensor names each report section prints in full.
REPORT_DETAIL_CAP = 20

_DIFFED_FIELDS = ("shape", "dtype", "stride", "kind", "persistent", "byte_hash")


class TensorRecord(msgspec.Struct, frozen=True):
    shape: List[int]
    stride: List[int]
    dtype: str
    kind: str  # "param" | "buffer"
    persistent: bool  # buffers only; params are always True
    byte_hash: str


class RankManifest(msgspec.Struct, frozen=True):
    side: str  # "ref" | "ipc"
    tp_rank: int
    pp_rank: int
    records: Dict[str, TensorRecord]


class FieldMismatch(msgspec.Struct, frozen=True):
    name: str
    field: str
    ref_value: str
    ipc_value: str


class RankDiff(msgspec.Struct, frozen=True):
    missing_in_ipc: List[str]
    extra_in_ipc: List[str]
    mismatches: List[FieldMismatch]

    @property
    def is_clean(self) -> bool:
        return not (self.missing_in_ipc or self.extra_in_ipc or self.mismatches)


def _hash_tensor_bytes(tensor) -> str:
    import torch

    flat = tensor.detach().contiguous().view(-1).cpu()
    if flat.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(flat.view(torch.uint8).numpy()).hexdigest()


def _iter_module_tensors(model):
    """Yield (name, tensor, kind, persistent) for every param/buffer path.

    No visited-module memo (mirrors state_dict recursion): a module shared
    under several parents (rotary _ROPE_DICT) appears under every alias path,
    so both sides enumerate the same name set.
    """
    stack = [("", model)]
    while stack:
        prefix, module = stack.pop()
        for name, param in module._parameters.items():
            if param is not None:
                yield prefix + name, param, "param", True
        for name, buf in module._buffers.items():
            if buf is not None:
                persistent = name not in module._non_persistent_buffers_set
                yield prefix + name, buf, "buffer", persistent
        for child_name, child in module._modules.items():
            if child is not None:
                stack.append((prefix + child_name + ".", child))


def snapshot_state(model) -> Dict[str, TensorRecord]:
    """Record every parameter and buffer of a loaded model, keyed by path."""
    records = {}
    for name, tensor, kind, persistent in _iter_module_tensors(model):
        records[name] = TensorRecord(
            shape=list(tensor.shape),
            stride=list(tensor.stride()),
            dtype=str(tensor.dtype).replace("torch.", ""),
            kind=kind,
            persistent=persistent,
            byte_hash=_hash_tensor_bytes(tensor),
        )
    return records


def diff_manifests(
    ref: Dict[str, TensorRecord], ipc: Dict[str, TensorRecord]
) -> RankDiff:
    missing = sorted(set(ref) - set(ipc))
    extra = sorted(set(ipc) - set(ref))
    mismatches = []
    for name in sorted(set(ref) & set(ipc)):
        ref_rec, ipc_rec = ref[name], ipc[name]
        for field in _DIFFED_FIELDS:
            ref_value, ipc_value = getattr(ref_rec, field), getattr(ipc_rec, field)
            if ref_value != ipc_value:
                mismatches.append(
                    FieldMismatch(
                        name=name,
                        field=field,
                        ref_value=str(ref_value),
                        ipc_value=str(ipc_value),
                    )
                )
    return RankDiff(missing_in_ipc=missing, extra_in_ipc=extra, mismatches=mismatches)


def format_rank_report(
    *, tp_rank: int, pp_rank: int, diff: RankDiff, num_ref: int, num_ipc: int
) -> str:
    verdict = "PASS" if diff.is_clean else "FAIL"
    lines = [
        f"[pp={pp_rank} tp={tp_rank}] tensors: ref={num_ref} ipc={num_ipc} | "
        f"missing={len(diff.missing_in_ipc)} extra={len(diff.extra_in_ipc)} "
        f"mismatched={len(diff.mismatches)} -> {verdict}"
    ]

    def _capped(title: str, items: List[str]) -> None:
        if not items:
            return
        lines.append(f"  {title}:")
        for item in items[:REPORT_DETAIL_CAP]:
            lines.append(f"    {item}")
        if len(items) > REPORT_DETAIL_CAP:
            lines.append(f"    ... and {len(items) - REPORT_DETAIL_CAP} more")

    _capped("missing in ipc (daemon did not serve)", diff.missing_in_ipc)
    _capped("extra in ipc (not in disk-loaded model)", diff.extra_in_ipc)
    _capped(
        "field mismatches",
        [
            f"{m.name}: {m.field} ref={m.ref_value} ipc={m.ipc_value}"
            for m in diff.mismatches
        ],
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-rank snapshot subprocess
# ---------------------------------------------------------------------------


def _build_daemon(args: argparse.Namespace):
    from sglang.srt.server_args import ServerArgs

    from .daemon import WeightCacheDaemon

    server_args = ServerArgs(
        model_path=args.model_path,
        dtype=args.dtype,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        load_format=args.load_format,
        model_loader_extra_config=args.model_loader_extra_config,
    )
    return WeightCacheDaemon(
        server_args=server_args,
        gpu_id=args.gpu_id,
        tp_rank=args.tp_rank,
        pp_rank=args.pp_rank,
        dist_init_method=args.dist_init_method,
    )


def _snapshot_ipc_model(args: argparse.Namespace):
    """Load the model from a live daemon exactly the way an engine rank does."""
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadFormat
    from sglang.srt.platforms import current_platform

    from .ipc_loader import IpcModelLoader

    setup = _build_daemon(args)
    model_config, load_config = setup._prepare_load()
    load_config.load_format = LoadFormat.IPC_CACHE

    # weight_cache_mode="daemon" forbids the disk fallback: a fallback here
    # would diff a disk load against a disk load and vacuously pass.
    loader = IpcModelLoader(
        load_config=load_config,
        weight_cache_mode="daemon",
    )
    return loader.load_model(
        model_config=model_config,
        device_config=DeviceConfig(current_platform.device_type, args.gpu_id),
    )


def run_snapshot_role(args: argparse.Namespace) -> None:
    """Child entry: load one rank's model on one side, write its manifest."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s [Parity {args.snapshot_side} pp={args.pp_rank} "
            f"tp={args.tp_rank}] %(levelname)s %(message)s"
        ),
    )
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    if args.snapshot_side == "ref":
        daemon = _build_daemon(args)
        daemon.load()
        model = daemon.model
    else:
        model = _snapshot_ipc_model(args)

    manifest = RankManifest(
        side=args.snapshot_side,
        tp_rank=args.tp_rank,
        pp_rank=args.pp_rank,
        records=snapshot_state(model),
    )
    with open(args.manifest_out, "wb") as f:
        f.write(msgspec.json.encode(manifest))
    logger.info(f"Wrote {len(manifest.records)} tensor records to {args.manifest_out}")


# ---------------------------------------------------------------------------
# GSM8K accuracy gate
# ---------------------------------------------------------------------------

_GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
_GSM8K_FEW_SHOT = 5
# Small-sample accuracy tolerance; identical greedy outputs are reported but
# not required (batching boundaries are not bit-deterministic).
_GSM8K_ACC_TOL = 0.05
_GSM8K_INVALID = -9999999


class GsmResult(msgspec.Struct, frozen=True):
    side: str
    outputs: List[str]
    answers: List[int]
    labels: List[int]

    @property
    def accuracy(self) -> float:
        hits = sum(a == l for a, l in zip(self.answers, self.labels))
        return hits / max(1, len(self.labels))


def _gsm8k_answer_value(text: str) -> int:
    import re

    numbers = re.findall(r"-?\d+", text.replace(",", ""))
    if not numbers:
        return _GSM8K_INVALID
    try:
        return int(numbers[-1])
    except ValueError:
        return _GSM8K_INVALID


def _gsm8k_prompts(num_questions: int):
    from sglang.utils import download_and_cache_file, read_jsonl

    lines = list(read_jsonl(download_and_cache_file(_GSM8K_URL)))
    few_shot = "".join(
        f"Question: {line['question']}\nAnswer: {line['answer']}\n\n"
        for line in lines[:_GSM8K_FEW_SHOT]
    )
    questions = lines[_GSM8K_FEW_SHOT : _GSM8K_FEW_SHOT + num_questions]
    prompts = [
        few_shot + f"Question: {line['question']}\nAnswer:" for line in questions
    ]
    labels = [_gsm8k_answer_value(line["answer"]) for line in questions]
    return prompts, labels


def run_gsm8k_role(args: argparse.Namespace) -> None:
    """Child entry: greedy-decode GSM8K on one side, write a GsmResult."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Parity gsm8k {args.gsm8k_side}] %(levelname)s %(message)s",
    )
    import sglang as sgl
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    prompts, labels = _gsm8k_prompts(args.gsm8k)
    engine = sgl.Engine(
        model_path=args.model_path,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        dtype=args.dtype,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        weight_cache_mode="client" if args.gsm8k_side == "ipc" else "off",
        mem_fraction_static=0.5,
        log_level="error",
    )
    try:
        outs = engine.generate(
            prompts,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 512,
                "stop": ["Question:"],
            },
        )
    finally:
        engine.shutdown()

    outputs = [o["text"] for o in outs]
    result = GsmResult(
        side=args.gsm8k_side,
        outputs=outputs,
        answers=[_gsm8k_answer_value(t) for t in outputs],
        labels=labels,
    )
    with open(args.gsm8k_out, "wb") as f:
        f.write(msgspec.json.encode(result))
    logger.info(f"accuracy={result.accuracy:.3f} over {len(labels)} questions")


def _run_gsm8k_phase(args: argparse.Namespace, *, side: str, out_dir: str) -> str:
    out_path = os.path.join(out_dir, f"gsm8k_{side}.json")
    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.weight_cache.parity",
        "--gsm8k-side",
        side,
        "--gsm8k",
        str(args.gsm8k),
        "--gsm8k-out",
        out_path,
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--pp-size",
        str(args.pp_size),
        "--dtype",
        args.dtype,
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.revision:
        cmd += ["--revision", args.revision]
    _wait_procs([subprocess.Popen(cmd)], what=f"gsm8k {side}", timeout=args.timeout)
    return out_path


def _report_gsm8k(ref_path: str, ipc_path: str) -> bool:
    with open(ref_path, "rb") as f:
        ref = msgspec.json.decode(f.read(), type=GsmResult)
    with open(ipc_path, "rb") as f:
        ipc = msgspec.json.decode(f.read(), type=GsmResult)
    identical = sum(a == b for a, b in zip(ref.outputs, ipc.outputs))
    ok = ipc.accuracy >= ref.accuracy - _GSM8K_ACC_TOL
    print(
        f"gsm8k gate: ref_acc={ref.accuracy:.3f} ipc_acc={ipc.accuracy:.3f} "
        f"identical_outputs={identical}/{len(ref.outputs)} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_pairs(args: argparse.Namespace):
    for pp_rank in range(args.pp_size):
        for tp_rank in range(args.tp_size):
            yield pp_rank, tp_rank


def _manifest_path(out_dir: str, side: str, pp_rank: int, tp_rank: int) -> str:
    return os.path.join(out_dir, f"{side}_pp{pp_rank}_tp{tp_rank}.json")


def _rank_ready_path(args: argparse.Namespace, pp_rank: int, tp_rank: int) -> str:
    from sglang.srt.platforms import current_platform

    gpu_id = compute_local_gpu_id(
        pp_rank,
        tp_rank,
        pp_size_per_node=args.pp_size,
        tp_size_per_node=args.tp_size,
        base_gpu_id=args.base_gpu_id,
        gpu_id_step=args.gpu_id_step,
    )
    return get_ready_path(current_platform.get_device_uuid(gpu_id))


def _wait_procs(procs: List[subprocess.Popen], *, what: str, timeout: int) -> None:
    deadline = time.time() + timeout
    for proc in procs:
        remaining = max(1.0, deadline - time.time())
        try:
            retcode = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            raise TimeoutError(f"{what} did not finish within {timeout}s")
        if retcode != 0:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            raise RuntimeError(f"{what} (pid={proc.pid}) exited with {retcode}")


def _run_snapshot_phase(args: argparse.Namespace, *, side: str, out_dir: str) -> None:
    dist_init_method = f"tcp://127.0.0.1:{_find_free_port()}"
    procs = []
    for pp_rank, tp_rank in _rank_pairs(args):
        gpu_id = compute_local_gpu_id(
            pp_rank,
            tp_rank,
            pp_size_per_node=args.pp_size,
            tp_size_per_node=args.tp_size,
            base_gpu_id=args.base_gpu_id,
            gpu_id_step=args.gpu_id_step,
        )
        cmd = [
            sys.executable,
            "-m",
            "sglang.srt.weight_cache.parity",
            "--snapshot-side",
            side,
            "--manifest-out",
            _manifest_path(out_dir, side, pp_rank, tp_rank),
            "--model-path",
            args.model_path,
            "--gpu-id",
            str(gpu_id),
            "--tp-size",
            str(args.tp_size),
            "--tp-rank",
            str(tp_rank),
            "--pp-size",
            str(args.pp_size),
            "--pp-rank",
            str(pp_rank),
            "--load-format",
            args.load_format,
            "--dtype",
            args.dtype,
            "--dist-init-method",
            dist_init_method,
        ]
        if args.quantization:
            cmd += ["--quantization", args.quantization]
        if args.model_loader_extra_config != "{}":
            cmd += ["--model-loader-extra-config", args.model_loader_extra_config]
        if args.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if args.revision:
            cmd += ["--revision", args.revision]
        procs.append(subprocess.Popen(cmd))
    _wait_procs(procs, what=f"{side} snapshot", timeout=args.timeout)


def _launch_daemons(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.weight_cache.daemon",
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--pp-size",
        str(args.pp_size),
        "--base-gpu-id",
        str(args.base_gpu_id),
        "--gpu-id-step",
        str(args.gpu_id_step),
        "--load-format",
        args.load_format,
        "--dtype",
        args.dtype,
        "--timeout",
        str(args.timeout),
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.model_loader_extra_config != "{}":
        cmd += ["--model-loader-extra-config", args.model_loader_extra_config]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.revision:
        cmd += ["--revision", args.revision]
    if args.force:
        cmd += ["--force"]
    return subprocess.Popen(cmd)


def _wait_daemons_ready(
    args: argparse.Namespace, launcher: subprocess.Popen, *, launched_at: float
) -> None:
    deadline = time.time() + args.timeout

    def _fresh(ready_path: str) -> bool:
        # A ready file older than the launcher is a stale leftover the launcher
        # has not cleaned up yet, not this run's daemon.
        try:
            return os.path.getmtime(ready_path) >= launched_at - 1.0
        except OSError:
            return False

    for pp_rank, tp_rank in _rank_pairs(args):
        ready_path = _rank_ready_path(args, pp_rank, tp_rank)
        while not _fresh(ready_path):
            if launcher.poll() is not None:
                raise RuntimeError(
                    f"daemon launcher exited with {launcher.returncode} "
                    f"before all ranks became ready"
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"daemon rank pp={pp_rank} tp={tp_rank} not ready "
                    f"within {args.timeout}s"
                )
            time.sleep(1)


def _stop_daemons(args: argparse.Namespace, launcher: subprocess.Popen) -> None:
    from .protocol import _is_pid_alive, _read_ready_pid

    if launcher.poll() is None:
        # SIGINT, not SIGTERM: the launcher only runs its child-terminating
        # cleanup via KeyboardInterrupt.
        launcher.send_signal(signal.SIGINT)
        try:
            launcher.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

    # A launcher spawned from a background shell may inherit SIGINT ignored;
    # SIGTERM each daemon directly (its handler unlinks socket/ready).
    for pp_rank, tp_rank in _rank_pairs(args):
        pid = _read_ready_pid(_rank_ready_path(args, pp_rank, tp_rank))
        if pid is not None and _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if launcher.poll() is None:
        launcher.kill()
        launcher.wait(timeout=5)


def _load_manifest(path: str) -> RankManifest:
    with open(path, "rb") as f:
        return msgspec.json.decode(f.read(), type=RankManifest)


def run_parity(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [Parity] %(levelname)s %(message)s"
    )
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tic = time.perf_counter()
    logger.info("Phase 1/3: reference disk-load snapshots")
    _run_snapshot_phase(args, side="ref", out_dir=out_dir)

    gsm_ref_path = None
    if args.gsm8k > 0:
        logger.info("GSM8K gate: reference (disk-load) generation")
        gsm_ref_path = _run_gsm8k_phase(args, side="ref", out_dir=out_dir)

    logger.info("Phase 2/3: launching weight cache daemons")
    launched_at = time.time()
    launcher = _launch_daemons(args)
    try:
        _wait_daemons_ready(args, launcher, launched_at=launched_at)
        logger.info("Phase 3/3: daemon IPC-load snapshots")
        _run_snapshot_phase(args, side="ipc", out_dir=out_dir)
        if args.gsm8k > 0:
            logger.info("GSM8K gate: IPC (daemon-load) generation")
            gsm_ipc_path = _run_gsm8k_phase(args, side="ipc", out_dir=out_dir)
    finally:
        _stop_daemons(args, launcher)

    all_clean = True
    for pp_rank, tp_rank in _rank_pairs(args):
        ref = _load_manifest(_manifest_path(out_dir, "ref", pp_rank, tp_rank))
        ipc = _load_manifest(_manifest_path(out_dir, "ipc", pp_rank, tp_rank))
        diff = diff_manifests(ref.records, ipc.records)
        all_clean = all_clean and diff.is_clean
        print(
            format_rank_report(
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                diff=diff,
                num_ref=len(ref.records),
                num_ipc=len(ipc.records),
            )
        )

    if gsm_ref_path is not None:
        all_clean = _report_gsm8k(gsm_ref_path, gsm_ipc_path) and all_clean

    elapsed = time.perf_counter() - tic
    verdict = "PASS" if all_clean else "FAIL"
    print(
        f"weight-cache parity: {verdict} "
        f"(model={args.model_path}, tp={args.tp_size}, pp={args.pp_size}, "
        f"quant={args.quantization or 'none'}, {elapsed:.1f}s, "
        f"manifests in {out_dir})"
    )
    return 0 if all_clean else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGLang weight-cache parity harness "
        "(disk-load vs daemon-load state_dict diff)"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--gpu-id-step", type=int, default=1)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--model-loader-extra-config", default="{}")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-phase timeout in seconds (snapshot phases and daemon readiness)",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/sglang_weight_cache_parity",
        help="Directory the per-rank manifests are written to",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Kill and take over live daemons already holding this node's ranks",
    )

    parser.add_argument(
        "--gsm8k",
        type=int,
        default=0,
        help="Also run a few-shot GSM8K accuracy gate on this many questions "
        "(disk-loaded vs daemon-loaded engine); 0 disables it",
    )

    # Internal single-rank snapshot mode (used by the orchestrator's children).
    parser.add_argument("--snapshot-side", choices=("ref", "ipc"), default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--pp-rank", type=int, default=0)
    parser.add_argument("--dist-init-method", default=None)
    parser.add_argument("--gsm8k-side", choices=("ref", "ipc"), default=None)
    parser.add_argument("--gsm8k-out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    if cli_args.snapshot_side is not None:
        if cli_args.manifest_out is None:
            raise SystemExit("--manifest-out is required with --snapshot-side")
        run_snapshot_role(cli_args)
    elif cli_args.gsm8k_side is not None:
        if cli_args.gsm8k_out is None or cli_args.gsm8k <= 0:
            raise SystemExit(
                "--gsm8k-out and --gsm8k > 0 are required with --gsm8k-side"
            )
        run_gsm8k_role(cli_args)
    else:
        sys.exit(run_parity(cli_args))
