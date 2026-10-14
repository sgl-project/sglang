"""PPU 评测环境证据；纯标准库、有界采集，不参与测试门禁。

每次采集在独立进程组内执行，包含 NAS 读写的整个过程受父进程超时保护。
只读实际服务进程树的映射，不遍历共享缓存、不加载 torch、不执行 GPU 计算。
分组哈希用于快速比较已采集字段，不能证明未采集的环境或内存状态相同。
"""

import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "ppu-environment/v1"
PHASES = {"before_start", "server_ready", "before_stop", "setup_failed"}
MAX_PROCESSES = 64
MAX_LIBRARIES = 64
MAX_TEXT_BYTES = 2 * 1024 * 1024
ENV_ALLOWLIST = frozenset(
    """
CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER LD_LIBRARY_PATH LD_PRELOAD
HF_HUB_OFFLINE HF_HUB_CACHE PPU_SUPPORTS_FP8
SGLANG_WARMUP_TIMEOUT SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
SGLANG_SAIL_PLA_CUDA SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED
SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE PCCL_MNNVL_ENABLE
SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8
SGLANG_DG_CACHE_DIR DG_JIT_CACHE_DIR SGLANG_JIT_CACHE_DIR
TVM_FFI_CACHE_DIR TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH
GLOO_SOCKET_IFNAME NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX MC_GID_INDEX
NCCL_IB_HCA NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_DEBUG
RANK LOCAL_RANK NODE_RANK WORLD_SIZE NNODES NPROC_PER_NODE
""".split()
)
JIT_CACHE_KEYS = frozenset(
    "SGLANG_DG_CACHE_DIR DG_JIT_CACHE_DIR SGLANG_JIT_CACHE_DIR TVM_FFI_CACHE_DIR TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH".split()
)
SECRET_KEY = re.compile(
    r"password|secret|credential|authorization|api[_-]?key|access[_-]?token|^token$",
    re.I,
)
LIBRARY_NAME = re.compile(
    r"hggc|acext|acrtc|pccl|nccl|cuda|cublas|torch|sgl|flash|tvm|triton", re.I
)
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "quantize_config.json",
    "quant_config.json",
    "model.safetensors.index.json",
)
PACKAGES = (
    "torch",
    "sglang",
    "sglang-kernel",
    "acext",
    "hggc",
    "pccl",
    "transformers",
    "flashinfer-python",
    "apache-tvm-ffi",
)


def redact(value):
    """不序列化凭证字段，也不保留可能含认证参数的 URL。"""
    if isinstance(value, dict):
        return {
            str(k): (
                "[redacted]"
                if SECRET_KEY.search(str(k))
                else (
                    safe_environment(v)
                    if k in {"environment", "env"} and isinstance(v, dict)
                    else redact(v)
                )
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        if re.search(r"https?://", value, re.I):
            return "[url omitted]"
        return value[:8192]
    return value


def redact_args(args):
    result, hide_next = [], False
    for arg in args:
        arg = str(arg)
        if hide_next:
            result.append("[redacted]")
            hide_next = False
        elif arg.startswith("--") and SECRET_KEY.search(arg[2:].split("=", 1)[0]):
            result.append(arg.split("=", 1)[0] + ("=[redacted]" if "=" in arg else ""))
            hide_next = "=" not in arg
        else:
            result.append(redact(arg))
    return result


def safe_environment(environment):
    return {
        key: redact(str(environment[key]))
        for key in sorted(ENV_ALLOWLIST)
        if key in environment
    }


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def error_status(error):
    if isinstance(error, FileNotFoundError):
        return "missing"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, CollectionLimit):
        return error.status
    return type(error).__name__


class CollectionLimit(ValueError):
    def __init__(self, status):
        self.status = status


class ReadBudget:
    def __init__(
        self, max_file_bytes=8 * 1024 * 1024, max_bytes=32 * 1024 * 1024, max_files=1024
    ):
        self.max_file_bytes = max_file_bytes
        self.remaining = max_bytes
        self.files_remaining = max_files

    def open_file(self):
        if self.files_remaining <= 0:
            raise CollectionLimit("file_count_limit")
        self.files_remaining -= 1

    def read(self, stream, size):
        if size > self.remaining:
            raise CollectionLimit("budget_exhausted")
        data = stream.read(size)
        self.remaining -= len(data)
        return data


def elf_build_id(stream, budget):
    """只读 ELF64 程序头及 PT_NOTE，不让 readelf 隐式扫描大库。"""
    remaining = min(65536, budget.max_file_bytes)

    def read_at(offset, size):
        nonlocal remaining
        if size > remaining:
            raise CollectionLimit("elf_note_limit")
        stream.seek(offset)
        data = budget.read(stream, size)
        remaining -= len(data)
        return data

    if remaining < 64:
        return None
    header = read_at(0, 64)
    if len(header) != 64 or header[:5] != b"\x7fELF\x02" or header[5] not in (1, 2):
        return None
    endian = "<" if header[5] == 1 else ">"
    phoff = struct.unpack_from(endian + "Q", header, 32)[0]
    entsize, count = struct.unpack_from(endian + "HH", header, 54)
    if entsize != 56 or count > 128:
        return None
    for index in range(count):
        program = read_at(phoff + index * entsize, entsize)
        if len(program) != entsize:
            return None
        kind, _, offset, _, _, size, _, _ = struct.unpack(endian + "IIQQQQQQ", program)
        if kind != 4:
            continue
        notes = read_at(offset, size)
        position = 0
        while position + 12 <= len(notes):
            namesize, descsize, kind = struct.unpack_from(
                endian + "III", notes, position
            )
            position += 12
            name = notes[position : position + namesize]
            position += (namesize + 3) & ~3
            desc = notes[position : position + descsize]
            if (
                kind == 3
                and name == b"GNU\0"
                and 0 < descsize <= 64
                and len(desc) == descsize
            ):
                return desc.hex()
            position += (descsize + 3) & ~3
    return None


def file_identity(path, budget, mapped_identities=None):
    """哈希前限制大小；元数据不是内容校验，不给缺失/截断内容生成哈希。"""
    path = Path(path)
    result = {"path": str(path)}
    try:
        budget.open_file()
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return {**result, "status": "not_regular"}
            if mapped_identities is not None and mapped_identities != {
                (before.st_dev, before.st_ino)
            }:
                return {**result, "status": "mapped_file_replaced"}
            result.update(size_bytes=before.st_size, resolved_path=str(path.resolve()))
            if before.st_size > budget.max_file_bytes:
                build_id = elf_build_id(stream, budget)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    return {**result, "status": "changed_during_read"}
                if build_id:
                    return {**result, "status": "build_id_only", "build_id": build_id}
                return {**result, "status": "size_limit"}
            if before.st_size > budget.remaining:
                return {**result, "status": "budget_exhausted"}
            data = budget.read(stream, min(before.st_size + 1, budget.remaining))
            after = os.fstat(stream.fileno())
            if len(data) != before.st_size or (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return {**result, "status": "changed_during_read"}
            return {
                **result,
                "status": "ok",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    except IsADirectoryError:
        return {**result, "status": "not_regular"}
    except (OSError, CollectionLimit) as error:
        return {**result, "status": error_status(error)}


def read_text(path, limit=MAX_TEXT_BYTES, budget=None):
    budget = budget or ReadBudget()
    budget.open_file()
    limit = min(limit, budget.max_file_bytes)
    with open(path, "rb") as stream:
        data = budget.read(stream, min(limit + 1, budget.remaining))
    if len(data) > limit:
        raise CollectionLimit("text_size_limit")
    if budget.remaining == 0:
        raise CollectionLimit("budget_exhausted")
    return data.decode("utf-8", errors="replace")


def command(args, timeout=2, budget=None):
    """只执行固定诊断命令；不记录 stderr，以免带出环境中的认证信息。"""
    process = None
    budget = budget or ReadBudget()
    try:
        budget.open_file()
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        deadline, output = time.monotonic() + timeout, bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    return {"status": "timeout"}
                if budget.remaining <= 0:
                    return {"status": "budget_exhausted"}
                block = os.read(
                    process.stdout.fileno(),
                    min(65536, MAX_TEXT_BYTES + 1 - len(output), budget.remaining),
                )
                budget.remaining -= len(block)
                if not block:
                    break
                output.extend(block)
                if len(output) > MAX_TEXT_BYTES:
                    return {"status": "output_limit"}
        code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if code:
            return {"status": "command_failed", "returncode": code}
        return {
            "status": "ok",
            "output": output.decode("utf-8", errors="replace").strip(),
        }
    except (OSError, subprocess.SubprocessError, CollectionLimit) as error:
        return {"status": error_status(error)}
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def parse_pod_metadata(text, node_rank):
    candidates = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) == 5 and fields[0].endswith(f"-worker-{node_rank}"):
            candidates.append(
                dict(
                    zip(
                        (
                            "pod_name",
                            "pod_uid",
                            "node_name",
                            "requested_image",
                            "image_id",
                        ),
                        fields,
                    )
                )
            )
    if len(candidates) != 1:
        return {"status": "ambiguous" if candidates else "missing"}
    row = candidates[0]
    row["status"] = "ok" if "sha256:" in row["image_id"] else "pending"
    return row


def pod_snapshot(node_rank, budget=None):
    path = os.environ.get("SGLANG_PPU_POD_METADATA")
    if not path:
        return {"status": "not_configured", "node_name": os.environ.get("NODE_NAME")}
    try:
        observer_status = read_text(
            Path(path).with_suffix(".status"), 128, budget
        ).strip()
        if not re.fullmatch(
            r"(?:timeout:)?(?:ok|waiting|waiting_for_image_ids|query_failed|kubectl_unavailable)",
            observer_status,
        ):
            observer_status = "invalid_status"
    except (OSError, ValueError) as error:
        observer_status = error_status(error)
    try:
        result = parse_pod_metadata(read_text(path, budget=budget), node_rank)
        expected_node = os.environ.get("NODE_NAME")
        if (
            expected_node
            and result.get("node_name")
            and result["node_name"] != expected_node
        ):
            result = {"status": "node_mismatch"}
    except (OSError, ValueError) as error:
        result = {"status": error_status(error)}
    return {**result, "observer_status": observer_status}


def model_snapshot(model_path, budget):
    if not model_path:
        return {"status": "not_configured"}
    return {
        "status": "partial",
        "path": str(model_path),
        "files": [
            file_identity(Path(model_path) / name, budget) for name in MODEL_FILES
        ],
        "weights_verification": "not_collected",
    }


def is_jit_path(path):
    roots = [
        os.environ[key].rstrip("/") + "/"
        for key in JIT_CACHE_KEYS
        if os.environ.get(key)
    ]
    return "/.cache/" in path or any(path.startswith(root) for root in roots)


def process_snapshot(pid, budget, proc_root=Path("/proc")):
    """只访问服务及其子进程；PID/进程环境单独保留，不进入二进制指纹。"""
    if not pid:
        return {"status": "not_running", "processes": [], "libraries": []}
    pending, processes, paths, seen = [int(pid)], [], {}, set()
    while pending and len(seen) < MAX_PROCESSES:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        base = Path(proc_root) / str(current)
        item = {"pid": current}
        try:
            with os.scandir(base / "task") as tasks:
                threads = list(itertools.islice(tasks, 257))
            item["children_status"] = "thread_limit" if len(threads) > 256 else "ok"
            for thread in threads[:256]:
                children = read_text(Path(thread.path) / "children", 65536, budget)
                pending.extend(
                    int(p)
                    for p in children.split()
                    if p.isdigit() and int(p) not in seen
                )
        except (OSError, ValueError) as error:
            item["children_status"] = error_status(error)
        try:
            mappings = read_text(base / "maps", budget=budget)
            item["status"] = "ok"
            local_paths = set()
            for line in mappings.splitlines():
                fields = line.split(None, 5)
                if len(fields) != 6 or not fields[5].startswith("/"):
                    continue
                path = fields[5]
                if is_jit_path(path) or (
                    ".so" in path and LIBRARY_NAME.search(Path(path).name)
                ):
                    major, minor = fields[3].split(":")
                    mapped = (
                        os.makedev(int(major, 16), int(minor, 16)),
                        int(fields[4]),
                    )
                    paths.setdefault(path, set()).add(mapped)
                    local_paths.add(path)
            item["library_paths"] = sorted(local_paths)[:MAX_LIBRARIES]
            try:
                environment = dict(
                    field.split("=", 1)
                    for field in read_text(base / "environ", budget=budget).split("\0")
                    if "=" in field
                )
                item["rank_environment"] = {
                    key: redact(environment[key])
                    for key in (
                        "RANK",
                        "LOCAL_RANK",
                        "NODE_RANK",
                        "CUDA_VISIBLE_DEVICES",
                    )
                    if key in environment
                }
            except (OSError, ValueError) as error:
                item["environment_status"] = error_status(error)
        except (OSError, ValueError) as error:
            item["status"] = error_status(error)
        processes.append(item)
    libraries = []
    for path in sorted(paths)[:MAX_LIBRARIES]:
        if path.endswith(" (deleted)"):
            libraries.append({"path": path, "status": "deleted_mapping"})
            continue
        identity = file_identity(path, budget, paths[path])
        identity["is_jit"] = is_jit_path(path)
        libraries.append(identity)
    complete = bool(processes) and all(
        p.get("status") == "ok" and p.get("children_status") == "ok" for p in processes
    )
    return {
        "status": (
            "ok"
            if complete and not pending and len(paths) <= MAX_LIBRARIES
            else "partial"
        ),
        "processes": processes,
        "libraries": libraries,
        "truncated": bool(pending or len(paths) > MAX_LIBRARIES),
    }


def hardware_snapshot(budget=None):
    fields = ("index", "uuid", "pci.bus_id", "name", "driver_version", "vbios_version")
    answer = command(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        budget=budget,
    )
    firmware_status = "ok"
    if answer["status"] == "command_failed":
        firmware_status = "unsupported_query"
        fields = fields[:-1]
        answer = command(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            budget=budget,
        )
    if answer["status"] != "ok":
        return answer
    devices = []
    for row in csv.reader(answer["output"].splitlines(), skipinitialspace=True):
        if len(row) == len(fields):
            devices.append(dict(zip(fields, (v.strip() for v in row))))
    return {
        "status": (
            ("ok" if firmware_status == "ok" else "partial")
            if devices
            else "unavailable"
        ),
        "firmware_status": firmware_status,
        "devices": devices,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mapping_scope": "physical_inventory_and_visibility_not_inferred_tp_rank",
    }


def package_snapshot(budget=None):
    versions = {}
    for name in PACKAGES:
        try:
            distribution = importlib.metadata.distribution(name)
            # version() 会无界读取 METADATA；仅支持磁盘上的标准 dist-info。
            metadata_path = getattr(distribution, "_path", None)
            if not isinstance(metadata_path, Path):
                versions[name] = {"status": "unsupported_metadata_provider"}
                continue
            text = read_text(metadata_path / "METADATA", budget=budget)
            match = re.search(r"^Version: ([^\r\n]+)", text, re.M)
            versions[name] = (
                {"status": "ok", "version": match.group(1)}
                if match
                else {"status": "missing_version"}
            )
        except importlib.metadata.PackageNotFoundError:
            versions[name] = {"status": "missing"}
        except (OSError, ValueError) as error:
            versions[name] = {"status": error_status(error)}
    return {"status": "ok", "python": platform.python_version(), "packages": versions}


def source_snapshot(budget=None):
    root = Path(__file__).resolve().parents[4]
    revision = command(["git", "-C", str(root), "rev-parse", "HEAD"], budget=budget)
    return {
        "status": revision["status"],
        "source_sha": revision.get("output"),
        "workflow_sha": os.environ.get("SGLANG_PPU_WORKFLOW_SHA"),
        "workflow_sha_scope": "github.workflow_sha_context; reusable_callee_sha_not_inferred",
        "distributed_action_sha": os.environ.get("SGLANG_PPU_DISTRIBUTED_ACTION_SHA"),
    }


def make_group(data):
    # data 只包含可比较字段；采集时间、PID、Pod UID 留在 snapshot.observation。
    observed = data.get("status", "ok") in {"ok", "partial", "build_id_only"}
    return {
        "data": data,
        "sha256": digest(data) if observed else None,
        "comparison_scope": "observed_fields_only; missing_fields_are_not_equality_evidence",
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.partial")
    staged.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staged.replace(path)


def capture_to_file(request, collect=True):
    path = Path(request["output_dir"]) / "environment.json"
    try:
        report = json.loads(read_text(path))
    except (OSError, ValueError):
        report = {
            "schema_version": SCHEMA,
            "node_rank": request["node_rank"],
            "snapshots": {},
        }
    phase = request["phase"]
    if phase not in PHASES:
        raise ValueError("unknown_phase")
    if request.get("failure_status"):
        snapshot = report["snapshots"].setdefault(phase, {"groups": {}})
        snapshot["status"] = request["failure_status"]
        atomic_json(path, report)
        return
    snapshot = {
        "status": "collecting",
        "captured_at_unix": time.time(),
        "groups": {},
        "observation": {
            "server_pid": request.get("server_pid"),
            "node_name": os.environ.get("NODE_NAME"),
        },
    }
    report["snapshots"][phase] = snapshot
    atomic_json(path, report)
    if not collect:
        return
    budget = ReadBudget()

    def record(name, collector):
        try:
            value = collector()
        except Exception as error:
            value = {"status": error_status(error)}
        snapshot["groups"][name] = make_group(value)
        atomic_json(path, report)

    record("source", lambda: source_snapshot(budget))
    environment = safe_environment({**os.environ, **request["server_env"]})
    snapshot["observation"]["environment"] = environment
    record(
        "configuration",
        lambda: {
            "config": redact(request["config"]),
            "server_args": redact_args(request["server_args"]),
            "environment": {
                k: v for k, v in environment.items() if k not in JIT_CACHE_KEYS
            },
        },
    )
    pod = pod_snapshot(request["node_rank"], budget)
    snapshot["observation"]["pod"] = pod
    record(
        "image",
        lambda: {
            "status": pod["status"],
            "actual_image_id": pod.get("image_id"),
            "requested_image": pod.get(
                "requested_image", os.environ.get("PPU_BASE_IMAGE")
            ),
        },
    )
    record("hardware", lambda: hardware_snapshot(budget))
    record("packages", lambda: package_snapshot(budget))
    record(
        "model",
        lambda: model_snapshot(request["config"].get("model", {}).get("path"), budget),
    )
    process = process_snapshot(request.get("server_pid"), budget)
    snapshot["observation"]["processes"] = process["processes"]
    identities = [
        {k: v for k, v in item.items() if k not in {"path", "resolved_path"}}
        | {"name": Path(item["path"]).name}
        for item in process["libraries"]
    ]
    snapshot["observation"]["library_paths"] = process["libraries"]
    record(
        "libraries",
        lambda: {
            "status": process["status"],
            "truncated": process.get("truncated", False),
            "identities": sorted(
                identities, key=lambda x: (x["name"], x.get("sha256", ""))
            ),
        },
    )
    record(
        "jit",
        lambda: {
            "status": "partial",
            "scope": "mapped_files_only",
            "identities": sorted(
                (item for item in identities if item.get("is_jit")),
                key=lambda x: (x["name"], x.get("sha256", "")),
            ),
            "cache_hit": "unobserved",
            "in_memory_rtc_artifacts": "not_collected",
        },
    )
    snapshot["status"] = "collected"
    atomic_json(path, report)


class EnvironmentRecorder:
    """调用方无需捕获诊断异常；服务退出码和门禁保持原样。"""

    def __init__(self, output_dir, config, node_rank=0, timeout=15):
        self.output_dir = str(output_dir)
        self.config = config
        self.node_rank = node_rank
        self.timeout = timeout
        self.server_args = []
        self.server_env = {}

    @staticmethod
    def _run(request, timeout):
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--capture"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            process.communicate(json.dumps(request).encode(), timeout=timeout)
            return "ok" if process.returncode == 0 else "collector_failed"
        except Exception as error:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=0.5)
                except (OSError, subprocess.SubprocessError):
                    pass
            return error_status(error)

    def capture(self, phase, server_pid=None, server_args=None, server_env=None):
        try:
            if server_args is not None:
                self.server_args = redact_args(server_args)
            if server_env is not None:
                self.server_env = safe_environment(server_env)
            request = {
                "output_dir": self.output_dir,
                "config": redact(self.config),
                "node_rank": self.node_rank,
                "phase": phase,
                "server_pid": server_pid,
                "server_args": self.server_args,
                "server_env": self.server_env,
            }
            status = self._run(request, self.timeout)
            if status != "ok":
                request["failure_status"] = status
                self._run(request, 1)
                print(f"环境指纹采集未完成：{phase} ({status})", flush=True)
            return status == "ok"
        except Exception:
            print("环境指纹采集失败；保留原测试结果", flush=True)
            return False


if __name__ == "__main__":
    if sys.argv[1:] == ["--capture"]:
        capture_to_file(json.loads(sys.stdin.buffer.read(MAX_TEXT_BYTES)))
