"""Reference diagnostics for Kimi K3's Nightly NPU HiCache failure.

The environment CLI uses only the standard library. In particular, it does not
initialize torch_npu or prime its process-local compatibility caches.
"""

import ast
import ctypes
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit


def _observe(function):
    """Keep probe errors as data so they cannot replace the original failure."""
    try:
        return function()
    except Exception as exc:  # noqa: BLE001
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_source(path):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "__version__",
                    "git_version",
                ):
                    values[target.id] = ast.literal_eval(node.value)
    return values


def collect_environment_metadata():
    """Collect image/package identity without importing torch or torch_npu."""
    row = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "python": sys.executable,
        "packages": {},
        "library_hashes": {},
    }
    for name in ("torch", "torch-npu", "sgl-kernel-npu", "sglang"):
        row["packages"][name] = _observe(
            lambda name=name: importlib.metadata.version(name)
        )
    for package in ("torch", "torch_npu"):

        def version_source(package=package):
            distribution = importlib.metadata.distribution(package)
            return _version_source(
                Path(distribution.locate_file(f"{package}/version.py"))
            )

        row[f"{package}_version_source"] = _observe(version_source)
    for package, relative in (
        ("torch-npu", "torch_npu/lib/libtorch_npu.so"),
        ("sgl-kernel-npu", "sgl_kernel_npu/lib/libsgl_kernel_npu.so"),
    ):

        def library_identity(package=package, relative=relative):
            path = Path(
                importlib.metadata.distribution(package).locate_file(relative)
            ).resolve()
            return {"path": str(path), "sha256": _sha256(path)}

        row["library_hashes"][package] = _observe(library_identity)

    def kernel_wheel():
        raw = importlib.metadata.distribution("sgl-kernel-npu").read_text(
            "direct_url.json"
        )
        data = json.loads(raw)
        # Keep the build-tagged filename, without URL credentials or signed queries.
        return {
            "filename": Path(unquote(urlsplit(data["url"]).path)).name,
            "archive_info": data.get("archive_info"),
        }

    row["kernel_wheel"] = _observe(kernel_wheel)
    cann = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    ).resolve()
    row["cann_version_files"] = {}
    for path in (
        cann / "aarch64-linux/ascend_toolkit_install.info",
        cann / "share/info/runtime/version.info",
        Path("/usr/local/Ascend/driver/version.info"),
    ):
        row["cann_version_files"][str(path)] = _observe(
            lambda path=path: path.read_text()
        )
    for name in ("libascendcl.so", "libruntime.so"):
        path = cann / "lib64" / name
        row["library_hashes"][name] = _observe(
            lambda path=path: {"path": str(path.resolve()), "sha256": _sha256(path)}
        )
    mamba_path = Path(__file__).with_name("mamba.py")
    row["checkout_mamba_source"] = {
        "path": str(mamba_path),
        "sha256": _observe(lambda: _sha256(mamba_path)),
    }
    return row


def _emit(row, event):
    row["event"] = event
    metrics = os.environ.get("METRICS_DATA_FILE")
    if metrics:
        try:
            directory = Path(metrics) / "npu_pin_diagnostics"
            directory.mkdir(parents=True, exist_ok=True)
            output = (
                directory
                / f"{event}-{row['hostname']}-{row['pid']}-{row['time_ns']}.json"
            )
            row["artifact_path"] = str(output)
            output.write_text(json.dumps(row, indent=2))
        except (OSError, TypeError, ValueError) as exc:
            row["artifact_error"] = f"{type(exc).__name__}: {exc}"
    print("NPU_PIN_DIAGNOSTIC " + json.dumps(row), file=sys.stderr, flush=True)
    return row


def _read_known_branch_cache(library_path):
    """Read a verified ELF's static cache, without invoking C++ through ctypes."""
    if (
        _sha256(library_path)
        != "6557790a35e6cabf6300e91b8eda3944bc30d8ac834fff48161cf72d0c6e6b39"
    ):
        return {"available": False, "reason": "unrecognized torch_npu ELF"}
    library = ctypes.CDLL(str(library_path))
    symbol = library._ZN6at_npu6native29CachingHostAllocator_isPinnedEPv
    address = ctypes.cast(symbol, ctypes.c_void_p).value
    # Confirm the call site as well as the file hash before reading known offsets.
    if int.from_bytes(ctypes.string_at(address + 0x34, 4), "little") != 0x9409ABE0:
        return {"available": False, "reason": "unexpected function address"}
    base = address - 0x1706D50
    guard = ctypes.c_uint64.from_address(base + 0x4E22398).value
    return {
        "available": True,
        "guard": guard,
        "AclrtPointerGetAttributesExist": (
            ctypes.c_bool.from_address(base + 0x4E223A0).value if guard & 1 else None
        ),
    }


def _pointer_attributes(pointer):
    class Location(ctypes.Structure):
        _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int)]

    class Attributes(ctypes.Structure):
        _fields_ = [
            ("location", Location),
            ("page_size", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32 * 4),
        ]

    library = ctypes.CDLL("libascendcl.so")
    query = library.aclrtPointerGetAttributes
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Attributes)]
    query.restype = ctypes.c_int
    attributes = Attributes()
    ret = query(pointer, ctypes.byref(attributes))
    return {
        "ret": ret,
        "location_type": attributes.location.type if ret == 0 else None,
        "location_id": attributes.location.id if ret == 0 else None,
    }


def dump_mamba_pin_failure(host, device, error):
    """Observe the original failed buffer; never register or replace it."""
    import torch
    import torch_npu

    data_ptr = host.data_ptr()
    storage_ptr = host.untyped_storage().data_ptr()
    row = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "error": str(error),
        "npu_python_initialized": torch.npu.is_initialized(),
        "host": {
            "data_ptr": hex(data_ptr),
            "storage_ptr": hex(storage_ptr),
            "storage_offset": host.storage_offset(),
            "shape": list(host.shape),
            "stride": list(host.stride()),
            "dtype": str(host.dtype),
            "bytes": host.numel() * host.element_size(),
            "contiguous": host.is_contiguous(),
        },
        "device": {
            "device": str(device.device),
            "shape": list(device.shape),
            "stride": list(device.stride()),
            "dtype": str(device.dtype),
        },
        "tls_include": str(torch._C._dispatch_tls_local_include_set()),
        "tls_exclude": str(torch._C._dispatch_tls_local_exclude_set()),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS",
                "SGLANG_HICACHE_HOST_MEM_BACKEND",
                "SGLANG_NPU_HICACHE_MAMBA_IO",
                "PYTORCH_NPU_ALLOC_CONF",
            )
        },
    }
    # Capture the cache before issuing new pinned/version queries. Unknown ELF
    # builds still produce all other metadata; no offsets are read for them.
    library_path = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
    row["pinned_branch_cache_before_queries"] = _observe(
        lambda: _read_known_branch_cache(library_path)
    )
    registry = sys.modules.get(
        "sgl_kernel_npu.operators.mem_cache.memfabric.host_memory"
    )
    row["matching_registrations"] = []
    for pointer, length, mapping in getattr(registry, "_G_MAPPINGS", ()):
        if (
            pointer <= data_ptr < pointer + length
            or pointer <= storage_ptr < pointer + length
        ):
            row["matching_registrations"].append(
                {
                    "base": hex(pointer),
                    "bytes": length,
                    "covers_data": pointer <= data_ptr < pointer + length,
                    "covers_storage": pointer <= storage_ptr < pointer + length,
                    "kind": getattr(mapping, "kind", None),
                    "owned": getattr(mapping, "owned", None),
                }
            )
    acl_rt = sys.modules.get("acl.rt")
    if acl_rt is not None:
        row["acl_device_before_queries"] = _observe(acl_rt.get_device)
        row["acl_context_before_queries"] = _observe(acl_rt.get_context)
    row["acl_attributes_data"] = _observe(lambda: _pointer_attributes(data_ptr))
    row["acl_attributes_storage"] = _observe(lambda: _pointer_attributes(storage_ptr))
    row["python_is_pinned"] = _observe(host.is_pinned)
    row["aten_is_pinned"] = _observe(lambda: torch.ops.aten.is_pinned.default(host))
    row["cann_versions_seen_by_torch_npu"] = {
        key: _observe(lambda key=key: torch_npu._C._get_cann_version(key))
        for key in ("CANN", "DRIVER", "RUNTIME")
    }
    row["aten_dispatch"] = _observe(lambda: torch._C._dispatch_dump("aten::is_pinned"))
    row["mamba_dispatch"] = _observe(
        lambda: torch._C._dispatch_dump("npu::transfer_mamba_state")
    )
    row["mapped_libraries"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(
                name in line
                for name in (
                    "libtorch_npu.so",
                    "libtorch_cpu.so",
                    "libascendcl.so",
                    "libruntime.so",
                    "libsgl_kernel_npu",
                )
            )
        }
    )
    row["loaded_library_hashes"] = {
        path: _observe(lambda path=path: _sha256(path))
        for path in row["mapped_libraries"]
        if "libtorch_cpu.so" not in path
    }
    mamba = sys.modules.get("sglang.srt.mem_cache.pool_host.mamba")
    source = getattr(mamba, "__file__", None)
    row["loaded_mamba_source"] = {
        "path": source,
        "sha256": _observe(lambda: _sha256(source)),
    }
    row["environment_metadata"] = collect_environment_metadata()
    return _emit(row, "mamba-pin-failure")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", action="store_true", required=True)
    parser.parse_args()
    _emit(collect_environment_metadata(), "environment")
