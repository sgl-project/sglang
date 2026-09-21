"""Verify packaged source and runtime; does not send serving requests.

Run with the same CUDA_VISIBLE_DEVICES as the worker launcher. The manifest
is required; --gpu-report also checks exact test coverage and zero skips.
"""

import argparse
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def validate_gpu_report(manifest, path):
    expected = {node.split("::", 1)[1] for node in manifest["gpu_test_nodeids"]}
    cases = list(ET.parse(path).getroot().iter("testcase"))
    names = [case.attrib.get("name") for case in cases]
    if not expected or len(names) != len(set(names)) or set(names) != expected:
        raise AssertionError("GPU test inventory is incomplete or duplicated")
    for case in cases:
        if any(case.find(kind) is not None for kind in ("skipped", "failure", "error")):
            raise AssertionError(f"GPU test did not pass: {case.attrib.get('name')}")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path)
    args = parser.parse_args()
    import torch

    import sglang
    from sglang.srt.disaggregation.compression.protocol import (
        PROTOCOL_VERSION,
        REGISTRATION_VERSION,
    )
    from sglang.srt.kv_compression.types import NVCOMP_VERSION

    root = Path(sglang.__file__).resolve().parents[2]
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        for record in manifest["files"]:
            path = root / record["path"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise AssertionError(f"Delivered source mismatch: {path}")
    gpu_passed = (
        validate_gpu_report(manifest, args.gpu_report) if args.gpu_report else None
    )
    assert PROTOCOL_VERSION == 2
    assert REGISTRATION_VERSION == 2
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.device_count() == 1, (
        "Bind the allocated GPU before this preflight"
    )
    cuda_major = torch.version.cuda.split(".")[0]
    packages = [
        "torch",
        "sglang",
        "sglang-kernel",
        "sglang-router",
        "mooncake-transfer-engine-cuda" + cuda_major,
        "nvidia-nvcomp-cu" + cuda_major,
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    assert versions["nvidia-nvcomp-cu" + cuda_major] == NVCOMP_VERSION
    from nvidia import nvcomp

    from sglang.srt.kv_compression.provider import HostEncodedKVProvider
    from sglang.srt.mem_cache.hicache_lifecycle import HiCacheLifecycleMixin
    from sglang.srt.mem_cache.l2_completion import RestoreTicket, TransferCompletion

    assert (
        nvcomp.Codec
        and HiCacheLifecycleMixin
        and RestoreTicket
        and TransferCompletion
        and HostEncodedKVProvider
    )
    print(
        json.dumps(
            dict(
                source=str(root),
                protocol=PROTOCOL_VERSION,
                registration=REGISTRATION_VERSION,
                versions=versions,
                gpu=torch.cuda.get_device_name(0),
                source_verified=bool(args.manifest),
                gpu_tests_passed=gpu_passed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
