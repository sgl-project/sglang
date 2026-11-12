"""Rebuild FlashInfer's TRT-LLM MoE routing modules with the K3.1 DynBlock header.

patch_flashinfer.py raises DynBlockKernelMaxNumExperts to 1024, but the prebuilt
flashinfer-jit-cache modules were compiled with 512. This rebuilds every jit-cache copy of
the modules below from the patched header, for its provider's architecture, and replaces
it in place, so launches need no runtime switch and no on-site compile. Next to each module
it writes MARKER (see verify_image.py), recording what it was built from.

Each (module, architecture) builds in its own process, all in parallel: FlashInfer fixes
the target architectures from FLASHINFER_CUDA_ARCH_LIST when it is imported.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

MODULES = {
    "fused_moe_trtllm_sm100": "gen_trtllm_gen_fused_moe_sm100_module",
    "trtllm_gen_routing": "gen_trtllm_gen_routing_module",
}
MARKER = "k31-dynblock.json"
HEADER = (
    Path(sysconfig.get_paths()["purelib"])
    / "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
)

_BUILD_ONE = """
import shutil, sys
from flashinfer.jit import fused_moe
spec = getattr(fused_moe, sys.argv[1])()
spec.build(verbose=False)
shutil.copyfile(spec.jit_library_path, sys.argv[2])
"""


def header_sha256() -> str:
    return hashlib.sha256(HEADER.read_bytes()).hexdigest()


def arch_of(provider_id: str) -> str:
    # FLASHINFER_CUDA_ARCH_LIST spelling: "sm103a" -> "10.3a"
    major, minor, suffix = re.fullmatch(r"sm(\d+)(\d)([af]?)", provider_id).groups()
    return f"{major}.{minor}{suffix}"


def main() -> None:
    import flashinfer_jit_cache

    if "DynBlockKernelMaxNumExperts = 1024;" not in HEADER.read_text():
        sys.exit(f"{HEADER} is not patched; run patch_flashinfer.py first")
    marker = {
        "DynBlockKernelMaxNumExperts": 1024,
        "flashinfer": metadata.version("flashinfer-python"),
        "header_sha256": header_sha256(),
    }
    jobs = [
        (
            Path(provider.jit_cache_dir) / module / f"{module}.so",
            generator,
            arch_of(provider.provider_id),
        )
        for provider in flashinfer_jit_cache.get_jit_cache_providers()
        for module, generator in MODULES.items()
        if (Path(provider.jit_cache_dir) / module / f"{module}.so").exists()
    ]
    if not jobs:
        sys.exit(f"no jit-cache copy of {sorted(MODULES)} found")
    # Each build is dominated by a few large translation units, so they run side by side,
    # sharing the cores between them.
    jobs_per_build = str(max(1, (os.cpu_count() or 1) // len(jobs)))

    def build(job):
        target, generator, arch = job
        with tempfile.TemporaryDirectory() as workspace:
            env = dict(
                os.environ,
                FLASHINFER_CUDA_ARCH_LIST=arch,
                FLASHINFER_WORKSPACE_BASE=workspace,
                MAX_JOBS=jobs_per_build,
            )
            built = Path(workspace) / target.name
            subprocess.run(
                [sys.executable, "-c", _BUILD_ONE, generator, str(built)],
                env=env,
                check=True,
            )
            shutil.copyfile(built, target)
        (target.parent / MARKER).write_text(
            json.dumps({**marker, "arch": arch}, indent=1)
        )
        return f"rebuilt {target} for {arch}"

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for line in pool.map(build, jobs):
            print(line)


if __name__ == "__main__":
    main()
