"""Launch one Beam rank, or print its complete command without GPU work."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
from pathlib import Path

from curve_config import (
    ENGINE_COMMIT,
    MODEL_REVISION,
    SIMULATION,
    inside_repo,
    server_args,
)


def command(settings, layout, cc, rank):
    repo = Path(__file__).resolve().parents[2]
    runtime = inside_repo(settings["work_dir"])
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SGLANG_", "NCCL_", "DUMPER_"))
    }
    env.update(SIMULATION)
    env["PATH"] = (
        str(Path(settings["server_python"]).parent) + os.pathsep + env.get("PATH", "")
    )
    env.update(
        PYTHONPATH=os.pathsep.join(
            [*settings.get("runtime_pythonpaths", []), str(repo / "python")]
        ),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        NCCL_MNNVL_ENABLE="1" if layout.endswith("8") else "0",
        NCCL_CUMEM_ENABLE="1",
        NCCL_NVLS_ENABLE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        MAX_JOBS="2",
        CARGO_BUILD_JOBS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        CUDA_ENABLE_COREDUMP_ON_EXCEPTION="0",
        CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION="0",
    )
    if layout.endswith("4"):
        # This single-node host exposes IB IPv6 addresses in VRFs; NCCL cannot
        # bind its auto-selected address. Bootstrap over local IPv4 instead.
        env.update(NCCL_SOCKET_IFNAME="lo", NCCL_SOCKET_FAMILY="AF_INET")
    for k, rel in {
        "HF_HOME": "hf",
        "HF_HUB_CACHE": "hf/hub",
        "HUGGINGFACE_HUB_CACHE": "hf/hub",
        "HF_DATASETS_CACHE": "hf/datasets",
        "XDG_CACHE_HOME": "cache",
        "TMPDIR": "tmp",
        "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "FLASHINFER_WORKSPACE_BASE": "cache/flashinfer",
        "SGLANG_CACHE_DIR": "cache/sglang",
        "SGLANG_JIT_CACHE_DIR": "cache/sglang/jit",
        "SGLANG_CUTE_AOT_CACHE_DIR": "cache/sglang/cute_aot",
        "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "CUDA_CACHE_PATH": "cache/cuda",
        "DEEPGEMM_JIT_CACHE_DIR": "cache/deepgemm",
        "PYTHONPYCACHEPREFIX": "cache/pycache",
    }.items():
        env[k] = str(runtime / rel)
    env.pop("TRANSFORMERS_CACHE", None)
    argv = [
        settings["server_python"],
        "-B",
        "-m",
        "sglang.launch_server",
        *server_args(
            layout,
            cc,
            rank,
            settings["model_path"],
            settings["draft_path"],
            settings["head_addr"],
            settings.get("port", 30080),
            settings.get("dist_port", 39746),
            hicache_size=settings.get("hicache_size", 88),
        ),
    ]
    return argv, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("settings")
    p.add_argument("layout", choices=["tp8", "tep8", "tp4", "tep4"])
    p.add_argument("cc", type=int)
    p.add_argument("rank", type=int, choices=[0, 1])
    p.add_argument("--print", action="store_true", dest="print_only")
    a = p.parse_args()
    settings = json.loads(Path(a.settings).read_text())
    argv, env = command(settings, a.layout, a.cc, a.rank)
    if a.print_only:
        print(json.dumps({"argv": argv, "simulation": SIMULATION}, indent=2))
        return
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", ENGINE_COMMIT, "--", "python"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for key in ("model_path", "draft_path"):
        if not Path(settings[key]).is_dir():
            raise ValueError(
                f"{key} must be an existing local checkpoint; downloads are disabled"
            )
    for key in ("HF_HOME", "TMPDIR", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    model = Path(settings["model_path"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    for name in ["config.json", "tokenizer.json", *set(index["weight_map"].values())]:
        if not (model / name).is_file():
            raise ValueError(f"missing checkpoint file {name}")
        metadata = model / ".cache/huggingface/download" / (name + ".metadata")
        if (
            not metadata.is_file()
            or metadata.read_text().splitlines()[0] != MODEL_REVISION
        ):
            raise ValueError(f"checkpoint revision metadata missing/mismatched: {name}")
    draft_hash = hashlib.sha256()
    with (Path(settings["draft_path"]) / "model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            draft_hash.update(chunk)
    if (
        draft_hash.hexdigest()
        != "34b99ca539b1d595d0db0edf072158173d89411eca0e6b1dd2c2601437352bf4"
    ):
        raise ValueError("released DFlash2 weight checksum differs")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(json.dumps({"argv": argv, "simulation": SIMULATION}), flush=True)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
