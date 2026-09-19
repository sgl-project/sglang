from pathlib import Path

from torch.utils.cpp_extension import load

_ABS_PATH = Path(__file__).resolve().parent
_JIT_CSRC_PATH = _ABS_PATH.parents[1] / "jit_kernel" / "csrc" / "eplb"
_SOURCES = [(_JIT_CSRC_PATH / "eplb_expert_location.cpp").resolve()]

_eplb_expert_location_cpp = None
_eplb_expert_location_cpp_error = None


def _load_eplb_expert_location_cpp():
    global _eplb_expert_location_cpp, _eplb_expert_location_cpp_error
    if _eplb_expert_location_cpp is not None:
        return _eplb_expert_location_cpp
    if _eplb_expert_location_cpp_error is not None:
        raise RuntimeError(_eplb_expert_location_cpp_error)

    missing_sources = [str(path) for path in _SOURCES if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Missing EPLB expert location sources: " + ", ".join(missing_sources)
        )

    try:
        _eplb_expert_location_cpp = load(
            name="eplb_expert_location_cpp",
            sources=[str(path) for path in _SOURCES],
            extra_cflags=["-O3", "-std=c++17"],
            with_cuda=False,
        )
    except Exception as exc:
        _eplb_expert_location_cpp_error = str(exc)
        raise
    return _eplb_expert_location_cpp


def compute_logical_to_rank_dispatch_physical_map_cpp(
    logical_to_all_physical_map_cpu,
    *,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    num_gpus_per_node: int,
    seed: int,
):
    return (
        _load_eplb_expert_location_cpp().compute_logical_to_rank_dispatch_physical_map(
            logical_to_all_physical_map_cpu,
            ep_size,
            num_physical_experts,
            ep_rank,
            num_gpus_per_node,
            seed,
        )
    )


def warmup_eplb_expert_location_cpp():
    _load_eplb_expert_location_cpp()
