from pathlib import Path

from torch.utils.cpp_extension import load

_ABS_PATH = Path(__file__).resolve().parent
_JIT_CSRC_PATH = _ABS_PATH.parents[1] / "jit_kernel" / "csrc" / "eplb"
_SOURCES = [(_JIT_CSRC_PATH / "eplb_deepseek.cpp").resolve()]

_eplb_deepseek_cpp = None
_eplb_deepseek_cpp_error = None


def _load_eplb_deepseek_cpp():
    global _eplb_deepseek_cpp, _eplb_deepseek_cpp_error
    if _eplb_deepseek_cpp is not None:
        return _eplb_deepseek_cpp
    if _eplb_deepseek_cpp_error is not None:
        raise RuntimeError(_eplb_deepseek_cpp_error)

    missing_sources = [str(path) for path in _SOURCES if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Missing EPLB deepseek sources: " + ", ".join(missing_sources)
        )

    try:
        _eplb_deepseek_cpp = load(
            name="eplb_deepseek_cpp",
            sources=[str(path) for path in _SOURCES],
            extra_cflags=["-O3", "-std=c++17"],
            with_cuda=False,
        )
    except Exception as exc:
        _eplb_deepseek_cpp_error = str(exc)
        raise
    return _eplb_deepseek_cpp


def rebalance_experts_cpp(
    *,
    weight,
    num_replicas: int,
    num_groups: int | None,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
):
    if enable_hierarchical:
        if num_groups is None:
            raise ValueError(
                "num_groups must not be None when hierarchical EPLB is enabled"
            )
        cpp_num_groups = num_groups
    else:
        cpp_num_groups = 1 if num_groups is None else num_groups

    return _load_eplb_deepseek_cpp().rebalance_experts(
        weight,
        num_replicas,
        cpp_num_groups,
        num_nodes,
        num_gpus,
        enable_hierarchical,
    )


def warmup_eplb_deepseek_cpp():
    _load_eplb_deepseek_cpp()
