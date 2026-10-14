"""Build the Few service primitives with their original CUDA arithmetic flags."""

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import re


_SOURCES = {
    'caller': 'caller.cu',
    'post': 'post.cu',
    'planner': 'planner.cu',
    'direct_abs': 'direct_abs.cu',
    'sample': 'sample.cu',
    'h16': 'low.cu',
    'h18': 'low.cu',
    'audit': 'high.cu',
    'common': 'compact.cu',
    'exact': 'refine.cu',
    'seed': 'seed.cu',
    'integer': 'select.cu',
    'publisher': 'recovery.cu',
    'leaf': 'recovery.cu',
    'hierarchy': 'recovery.cu',
}


def source_closure(source):
    """Local includes participate in the ordinary compiler artifact cache key."""
    pending, files = [Path(source).resolve()], {}
    while pending:
        path = pending.pop()
        if path in files:
            continue
        data = path.read_bytes()
        files[path] = data
        pending.extend((path.parent / name).resolve() for name in
                       re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', data.decode(), re.M))
    return files


@lru_cache(None)
def load_module(role, score_heads=32):
    if role == "h18":
        return load_module("h16", score_heads)
    if role in ("publisher", "hierarchy"):
        return load_module("leaf", score_heads)
    from torch.utils.cpp_extension import CUDA_HOME, load

    if score_heads not in (32, 64):
        raise ValueError("DoSA supports32 or64 score heads")
    from sglang.kernels.jit.utils import KERNEL_PATH

    root = KERNEL_PATH / "csrc/attention/dosa"
    source = root / _SOURCES[role]
    package = "flashinfer" if role == "direct_abs" else "deep_gemm"
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None or CUDA_HOME is None:
        raise RuntimeError("DoSA requires CUDA sm_100a and installed DeepGEMM headers")
    include = Path(spec.origin).parent / (
        "data/cutlass/include" if role == "direct_abs" else "include")
    flags = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20",
             "-gencode=arch=compute_100a,code=sm_100a", f"-DDOSA_SCORE_HEADS={score_heads}"]
    if role in ("h16", "h18", "audit", "common", "exact", "direct_abs"):
        flags.append("--ftz=false")
    if role in ("h16", "h18", "direct_abs"):
        flags.append("--fmad=false")
    if role in ("h16", "h18"):
        flags.append("--expt-extended-lambda")
    if role in ("planner", "direct_abs"):
        flags.append("-DDOSA_NEGATIVE_FILL_MINORITY=1")
    digest = hashlib.sha256(repr(flags).encode())
    for path, data in sorted(source_closure(source).items()):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + data)
    return load(
        name=f"sglang_dosa_{role}_h{score_heads}_{digest.hexdigest()[:12]}",
        sources=[str(source)], extra_include_paths=[str(include)],
        extra_cflags=["-std=c++20"], extra_cuda_cflags=flags,
        extra_ldflags=[f"-L{CUDA_HOME}/lib64/stubs", "-lcuda"], with_cuda=True,
    )


def load_modules(score_heads=32):
    """Bind one explicit service ABI before allocating the model's KV pool."""
    return {role: load_module(role, score_heads) for role in _SOURCES}
