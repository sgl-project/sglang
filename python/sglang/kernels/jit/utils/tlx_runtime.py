"""Runtime support for the vendored TLX kernels (``sglang.kernels.tlx_ops``) on stock ROCm Triton.

The kernels import ``triton.language.extra.tlx``. fbtriton ships it. On stock ROCm
Triton, the uTLX plugin built for that Triton release registers it at interpreter
start through its ``.pth`` hook. Verified on MI355X:

    torch 2.14.1+rocm7.2, triton-rocm 3.8.0:   pip install --no-deps triton-utlx==3.8.0.post16
    torch 2.15.0+rocm7.14, triton-rocm 3.9.0:  pip install --no-deps triton-utlx==3.9.0.post4

``--no-deps`` because those wheels declare the CUDA ``triton`` package.
``triton-utlx-rocm`` builds do not work with PyTorch's ``triton-rocm``: 3.8.0.post1
fails at the first compile on a missing ``mlir::argSort``. With torch 2.15 and
``triton-utlx 3.9.0.post4``, loading uTLX before torch aborts with ``LLVM ERROR:
Option 'default' already exists!``; SGLang always imports torch first, and standalone
scripts must too.
"""

import functools

import triton


@functools.cache
def add_missing_triton_knobs() -> None:
    """Define fbtriton-only instrumentation knobs as False on stock Triton; a no-op on fbtriton.

    tlx.ops kernels read them before their compiled-cache fast launch, and stock Triton
    has none of that instrumentation.
    """
    for group, name in (
        ("runtime", "sanitize_overflow"),
        ("compilation", "fpsan_homomorphic_casts"),
    ):
        knobs = getattr(triton.knobs, group)
        if not hasattr(knobs, name):
            # On the class: knobs.scope() resets instance dicts.
            setattr(type(knobs), name, False)
