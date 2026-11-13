"""gfx950 BF16/FP16 GEMM core vendored from ``tlx.ops.mm`` in facebookexperimental/triton.

The kernels import ``triton.language.extra.tlx``. fbtriton ships it; on stock
ROCm Triton, importing ``utlx_plugin`` from the uTLX wheel registers it
(README.md lists a working pairing). Importing this package loads no kernel;
``_runtime.load_kernel`` imports one on first use, after TLX is registered.
README.md covers provenance and re-sync.
"""
