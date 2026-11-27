"""MiniMax-M3.1-Flash low-batch FFN (1-32 decode rows) in CuTe DSL: fused
routing, NVFP4 routed experts and an MXFP8 shared expert. Entry points are in
``ffn``; this package import stays metadata-only (no cutlass import, no
compilation)."""
