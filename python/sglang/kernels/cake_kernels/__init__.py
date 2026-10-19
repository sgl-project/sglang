"""Lazy adapters for Cake-generated kernels distributed by FlashInfer.

Runtime callers use the corresponding ``sglang.kernels.ops`` facade. Importing
this package does not load FlashInfer, initialize CUDA, or compile kernels.
"""
