"""Gluon kernels for GLM Quark MXFP4 MoE."""

from triton.experimental.gluon import language as gl

# Triton before 3.9 only has the private _reinterpret.
if not hasattr(gl.shared_memory_descriptor, "reinterpret"):
    gl.shared_memory_descriptor.reinterpret = gl.shared_memory_descriptor._reinterpret
