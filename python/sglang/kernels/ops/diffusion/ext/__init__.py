"""JIT-built C++/CUDA extensions that are not kernels.

Mesh rasterization and texture inpainting for Hunyuan3D: no backend dimension,
no numerical contract, not in the kernel registry.  Kept beside the diffusion
kernels because they share the JIT build/recovery machinery in
:mod:`.loader`.
"""
