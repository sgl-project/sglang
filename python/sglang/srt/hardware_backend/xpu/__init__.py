# XPU (Intel GPU) hardware backend

from sglang.srt.hardware_backend.xpu.xpu_scratchpad_patch import (
    patch_sgl_kernel_scratchpad_buffers,
)

patch_sgl_kernel_scratchpad_buffers()
