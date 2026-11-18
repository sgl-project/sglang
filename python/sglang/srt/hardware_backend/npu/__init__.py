# SPDX-License-Identifier: Apache-2.0


def get_torch_npu():
    """Load torch_npu only when an NPU is available."""
    from sglang.srt.utils import is_npu

    if not is_npu():
        raise RuntimeError("torch_npu requires an available NPU")
    import torch_npu

    return torch_npu
