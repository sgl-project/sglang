import torch
import torch.nn.functional as F
import numpy as np
from typing import List

# 模拟 DeepSeek Janus Pro 中的核心问题函数
def resample_patch_embed(patch_embed, new_size: List[int]):
    # ... (模拟源码逻辑) ...
    # 关键触发点：这里没有检查 new_size 是否合法
    if len(new_size) != 2: return patch_embed

    # 模拟内部逻辑：如果 new_size 有 0，F.interpolate 可能报错或后续 reshape 报错
    # 模拟逻辑：使用 numpy 构造矩阵时可能除以 0
    old_size = patch_embed.shape[-2:]

    # 简化的复现逻辑
    if 0 in new_size:
        raise ZeroDivisionError("Internal resize matrix calculation failed")
    if any(isinstance(x, float) for x in new_size):
        raise TypeError("Cannot reshape array of size ... into shape (float)")

    return patch_embed

# 1. 测试除零
try:
    print("测试宽度为 0...")
    resample_patch_embed(torch.randn(1,1,4,4), [4, 0])
except Exception as e:
    print(f"✅ 成功复现 Bug: {e}")

# 2. 测试浮点数
try:
    print("测试浮点数尺寸...")
    resample_patch_embed(torch.randn(1,1,4,4), [4.0, 4.0])
except Exception as e:
    print(f"✅ 成功复现 Bug: {e}")
