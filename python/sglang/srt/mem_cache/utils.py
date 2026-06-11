# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
mem_cache 模块的通用工具函数。

本文件提供：
  1. 淘汰策略工厂    — get_eviction_strategy()
  2. 自定义内存池初始化 — maybe_init_custom_mem_pool()
  3. Token 序列哈希   — get_hash_str() / hash_str_to_int64()
  4. Radix Tree 节点哈希 — compute_node_hash_values() / split_node_hash_value()
"""

import hashlib
from typing import Any, Callable, List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    get_mla_kv_buffer_kernel as get_mla_kv_buffer_kernel,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    get_mla_kv_buffer_triton as get_mla_kv_buffer_triton,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    set_mla_kv_buffer_fp8_quant_kernel as set_mla_kv_buffer_fp8_quant_kernel,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    set_mla_kv_buffer_kernel as set_mla_kv_buffer_kernel,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    set_mla_kv_buffer_triton as set_mla_kv_buffer_triton,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    set_mla_kv_buffer_triton_fp8_quant as set_mla_kv_buffer_triton_fp8_quant,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    set_mla_kv_scale_buffer_kernel as set_mla_kv_scale_buffer_kernel,
)
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    set_mla_kv_scale_buffer_triton as set_mla_kv_scale_buffer_triton,
)

# ============================================================================
# 淘汰策略工厂注册表
# ============================================================================
# 字符串名 → 策略类构造器（无参 callable）。
# 使用工厂函数而非直接暴露类，方便后续加入初始化逻辑。
_EVICTION_POLICY_FACTORIES: dict[str, Callable[[], EvictionStrategy]] = {
    "lru": LRUStrategy,
    "lfu": LFUStrategy,
    "fifo": FIFOStrategy,
    "mru": MRUStrategy,
    "filo": FILOStrategy,
    "priority": PriorityStrategy,
    "slru": SLRUStrategy,
}


def get_eviction_strategy(eviction_policy: str) -> EvictionStrategy:
    """
    根据策略名（大小写不敏感）创建对应的淘汰策略实例。

    支持 7 种策略：
      lru, lfu, fifo, mru, filo, priority, slru

    参数：
        eviction_policy: 策略名字符串，如 "lru"、"LRU"。

    返回：
        EvictionStrategy 子类实例。

    抛出：
        ValueError: 当策略名不在已知列表中时。
    """
    policy = eviction_policy.lower()
    try:
        return _EVICTION_POLICY_FACTORIES[policy]()
    except KeyError:
        supported = "', '".join(_EVICTION_POLICY_FACTORIES)
        raise ValueError(
            f"Unknown eviction policy: {policy}. Supported policies: '{supported}'."
        ) from None


def maybe_init_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    尝试初始化自定义内存池（当前仅支持 Mooncake PD 分离场景）。

    当环境变量 SGLANG_MOONCAKE_CUSTOM_MEM_POOL 被设置时，
    调用 Mooncake 的自定义内存池初始化逻辑。

    参数：
        device: 目标设备字符串（如 "cuda:0"）。

    返回：
        (是否启用, 内存池对象或None, 内存池类型字符串或None)。
    """
    enable_custom_mem_pool = (
        True if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None else False
    )

    if enable_custom_mem_pool:
        # 当前仅 Mooncake 需要自定义内存池（MNNVL/Barex PD 分离场景）
        from sglang.srt.disaggregation.mooncake.utils import (
            init_mooncake_custom_mem_pool,
        )

        return init_mooncake_custom_mem_pool(device)
    else:
        return False, None, None


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    """
    将 token ID 列表转换为确定性的 SHA256 哈希字符串。

    支持两种 token 编码模式：
      - 常规模式：token 是 int，每个 token 的 4 字节小端表示写入 hasher。
      - EAGLE bigram 模式：token 是 tuple(int, int)，tuple 内每个元素
        都被写入 hasher，用于唯一标识一个 draft→target token 对。

    如果提供了 prior_hash，会先把它（hex bytes）写入 hasher，
    实现「链式哈希」——相当于在旧哈希的基础上继续哈希新 token。

    参数：
        token_ids:   token ID 列表，元素可以是 int 或 tuple(int, int)。
        prior_hash:  可选的先前 hash 值（SHA256 hex string），用于链式哈希。

    返回：
        SHA256 哈希值的十六进制字符串（64 个字符）。
    """
    hasher = hashlib.sha256()

    # 先写入旧 hash（如果有），实现哈希链
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram 模式：(draft_token, target_token)
            # 把两个元素都写入 hasher，唯一标识这个 token 对
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # 常规模式：单个 int token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """
    将 SHA256 hex 字符串转换为 signed int64。

    取 hex 字符串的前 16 个字符（64 bits），
    如果无符号值 >= 2^63，则减去 2^64 转为有符号负数。

    用途：将缓存 hash 压缩为 int64，用于 event / tracing 等场景。

    参数：
        hash_str: get_hash_str() 返回的 64 字符 hex 字符串。

    返回：
        [-2^63, 2^63-1] 范围内的有符号整数。
    """
    # 取前 16 hex 字符 = 64 bits
    uint64_val = int(hash_str[:16], 16)
    # 无符号 → 有符号转换
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


def compute_node_hash_values(node: Any, page_size: int) -> List[str]:
    """
    为 radix tree 节点计算分页 SHA256 hash 值列表。

    每个 page 对应一个 hash 值。hash 是链式的：
    当前 page 的 hash 依赖于上一 page 的 hash（或 parent 的最后一个 hash）。

    这种设计保证了：
      - 同一个 token 序列（在不同请求中）产生相同的 hash 值
      - hash 顺序敏感（因为 SHA256 链接了前后 page）

    参数：
        node:      radix tree 节点（需要有 key、parent、hash_value 等属性）。
        page_size: 每个 page 覆盖的 token 数。

    返回：
        hash 字符串列表，每个 page 一个。
    """
    hash_values = []

    # 获取 parent 最后一个 hash 作为初始种子
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # 按 page 粒度计算 hash
    logical_len = len(node.key)
    for start in range(0, logical_len, page_size):
        end = min(start + page_size, logical_len)
        if end <= start:
            continue
        hash_val = node.key.hash_page(start, end, parent_hash)
        hash_values.append(hash_val)
        # 当前 page 的 hash 成为下一个 page 的种子
        parent_hash = hash_val
    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """
    在 radix tree 节点分裂时，将 hash_value 拆分为 parent 和 child 两部分。

    节点分裂示意图：
        分裂前：  [page0, page1, page2, page3]  ← child 的全部 hash
        分裂后：  parent 取走前 N 个 page → [page0, page1]
                 child 保留剩余        → [page2, page3]

    参数：
        child_hash_value: 原 child 节点的 hash 列表（可能为 None）。
        split_len:        分裂点的 token 位置。
        page_size:        每页的 token 数。

    返回：
        (new_node_hash_value, updated_child_hash_value)
        两者都是 list 或 None。
    """
    if child_hash_value is None:
        return None, None

    # 将 split_len (token 粒度) 转换为 page 粒度
    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    # 按 page 索引切片
    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash
