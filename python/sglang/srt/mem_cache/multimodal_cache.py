"""
多模态（图片/视频/音频）Embedding 的服务器级缓存。

当 VLM 接收到一张图片时，需要先用视觉编码器把图片编码成 tensor（embedding）。
这个编码过程很昂贵，通常需要 50-500ms。如果同一张图片出现在多个请求中，
就可以跳过重复编码，直接从缓存取。

本模块提供两层抽象：
  - MultimodalCache（抽象基类）：定义缓存接口
  - MultiModalStaticCache：基于 OrderedDict 的 LRU 缓存，字节数超限时淘汰最久未用的条目

使用示例：
  1. 用户发来一张 "cat.jpg"
  2. 计算 hash("cat.jpg") → 12345
  3. cache.set(12345, embedding) 存进去
  4. 另一个请求也带 "cat.jpg" → cache.get([12345]) 命中 → 跳过编码
  5. 缓存满了之后，最久没被访问的条目自动淘汰
"""

import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


class MultimodalCache(abc.ABC):
    """多模态 embedding 缓存的抽象接口。子类自行决定存储策略（内存/磁盘/分布式）。"""

    @abc.abstractmethod
    def __init__(self): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """把多张图片各自的 hash 合并成一个组合 key。

        hash 顺序是敏感的： [A, B] 和 [B, A] 算出来的 key 不同，
        这保证了请求中图片的排列顺序会影响缓存命中。

        空列表返回 None（没东西可缓存）。
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """根据一组图片 hash 查找缓存。

        内部调用 combine_hashes 把列表合并成一个 key。
        不支持单张图片 fallback 到多张 —— 如果你只需要查单张，用 get_single()。
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """把预计算好的 embedding 存到指定 hash 下。

        返回 True 表示存入成功（可能触发淘汰），False 表示这个 embedding 比 max_size
        还大，即使是空缓存也存不下。
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        """检查给定 hash 的 embedding 是否在缓存中。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        """从缓存中删除一个条目。不存在则返回 False。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        """清空所有缓存条目。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        """返回当前缓存中的条目数量。"""
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor) -> int:
    """计算 tensor 占用的字节数（element_size × 元素个数）。"""
    return embedding.element_size() * embedding.numel()


@dataclass(kw_only=True)
class EmbeddingResult:
    """包装一个缓存的 embedding tensor。

    用 dataclass 而非裸 tensor，这样类型系统和序列化层能区分
    "这是缓存过的 embedding" 和 "这是一个普通 tensor"。
    """
    embedding: torch.Tensor


class MultiModalStaticCache(MultimodalCache):
    """基于内存的 LRU 多模态 embedding 缓存。

    容量按**字节数**计算。当新 embedding 会导致总量超过 max_size 时，
    从最久未使用的条目开始逐个淘汰，直到腾出足够空间。

    "Static" 指的是 embedding 在外部计算好后存进来，缓存本身不管理
    GPU 显存池（不是 pre-allocated 的）。

    用法::

        cache = MultiModalStaticCache(max_size=100 * 1024 * 1024)  # 100 MB
        emb = encode_image(img)               # 返回 EmbeddingResult
        cache.set(hash("cat.jpg"), emb)       # 存入
        result = cache.get([hash("cat.jpg")]) # 命中
        result = cache.get([hash("unknown")]) # 未命中 → None
    """

    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size
        # OrderedDict 记住插入顺序，配合 move_to_end() 实现 LRU：
        # 最近访问的条目被移到末尾，最久未用的在开头
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size: int = 0  # 当前已缓存的字节数

    # ------------------------------------------------------------------
    # 读路径
    # ------------------------------------------------------------------

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        """合并请求中所有图片的 hash 后查找。

        只支持组合 hash 查找，不会降级到单张查找。
        需要单张图片查询请用 get_single()。
        """
        combined_hash = self.combine_hashes(mm_hashes)

        embedding = self.mm_cache.get(combined_hash)
        if embedding is not None:
            # 移到 OrderedDict 末尾 → 标记为刚使用过（LRU 提升）
            self.mm_cache.move_to_end(combined_hash)
        return embedding

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """直接用单个 hash 查找（不走 combine_hashes）。

        适用于已知精确 hash、想跳过组合步骤的场景。
        """
        embedding = self.mm_cache.get(mm_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(mm_hash)
        return embedding

    # ------------------------------------------------------------------
    # 写路径
    # ------------------------------------------------------------------

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        """存入一个 embedding。空间不够时自动淘汰 LRU 条目。

        参数:
            mm_hash: 组合 hash（或单图片 hash），作为缓存 key。
            embedding: 预计算好的 embedding。
            loc: 未使用，保留仅为接口兼容。

        返回:
            True 表示存入成功。False 表示这个 embedding 比 max_size 还大，
            即使清空整个缓存也放不下。
        """
        assert isinstance(embedding, EmbeddingResult), embedding

        # 已经在缓存中？只需要更新时间戳（移到末尾）
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True

        data_size = _get_tensor_size(embedding.embedding)

        # 空间不够 → 从最旧的开始淘汰，直到能放下
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                return False  # 空缓存都放不下，直接放弃
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding.embedding)

        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def has(self, mm_hash: int) -> bool:
        """检查是否存在（不影响 LRU 顺序）。"""
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        """按 hash 删除一个缓存条目。不存在则返回 False。"""
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding.embedding)
        return True

    def clear(self):
        """清空所有条目。"""
        self.mm_cache.clear()
        self.current_size = 0

    def __len__(self) -> int:
        return len(self.mm_cache)

    def available_size(self) -> int:
        """缓存中的条目数。"""
        return self.__len__()
