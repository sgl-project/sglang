"""Opt-in FP8 -> BF16 temporary KV for eager FlashInfer FA2 paged prefill.

The persistent KV pool remains FP8. This changes the attention kernel's KV
dtype, so exact conversion does not imply identical FP8-kernel outputs.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@dataclass
class PrefillKVDequantPlan:
    wrapper: Any
    pages: torch.Tensor
    page_size: int

    def gather(self, kv_cache):
        if torch.cuda.is_current_stream_capturing():
            return None
        from sglang.kernels.ops.attention.gather_dequant_kv import (
            _page_contiguous,
            gather_dequant_kv,
        )

        if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
            return None
        k, v = kv_cache
        # SGLang's FlashInfer metadata uses one-token pages, even for a
        # multi-token allocator page size. No copy of the entire pool is needed.
        if self.page_size == 1 and k.ndim == v.ndim == 3:
            k, v = k.unsqueeze(1), v.unsqueeze(1)
        if not (
            k.dtype in FP8_DTYPES
            and v.dtype == k.dtype
            and k.is_cuda
            and k.device == v.device == self.pages.device
            and k.shape == v.shape
            and k.ndim == 4
            and k.shape[1] == self.page_size
            and _page_contiguous(k)
            and _page_contiguous(v)
        ):
            return None
        return gather_dequant_kv(k, v, self.pages)


class FlashInferPrefillKVDequant:
    def __init__(
        self,
        wrapper_factory,
        min_tokens=1024,
        min_kv_tokens=16384,
        max_kv_tokens=262144,
    ):
        if min_tokens < 1 or min_kv_tokens < 0 or max_kv_tokens < min_kv_tokens:
            raise ValueError("Invalid prefill KV dequant token thresholds")
        self.wrapper_factory = wrapper_factory
        self.wrapper = None
        self.min_tokens = min_tokens
        self.min_kv_tokens = min_kv_tokens
        self.max_kv_tokens = max_kv_tokens

    def prepare(
        self,
        *,
        num_new_tokens: Optional[int],
        num_pages: int,
        use_ragged: bool,
        spec_info,
        original_wrapper,
        plan_kwargs,
    ):
        """Use host-known counts; never read GPU indptr scalars here.

        The caller enables this helper only for plain FP8, BF16 Q, FA2,
        non-deterministic single-wrapper attention. All unsupported batches
        keep the original wrapper, including the batch following an enabled one.
        """
        if (
            num_new_tokens is None
            or num_new_tokens < self.min_tokens
            or use_ragged
            or spec_info is not None
            or original_wrapper.is_cuda_graph_enabled
            or torch.cuda.is_current_stream_capturing()
            or plan_kwargs.get("custom_mask") is not None
            or plan_kwargs.get("packed_custom_mask") is not None
            or plan_kwargs.get("prefix_len_ptr") is not None
            or plan_kwargs.get("window_left", -1) != -1
            or plan_kwargs["kv_data_type"] not in FP8_DTYPES
            or plan_kwargs["q_data_type"] != torch.bfloat16
        ):
            return None
        page_size = plan_kwargs["page_size"]
        if (
            num_pages <= 0
            or not self.min_kv_tokens <= num_pages * page_size <= self.max_kv_tokens
        ):
            return None
        pages = plan_kwargs["paged_kv_indices"][:num_pages]
        if pages.numel() != num_pages:
            raise ValueError("Host KV page count exceeds the page-index buffer")
        if self.wrapper is None:
            self.wrapper = self.wrapper_factory()
        kwargs = dict(plan_kwargs)
        kwargs["paged_kv_indices"] = torch.arange(
            num_pages, dtype=pages.dtype, device=pages.device
        )
        kwargs["kv_data_type"] = torch.bfloat16
        self.wrapper.begin_forward(**kwargs)
        return PrefillKVDequantPlan(self.wrapper, pages, page_size)
