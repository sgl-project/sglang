"""Paged LoRA memory pool — page-level allocation, eviction, and weight scatter.

Provides :class:`LoRAPagePool`, a paged alternative to :class:`LoRAMemoryPool`
that manages GPU memory for LoRA adapter weights at page granularity.

* Each adapter's weight tensor is split into fixed-size *logical pages*.
* A :attr:`page_table` maps logical pages → physical page indices (``-1``
  means the page is swapped out).
* Eviction is page-level (not adapter-level), so an adapter can be partially
  resident and partially swapped.

All ``sglang``-package imports are lazy (inside method bodies) so that the
module can be imported in test environments without the full serving stack.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.lora.layers import BaseLayerWithLoRA
    from sglang.srt.lora.lora import LoRAAdapter
    from sglang.srt.lora.lora_config import LoRAConfig

logger = logging.getLogger(__name__)


class LoRAPagePool:
    """Paged LoRA GPU memory pool with page-level allocation and eviction.

    Each physical page holds ``PAGE_RANK_SIZE`` rows of the rank dimension.
    An adapter with rank ``r`` needs ``ceil(r / PAGE_RANK_SIZE)`` logical pages.

    Physical page storage (per-module, per-layer):

    * ``A_pages[module_name][layer]``: ``[total_pages, PAGE_RANK_SIZE * c, input_dim]``
    * ``B_pages[module_name][layer]``: ``[total_pages, output_dim, PAGE_RANK_SIZE]``

    where ``c = get_stacked_multiply(module_name)`` (e.g. 3 for ``qkv_proj``).
    """

    # Default page rank size. Can be overridden in __init__.
    PAGE_RANK_SIZE: int = 8

    def __init__(
        self,
        total_pages: int,
        dtype: torch.dtype,
        device: torch.device,
        target_modules: Set[str],
        num_layers: int,
        base_model: torch.nn.Module,
        page_rank_size: int = 8,
        tp_size: int = 1,
        tp_rank: int = 0,
        max_lora_rank: int = 0,
        max_loras_per_batch: int = 0,
    ):
        self.total_pages = total_pages
        self.PAGE_RANK_SIZE = page_rank_size
        self.dtype = dtype
        self.device = device
        self.num_layers = num_layers
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.max_lora_rank = max_lora_rank
        self.max_loras_per_batch = max_loras_per_batch

        cfg = base_model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        self.base_hf_config = cfg

        # Free / allocated page tracking
        self.free_page_indices: Set[int] = set(range(total_pages))

        # Reverse map: physical page index → owning uid (O(1) eviction lookup)
        self.phys_page_to_uid: Dict[int, str] = {}

        # uid → list of physical page indices (length = ceil(rank / PAGE_RANK_SIZE))
        self.page_table: Dict[str, List[int]] = {}

        # Generation counter: increments on every page modification
        # (evict/page_in/allocate/free). Used to invalidate page_table caches.
        self.page_generation: int = 0

        # uid → actual LoRA rank
        self.adapter_ranks: Dict[str, int] = {}

        # Per-page LRU access timestamps
        self.page_access_times: List[float] = [0.0] * total_pages
        # Eviction statistics for I7 metric
        self.total_bytes_evicted: int = 0
        # Actual host->device I/O: bytes actually scattered into GPU pages.
        # Only counts real transfers (resident adapter hits do NOT scatter, so
        # they add nothing) — this is the true swap-in cost, NOT eviction count
        # × page size. Bandwidth is constant, so bytes ∝ I/O time.

        # Pinned adapters: their pages are never evicted once resident.
        self.pinned_uids: Set[str] = set()

        # Physical page storage — populated by _init_pages
        self.A_pages: Dict[str, List[torch.Tensor]] = {}
        self.B_pages: Dict[str, List[torch.Tensor]] = {}

        # Embedding page storage (optional)
        self.embedding_A_pages: Dict[str, torch.Tensor] = {}
        self.embedding_B_pages: Dict[str, torch.Tensor] = {}

        self._init_pages(base_model, target_modules)
        self._init_embedding_pages(base_model, target_modules)

    # ── compatibility with LoRAManager ────────────────────────────────────

    def can_support(self, config: LoRAConfig) -> bool:
        """Check if the pool can accommodate a LoRA adapter with *config*."""
        if config.r > self.max_lora_rank:
            return False
        try:
            from sglang.srt.lora.utils import get_normalized_target_modules

            adapter_target_modules = get_normalized_target_modules(
                config.target_modules
            )
        except ImportError:
            # Full sglang stack not available (e.g. sgl_kernel missing).
            # Fall back to a simple check with the raw target_modules.
            raw = config.target_modules or []
            if raw == "all":
                return True
            adapter_target_modules = set(raw if isinstance(raw, list) else [raw])
        if "all" in adapter_target_modules:
            return True
        return adapter_target_modules.issubset(self.target_modules)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _has_moe_module(base_model: torch.nn.Module) -> bool:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        return any(isinstance(m, FusedMoE) for m in base_model.modules())

    # ── page storage initialisation ────────────────────────────────────────

    def _init_pages(self, base_model: torch.nn.Module, target_modules: Set[str]):
        """Allocate ``A_pages`` and ``B_pages`` tensors for each target module.

        Under TP>1, the weight storage dimensions need to match the TP-sliced
        weights used at serving time. We allocate A_pages with ``input_dim // tp``
        and B_pages with ``output_dim // tp`` (ceil division). The scatter
        functions handle modules where the weight is not actually sharded on
        that axis by slicing the weight tensor.
        """
        from sglang.srt.lora.utils import (
            EMBEDDING_NAMES,
            REPLICATED_LINEAR_LORA_NAMES,
            ROW_PARALLELISM_LINEAR_LORA_NAMES,
            get_hidden_dim,
            get_stacked_multiply,
        )

        pr = self.PAGE_RANK_SIZE
        has_moe = self._has_moe_module(base_model)
        tp = max(self.tp_size, 1)

        def allocate_module_pages(
            buffer: Dict[str, List[torch.Tensor]],
            get_shape: str,  # "A" or "B"
        ):
            for module_name in target_modules:
                if module_name in EMBEDDING_NAMES:
                    continue
                ambiguous = module_name in {"gate_up_proj", "down_proj"}
                if ambiguous and has_moe:
                    if f"{module_name}_moe" in target_modules:
                        continue

                tensors: List[torch.Tensor] = []
                for layer_idx in range(self.num_layers):
                    input_dim, output_dim = get_hidden_dim(
                        module_name, self.base_hf_config, base_model, layer_idx
                    )
                    c = get_stacked_multiply(module_name, base_model)
                    if get_shape == "A":
                        # LoRA-A is on the *input* dim. Mirror flat
                        # mem_pool.get_lora_A_shape: only RowParallel modules
                        # shard the input (divide by tp); ColumnParallel
                        # modules (qkv_proj, gate_up_proj) keep the FULL
                        # input_dim because their input is replicated. The
                        # shrink kernel strides A_pages by INPUT_DIM ==
                        # x.shape[1], so under-allocating here (e.g. gate_up
                        # 8192 -> 1024) makes n_offset*INPUT_DIM read A_pages
                        # out of bounds -> illegal memory access.
                        if (
                            tp > 1
                            and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES
                            and module_name not in REPLICATED_LINEAR_LORA_NAMES
                        ):
                            a_dim = (input_dim + tp - 1) // tp  # ceil div
                        else:
                            a_dim = input_dim
                        shape = (self.total_pages, pr * c, a_dim)
                    else:
                        # LoRA-B is on the *output* dim. Only ColumnParallel
                        # modules shard the output; RowParallel/Replicated do
                        # NOT (each rank owns the full output). Mirror flat
                        # mem_pool.get_lora_B_shape: skip the tp split for
                        # ROW_PARALLELISM / REPLICATED names. Otherwise B_pages
                        # is undersized (e.g. down_proj 8192 -> 1024) and the
                        # expand kernel reads B_pages out of bounds.
                        if (
                            tp > 1
                            and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
                            and module_name not in REPLICATED_LINEAR_LORA_NAMES
                        ):
                            b_dim = (output_dim + tp - 1) // tp  # ceil div
                        else:
                            b_dim = output_dim
                        shape = (self.total_pages, b_dim, pr)
                    tensors.append(
                        torch.zeros(shape, dtype=self.dtype, device=self.device)
                    )
                buffer[module_name] = tensors

        allocate_module_pages(self.A_pages, "A")
        allocate_module_pages(self.B_pages, "B")

    def _init_embedding_pages(
        self, base_model: torch.nn.Module, target_modules: Set[str]
    ):
        """Allocate embedding page storage for embed_tokens and lm_head."""
        from sglang.srt.lora.utils import (
            EMBEDDING_NAMES,
            get_hidden_dim,
            get_lm_head_lora_b_shard_size,
        )

        emb_targets = target_modules & set(EMBEDDING_NAMES)
        if not emb_targets:
            return

        pr = self.PAGE_RANK_SIZE
        cfg = self.base_hf_config

        # Cache lm_head shard_indices for B shard sizing
        lm_head_shard_indices = None
        if "lm_head" in emb_targets and self.tp_size > 1:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            for _, mod in base_model.named_modules():
                if isinstance(mod, ParallelLMHead):
                    lm_head_shard_indices = mod.shard_indices
                    break

        for module_name in emb_targets:
            # A_shape: [total_pages, pr, input_dim]
            input_dim, _ = get_hidden_dim(
                module_name, cfg, base_model, 0, lora_added_vocab_size=0
            )
            a_shape = (self.total_pages, pr, input_dim)
            self.embedding_A_pages[module_name] = torch.zeros(
                a_shape, dtype=self.dtype, device=self.device
            )

            # B_shape: [total_pages, output_dim, pr]
            _, output_dim = get_hidden_dim(
                module_name, cfg, base_model, 0, lora_added_vocab_size=0
            )
            if module_name == "lm_head":
                output_dim = get_lm_head_lora_b_shard_size(
                    output_dim,
                    shard_indices=lm_head_shard_indices,
                )
            b_shape = (self.total_pages, output_dim, pr)
            self.embedding_B_pages[module_name] = torch.zeros(
                b_shape, dtype=self.dtype, device=self.device
            )

    # ── B0: page management ───────────────────────────────────────────────

    def get_num_pages_for_rank(self, rank: int) -> int:
        """Number of logical pages needed for an adapter of *rank*."""
        if rank <= 0:
            return 0
        return ceil_div(rank, self.PAGE_RANK_SIZE)

    def allocate_pages(self, uid: str, rank: int) -> bool:
        """Allocate all required physical pages for *uid* with given *rank*.

        Returns ``True`` on success.  Returns ``False`` (without partial
        allocation) when there aren't enough free pages.
        """
        if not hasattr(self, "phys_page_to_uid"):
            self.phys_page_to_uid = {}
        needed = self.get_num_pages_for_rank(rank)
        if needed == 0:
            self.page_table[uid] = []
            self.adapter_ranks[uid] = 0
            return True
        if len(self.free_page_indices) < needed:
            return False
        pages = []
        for _ in range(needed):
            p = self.free_page_indices.pop()
            pages.append(p)
        self.page_table[uid] = pages
        self.adapter_ranks[uid] = rank
        for p in pages:
            self.phys_page_to_uid[p] = uid
        self.page_generation = getattr(self, "page_generation", 0) + 1
        return True

    def free_pages(self, uid: str):
        """Free all physical pages owned by *uid*."""
        pinned_uids = getattr(self, "pinned_uids", None)
        if pinned_uids is not None:
            pinned_uids.discard(uid)
        if not hasattr(self, "phys_page_to_uid"):
            self.phys_page_to_uid = {}
        if uid not in self.page_table:
            return
        for p in self.page_table[uid]:
            if p != -1:
                self.free_page_indices.add(p)
                self.phys_page_to_uid.pop(p, None)
        del self.page_table[uid]
        self.page_generation = getattr(self, "page_generation", 0) + 1
        if uid in self.adapter_ranks:
            del self.adapter_ranks[uid]

    def mark_page_accessed(self, page_idx: int):
        """Record a page access for LRU ordering."""
        self.page_access_times[page_idx] = time.monotonic()

    def mark_adapter_pages_accessed(self, uid: str):
        """Mark all resident pages of an adapter as recently accessed.
        Called from fetch_new_loras every batch to ensure LRU eviction
        reflects actual usage, not just initial page-in order.
        """
        if uid not in self.page_table:
            return
        now = time.monotonic()
        for p in self.page_table[uid]:
            if p != -1:
                self.page_access_times[p] = now

    def evict_pages(
        self, num_pages_needed: int, protected_page_set: Set[int]
    ) -> List[int]:
        """Evict up to *num_pages_needed* physical pages using page-level LRU.

        Skips pages in *protected_page_set* (running-batch pages).
        Evicted pages are freed and their ``page_table`` entries are set to
        ``-1``.

        Returns the list of evicted physical page indices.
        """
        pinned_pages = self.get_pinned_pages()
        candidates = [
            p
            for p in range(self.total_pages)
            if p not in self.free_page_indices
            and p not in protected_page_set
            and p not in pinned_pages
        ]
        # LRU: sort by last access time (oldest first)
        candidates.sort(key=lambda p: self.page_access_times[p])

        evicted: List[int] = []
        for p in candidates:
            if len(evicted) >= num_pages_needed:
                break
            uid = getattr(self, "phys_page_to_uid", {}).get(p)
            logic_idx = -1
            if uid is not None and uid in self.page_table:
                pt = self.page_table[uid]
                for logic_idx, phys_idx in enumerate(pt):
                    if phys_idx == p:
                        pt[logic_idx] = -1
                        break
            if hasattr(self, "phys_page_to_uid"):
                self.phys_page_to_uid.pop(p, None)
            self.free_page_indices.add(p)
            evicted.append(p)

        self.page_generation = getattr(self, "page_generation", 0) + 1
        return evicted

    # ── utilities used by LoRAManager ─────────────────────────────────────

    @property
    def target_modules(self) -> Set[str]:
        """Return the set of module names known to the pool."""
        return set(self.A_pages.keys()) | set(self.embedding_A_pages.keys())

    def get_embedding_tensor(
        self, module_name: str, is_a: bool = True
    ) -> Optional[torch.Tensor]:
        """Return the embedding page tensor for *module_name* (embed_tokens/lm_head).

        Returns ``None`` if the module is not in the pool.
        """
        if is_a:
            return self.embedding_A_pages.get(module_name)
        return self.embedding_B_pages.get(module_name)

    def max_pages_per_lora_for_batch(self, uids: List[Optional[str]]) -> int:
        """Max logical pages needed by any adapter in a batch."""
        max_p = 0
        for uid in uids:
            if uid is not None and uid in self.adapter_ranks:
                np = self.get_num_pages_for_rank(self.adapter_ranks[uid])
                if np > max_p:
                    max_p = np
        return max_p

    def is_complete(self, uid: str, rank: int) -> bool:
        """All logical pages for *uid* have valid physical pages."""
        if rank <= 0 or uid is None:
            return True
        if uid not in self.page_table:
            return False
        expected = self.get_num_pages_for_rank(rank)
        actual = self.page_table[uid]
        if len(actual) != expected:
            return False
        return all(p != -1 for p in actual)

    def get_missing_pages(self, uid: str, rank: int) -> List[int]:
        """Indices of logical pages that are swapped out (``-1``)."""
        if rank <= 0 or uid is None:
            return []
        expected = self.get_num_pages_for_rank(rank)
        actual = self.page_table.get(uid, [])
        missing = []
        for i in range(expected):
            if i >= len(actual) or actual[i] == -1:
                missing.append(i)
        return missing

    def build_page_table_tensor(
        self, uids: List[Optional[str]], max_pages_per_lora: int
    ) -> torch.Tensor:
        """Build a dense int32 page-table tensor for kernel usage.

        Shape: ``[len(uids), max_pages_per_lora]``.
        Entries are physical page indices or ``-1`` (swapped out / unused slot).
        """
        table = torch.full(
            (len(uids), max_pages_per_lora),
            -1,
            dtype=torch.int32,
        )
        for i, uid in enumerate(uids):
            if uid in self.page_table:
                pt = self.page_table[uid]
                n = min(len(pt), max_pages_per_lora)
                if n > 0:
                    table[i, :n] = torch.as_tensor(pt[:n], dtype=torch.int32)
        return table.to(self.device, non_blocking=True)

    def page_in(self, uid: str, logic_page_idx: int) -> int:
        """Allocate a physical page for *uid*'s logical page *logic_page_idx*.

        Returns -1 when the pool is exhausted.
        """
        if not self.free_page_indices:
            return -1
        phys = self.free_page_indices.pop()
        self.page_table[uid][logic_page_idx] = phys
        if not hasattr(self, "phys_page_to_uid"):
            self.phys_page_to_uid = {}
        self.phys_page_to_uid[phys] = uid
        self.page_generation = getattr(self, "page_generation", 0) + 1
        self.mark_page_accessed(phys)
        return phys

    def get_protected_pages(self, uids: Set[str]) -> Set[int]:
        """All physical pages currently used by the given adapters."""
        protected: Set[int] = set()
        for uid in uids:
            if uid in self.page_table:
                for p in self.page_table[uid]:
                    if p != -1:
                        protected.add(p)
        return protected

    def get_pinned_pages(self) -> Set[int]:
        """Physical pages belonging to pinned adapters — never evicted."""
        return self.get_protected_pages(getattr(self, "pinned_uids", set()))

    def pin_adapter(
        self,
        uid: str,
        adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ) -> bool:
        """Page-in *uid* and mark it as pinned (permanently resident).

        Evicts unpinned, non-protected pages if needed.
        Returns ``True`` on success.
        """
        if uid is None:
            return True
        protected = self.get_pinned_pages() | self.get_protected_pages({uid})
        if not self.ensure_adapter_ready(uid, adapter, protected, lora_modules):
            return False
        self.pinned_uids.add(uid)
        return True

    def unpin_adapter(self, uid: str):
        """Remove pin status.  Pages stay resident but become evictable."""
        self.pinned_uids.discard(uid)

    # ── B1: weight scatter / load ─────────────────────────────────────────

    def _compute_page_bytes(self) -> int:
        """Compute the approximate bytes per page from the A+B buffer shapes.

        Each page stores PAGE_RANK_SIZE rows of LoRA weights.
        For a single module: bytes = PAGE_RANK_SIZE * (c*input_dim + output_dim) * 2
        where 2 is for BF16 dtype.
        Summed over all target modules.
        """
        total = 0
        for module_name in self.A_pages:
            for layer_ap in self.A_pages[module_name]:
                if layer_ap is not None and layer_ap.numel() > 0:
                    # A_pages shape: (total_pages, PAGE_RANK_SIZE*c, input_dim)
                    per_page_a = layer_ap[0].numel() * layer_ap.element_size()
                    total += per_page_a
                    break
            for layer_bp in self.B_pages.get(module_name, []):
                if layer_bp is not None and layer_bp.numel() > 0:
                    per_page_b = layer_bp[0].numel() * layer_bp.element_size()
                    total += per_page_b
                    break
        return total if total > 0 else 0

    def _scatter_a_weight_to_pages(
        self,
        module_name: str,
        layer_id: int,
        weight_2d: Optional[torch.Tensor],
        lora_rank: int,
        phys_pages: List[int],
        c: Optional[int] = None,
        logic_page_indices: Optional[List[int]] = None,
    ):
        """Scatter a single LoRA-A weight tensor across physical pages.

        *weight_2d*: ``[lora_rank * c, hidden_dim]`` or ``None`` (zero-fill).
        *c*: stacked multiplier (default: auto-detected from module name).

        TP compatibility: if weight_2d is wider than the page buffer
        (ColumnParallel module where A_weight has full hidden_dim), slice
        weight_2d along the hidden dimension.
        """
        if c is None:
            from sglang.srt.lora.utils import get_stacked_multiply

            c = get_stacked_multiply(module_name)
        pr = self.PAGE_RANK_SIZE
        target = self.A_pages[module_name][layer_id]
        if logic_page_indices is None:
            logic_page_indices = list(range(len(phys_pages)))

        tp = max(self.tp_size, 1)
        page_hidden = target.shape[-1]

        for logic_idx, phys in zip(logic_page_indices, phys_pages):
            r_start = logic_idx * pr
            r_end = min(r_start + pr, lora_rank)
            target[phys].zero_()
            if r_end <= r_start:
                continue
            for ci in range(c):
                src_start = ci * lora_rank + r_start
                src_end = ci * lora_rank + r_end
                dst_start = ci * pr
                dst_end = ci * pr + (r_end - r_start)
                if weight_2d is not None:
                    w = weight_2d[src_start:src_end, :]
                    # If weight is wider than page buffer, slice along
                    # hidden dim (ColumnParallel: A_weight not sharded)
                    if w.shape[-1] > page_hidden:
                        hid_offset = self.tp_rank * page_hidden
                        w = w[:, hid_offset : hid_offset + page_hidden]
                    target[phys, dst_start:dst_end, :].copy_(w, non_blocking=True)
                    # Actual host->device bytes scattered for this page slice
                else:
                    target[phys, dst_start:dst_end, :].zero_()
            self.mark_page_accessed(phys)

    def _scatter_b_weight_to_pages(
        self,
        module_name: str,
        layer_id: int,
        weight_2d: Optional[torch.Tensor],
        lora_rank: int,
        phys_pages: List[int],
        scaling: float = 1.0,
        logic_page_indices: Optional[List[int]] = None,
    ):
        """Scatter a single LoRA-B weight tensor across physical pages.

        *weight_2d*: ``[output_dim, lora_rank]`` or ``None`` (zero-fill).

        TP compatibility: if weight_2d has a larger output_dim than the
        page buffer (RowParallel module where B_weight has full output_dim),
        slice weight_2d along the output dimension.
        """
        pr = self.PAGE_RANK_SIZE
        target = self.B_pages[module_name][layer_id]
        if logic_page_indices is None:
            logic_page_indices = list(range(len(phys_pages)))

        tp = max(self.tp_size, 1)
        page_output_dim = target.shape[1]

        for logic_idx, phys in zip(logic_page_indices, phys_pages):
            r_start = logic_idx * pr
            r_end = min(r_start + pr, lora_rank)
            target[phys].zero_()
            if r_end <= r_start:
                continue
            if weight_2d is not None:
                w_src = weight_2d[:, r_start:r_end]
                if w_src.shape[0] > page_output_dim:
                    out_offset = self.tp_rank * page_output_dim
                    w_src = w_src[out_offset : out_offset + page_output_dim, :]
                dst = target[phys, :, : w_src.shape[-1]]
                dst.copy_(w_src, non_blocking=True)
                # if scaling != 1.0:
                #     dst.mul_(scaling)

            else:
                target[phys, :, : (r_end - r_start)].zero_()
            self.mark_page_accessed(phys)

    def load_lora_weight_to_pages(
        self,
        uid: str,
        adapter: Optional[LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ):
        """Load a full adapter's weights into its allocated physical pages.

        *uid* can be ``None`` (base model) — zeroes out the allocated pages.
        Must be called after :meth:`allocate_pages`.
        """

        if uid is None:
            # Base model: zero out all pages owned by this uid
            if uid in self.page_table:
                for pages in self.A_pages.values():
                    for layer_t in pages:
                        for p in self.page_table[uid]:
                            layer_t[p].zero_()
                for pages in self.B_pages.values():
                    for layer_t in pages:
                        for p in self.page_table[uid]:
                            layer_t[p].zero_()
            return

        assert adapter is not None
        lora_rank = adapter.config.r
        phys_pages = self.page_table.get(uid, [])
        if not phys_pages:
            return  # rank=0

        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.utils import get_target_module_name

        for layer_id in range(self.num_layers):
            layer_weights = adapter.layers[layer_id].weights
            temp_A: Dict[str, Optional[torch.Tensor]] = {}
            temp_B: Dict[str, Optional[torch.Tensor]] = {}

            for name, wt in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)

                # Skip MoE for now (TODO B4/B5)
                if "experts" in name:
                    continue

                # Standard module
                if "lora_A" in name:
                    temp_A[target_module] = wt
                elif "lora_B" in name:
                    temp_B[target_module] = wt

            # Apply TP slicing and scatter
            cur_layer_modules = lora_modules[layer_id]
            for module_name, module in cur_layer_modules.items():
                if isinstance(module, FusedMoEWithLoRA):
                    continue  # MoE not yet supported

                target_module = get_target_module_name(module_name, self.target_modules)
                if target_module not in self.A_pages:
                    continue

                w_a = temp_A.get(target_module)
                w_b = temp_B.get(target_module)

                if w_a is not None:
                    if self.tp_size > 1:
                        w_a = module.slice_lora_a_weights(w_a, self.tp_rank)
                    self._scatter_a_weight_to_pages(
                        target_module, layer_id, w_a, lora_rank, phys_pages
                    )
                else:
                    self._scatter_a_weight_to_pages(
                        target_module, layer_id, None, lora_rank, phys_pages
                    )

                if w_b is not None:
                    if self.tp_size > 1:
                        w_b = module.slice_lora_b_weights(w_b, self.tp_rank)
                    self._scatter_b_weight_to_pages(
                        target_module,
                        layer_id,
                        w_b,
                        lora_rank,
                        phys_pages,
                        scaling=adapter.scaling,
                    )
                else:
                    self._scatter_b_weight_to_pages(
                        target_module, layer_id, None, lora_rank, phys_pages
                    )

    def load_missing_pages(
        self,
        uid: str,
        adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ):
        """Reload only the missing (swapped-out) pages for *uid*.

        For TP > 1, applies ``slice_lora_a_weights`` / ``slice_lora_b_weights``
        before scattering, to mirror the logic in :meth:`load_lora_weight_to_pages`.
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.utils import get_target_module_name

        missing = self.get_missing_pages(uid, adapter.config.r)
        if not missing:
            return
        lora_rank = adapter.config.r

        # Allocate a physical page for EACH missing logical page ONCE, before
        # scattering. page_table[uid] is adapter-level (all layers/modules share
        # one logical->physical map), so allocating inside the per-layer,
        # per-module loop would re-page-in the same logical_idx N*M times,
        # leaking the first N*M-1 physical pages and exhausting the pool.
        # load_lora_weight_to_pages does the same one-time allocation upfront.
        new_phys: Dict[int, int] = {}
        for logic_idx in missing:
            new_phys[logic_idx] = self.page_in(uid, logic_idx)

        for layer_id in range(self.num_layers):
            layer_weights = adapter.layers[layer_id].weights
            temp_A: Dict[str, Optional[torch.Tensor]] = {}
            temp_B: Dict[str, Optional[torch.Tensor]] = {}
            for name, wt in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)
                if "experts" in name:
                    continue
                if "lora_A" in name:
                    temp_A[target_module] = wt
                elif "lora_B" in name:
                    temp_B[target_module] = wt

            cur_layer_modules = lora_modules[layer_id]
            for module_name, module in cur_layer_modules.items():
                if isinstance(module, FusedMoEWithLoRA):
                    continue
                target_module = get_target_module_name(module_name, self.target_modules)
                if target_module not in self.A_pages:
                    continue

                w_a = temp_A.get(target_module)
                w_b = temp_B.get(target_module)

                for logic_idx in missing:
                    phys = new_phys[logic_idx]
                    if w_a is not None:
                        w_a_shard = (
                            module.slice_lora_a_weights(w_a, self.tp_rank)
                            if self.tp_size > 1
                            else w_a
                        )
                        self._scatter_a_weight_to_pages(
                            target_module,
                            layer_id,
                            w_a_shard,
                            lora_rank,
                            [phys],
                            logic_page_indices=[logic_idx],
                        )
                    else:
                        self._scatter_a_weight_to_pages(
                            target_module,
                            layer_id,
                            None,
                            lora_rank,
                            [phys],
                        )

                    if w_b is not None:
                        w_b_shard = (
                            module.slice_lora_b_weights(w_b, self.tp_rank)
                            if self.tp_size > 1
                            else w_b
                        )
                        self._scatter_b_weight_to_pages(
                            target_module,
                            layer_id,
                            w_b_shard,
                            lora_rank,
                            [phys],
                            scaling=adapter.scaling,
                            logic_page_indices=[logic_idx],
                        )
                    else:
                        self._scatter_b_weight_to_pages(
                            target_module,
                            layer_id,
                            None,
                            lora_rank,
                            [phys],
                        )

    def ensure_adapter_ready(
        self,
        uid: str,
        adapter: Optional[LoRAAdapter],
        protected_pages: Set[int],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ) -> bool:
        """Make sure *uid*'s pages are allocated and resident.

        * If adapter not in pool → allocate + full load.
        * If partially swapped → evict others if needed, then page-in + reload.
        * If fully resident → no-op.

        Returns ``True`` if the adapter is ready after this call.
        """
        if uid is None:
            return True

        rank = adapter.config.r
        if rank <= 0:
            return True

        if uid not in self.page_table:
            # Full allocation
            free_before = len(self.free_page_indices)
            if not self.allocate_pages(uid, rank):
                needed = self.get_num_pages_for_rank(rank)
                if free_before < needed:
                    self.evict_pages(needed - free_before, protected_pages)
                if not self.allocate_pages(uid, rank):
                    return False
            self.load_lora_weight_to_pages(uid, adapter, lora_modules)
            return True

        # Check for missing pages
        missing = self.get_missing_pages(uid, rank)
        if missing:
            needed = len(missing)
            free_before = len(self.free_page_indices)
            if free_before < needed:
                self.evict_pages(needed - free_before, protected_pages)
            if len(self.free_page_indices) < needed:
                return False
            self.load_missing_pages(uid, adapter, lora_modules)
            return self.is_complete(uid, rank)

        return True  # already complete
