"""MLX model runner for Apple Silicon using native paged attention.

KV data lives in a flat pool indexed by a radix trie for prefix sharing.
"""

import logging
import time

import mlx.core as mx
import psutil
from mlx_lm import load as mlx_lm_load

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.kv_cache.radix_trie import MlxRadixTrie

logger = logging.getLogger(__name__)


class MlxModelRunner:
    """MLX model runner with radix-cache prefix sharing and paged attention."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static

        self._load_model()

        # Pin MLX allocations to prevent OS paging
        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Wired memory limit set to {max_wired / (1024**3):.1f} GB")

        patch_model_attention(self.model)

        self._num_layers = get_num_layers(self.model)

        self._req_token_ids: dict[str, list[int]] = {}

        # Radix cache state
        self._kv_pool: MlxKVPool | None = None
        self._radix_trie: MlxRadixTrie | None = None
        self._req_slot_ids: dict[str, list[int]] = {}
        self._req_last_node: dict[str, object | None] = {}
        self._req_prefix_len: dict[str, int] = {}

        self._init_radix_cache(pool_size)

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )
        # Force-evaluate weights so mx.get_active_memory() reflects
        # actual usage before KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def _init_radix_cache(self, pool_size: int | None) -> None:
        """Initialize pool and trie.  Auto-sizes from available memory if needed."""
        num_layers = self._num_layers

        layer_list, attn_attr = find_attention_layers(self.model)
        if not layer_list:
            raise RuntimeError("Cannot init radix cache: no attention layers found")

        sample_block = layer_list[0]
        sample_attn = getattr(sample_block, attn_attr)
        # Handle if already patched
        if hasattr(sample_attn, "_inner"):
            sample_attn = sample_attn._inner

        n_kv_heads = sample_attn.n_kv_heads

        if hasattr(sample_attn, "head_dim"):
            head_dim = sample_attn.head_dim
        elif hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            # k_proj.weight shape: (n_kv_heads * head_dim, hidden_size)
            head_dim = sample_attn.k_proj.weight.shape[0] // n_kv_heads
        else:
            raise RuntimeError("Cannot determine head_dim from attention module")

        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype

        if pool_size is None:
            pool_size = self._profile_pool_size(num_layers, n_kv_heads, head_dim, dtype)

        self._kv_pool = MlxKVPool(
            pool_size=pool_size,
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self._radix_trie = MlxRadixTrie(pool_capacity=pool_size)
        logger.info(
            f"KV pool initialized: pool_size={pool_size}, "
            f"{num_layers} layers, {n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def _profile_pool_size(
        self,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
    ) -> int:
        """Derive KV pool slot count from available memory."""
        vm = psutil.virtual_memory()
        metal_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        mlx_used = mx.get_active_memory()

        usable = min(int(vm.total * self._mem_fraction_static), metal_limit)
        kv_budget = min(
            max(usable - mlx_used, 0),
            int(vm.available * self._mem_fraction_static),
        )

        bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * dtype.size
        pool_size = max(kv_budget // bytes_per_slot, 256)
        logger.info(
            f"Auto-sized KV pool: total_ram={vm.total / (1024**3):.1f} GB, "
            f"sys_available={vm.available / (1024**3):.2f} GB, "
            f"metal_limit={metal_limit / (1024**3):.1f} GB, "
            f"mlx_used={mlx_used / (1024**3):.2f} GB, "
            f"kv_budget={kv_budget / (1024**3):.2f} GB, "
            f"bytes_per_slot={bytes_per_slot}, pool_size={pool_size}"
        )
        return pool_size

    def prefill(
        self,
        req_id: str,
        token_ids: list[int],
    ) -> int:
        """Prefill a request.  Returns ``(next_token_id, prefix_len)``."""
        num_tokens = len(token_ids)

        match = self._radix_trie.match_prefix(token_ids)
        prefix_len = match.prefix_len
        matched_node = match.last_node

        if self.disable_radix_cache:
            prefix_len = 0
            matched_node = None

        if prefix_len > 0 and matched_node is not None:
            self._radix_trie.inc_ref(matched_node)

        new_token_count = num_tokens - prefix_len
        if new_token_count > 0:
            new_slots = self._kv_pool.allocator.alloc(new_token_count)
            if new_slots is None:
                freed = self._radix_trie.evict(new_token_count)
                if freed:
                    self._kv_pool.allocator.free(freed)
                new_slots = self._kv_pool.allocator.alloc(new_token_count)
                if new_slots is None:
                    if prefix_len > 0 and matched_node is not None:
                        self._radix_trie.dec_ref(matched_node)
                    raise RuntimeError(
                        f"KV pool exhausted: need {new_token_count} slots, "
                        f"only {self._kv_pool.allocator.available} available"
                    )
        else:
            new_slots = []

        if prefix_len > 0:
            all_slots = match.slot_ids + new_slots
        else:
            all_slots = new_slots

        # Only process new tokens
        if new_token_count > 0:
            extend_tokens = token_ids[prefix_len:]
        else:
            extend_tokens = token_ids[-1:]

        ctx = PagedAttentionContext(
            kv_pool=self._kv_pool,
            cu_seqlens=[0, len(extend_tokens)],
            offsets=[prefix_len],
            slot_mapping=new_slots if new_token_count > 0 else [all_slots[-1]],
            block_tables=[all_slots] if new_token_count > 0 else [all_slots],
            context_lens=[len(token_ids)],
        )

        set_context(ctx)
        try:
            input_ids = mx.array([extend_tokens], dtype=mx.int32)
            # Shim cache needed to bypass mlx_lm default cache logic
            # TODO: (Jonahcb): delete this if unnecessary
            # shim_cache = [None] * self._num_layers
            model_output = self.model(input_ids)#, cache=shim_cache)
            logits = self._extract_logits(model_output)

            last_logits = logits[:, -1, :]
            next_token_mlx = mx.argmax(last_logits, axis=-1)
            
            # Evaluate to ensure all async work executes
            eval_targets = [next_token_mlx]
            eval_targets.extend(self._kv_pool.all_buffers())
            mx.eval(*eval_targets)
        finally:
            clear_context()

        next_token = int(next_token_mlx.item())

        if not self.disable_radix_cache:
            self._radix_trie.insert(token_ids, all_slots)

        self._req_slot_ids[req_id] = all_slots
        self._req_token_ids[req_id] = list(token_ids) + [next_token]
        self._req_last_node[req_id] = matched_node if prefix_len > 0 else None
        self._req_prefix_len[req_id] = prefix_len

        return next_token, prefix_len

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
    ) -> int:
        """Continue prefill for a chunked request.

        Returns:
            Next token ID (greedy sampled).
        """
        assert req_id in self._req_slot_ids, f"extend called for unknown request {req_id}"

        num_new = len(new_token_ids)

        new_slots = self._kv_pool.allocator.alloc(num_new)
        if new_slots is None:
            freed = self._radix_trie.evict(num_new)
            if freed:
                self._kv_pool.allocator.free(freed)
            new_slots = self._kv_pool.allocator.alloc(num_new)
            if new_slots is None:
                raise RuntimeError(
                    f"KV pool exhausted: need {num_new} slots, "
                    f"only {self._kv_pool.allocator.available} available"
                )

        prev_tokens = self._req_token_ids[req_id]
        if prev_tokens:
            prev_tokens.pop()  # remove stale intermediate token
            
        old_len = len(prev_tokens)
        new_len = old_len + num_new

        all_slots = self._req_slot_ids[req_id] + new_slots

        ctx = PagedAttentionContext(
            kv_pool=self._kv_pool,
            cu_seqlens=[0, num_new],
            offsets=[old_len],
            slot_mapping=new_slots,
            block_tables=[all_slots],
            context_lens=[new_len],
        )

        set_context(ctx)
        try:
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            shim_cache = [None] * self._num_layers
            model_output = self.model(input_ids, cache=shim_cache)
            logits = self._extract_logits(model_output)

            last_logits = logits[:, -1, :]
            next_token_mlx = mx.argmax(last_logits, axis=-1)
            
            eval_targets = [next_token_mlx]
            eval_targets.extend(self._kv_pool.all_buffers())
            mx.eval(*eval_targets)
        finally:
            clear_context()

        next_token = int(next_token_mlx.item())

        prev_tokens.extend(new_token_ids)
        prev_tokens.append(next_token)

        self._req_slot_ids[req_id] = all_slots

        if not self.disable_radix_cache:
            full_prompt = prev_tokens[:-1]
            self._radix_trie.insert(full_prompt, self._req_slot_ids[req_id])

        logger.info(f"Extend req {req_id}: +{num_new} tokens")

        return next_token

    def prefill_batch(
        self,
        req_ids: list[str],
        token_ids_list: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Prefill multiple requests serially (BS=1 per forward)."""
        return [self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)]

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Decode one token per request using paged attention."""
        batch_size = len(req_ids)

        new_slots = self._kv_pool.allocator.alloc(batch_size)
        if new_slots is None:
            freed = self._radix_trie.evict(batch_size)
            if freed:
                self._kv_pool.allocator.free(freed)
            new_slots = self._kv_pool.allocator.alloc(batch_size)
            if new_slots is None:
                raise RuntimeError(
                    f"KV pool exhausted: need {batch_size} slots, "
                    f"only {self._kv_pool.allocator.available} available"
                )

        cu_seqlens = [0]
        offsets = []
        block_tables = []
        context_lens = []
        last_tokens = []

        for i, rid in enumerate(req_ids):
            self._req_slot_ids[rid].append(new_slots[i])
            old_len = len(self._req_slot_ids[rid]) - 1
            
            cu_seqlens.append(cu_seqlens[-1] + 1)
            offsets.append(old_len)
            block_tables.append(self._req_slot_ids[rid])
            context_lens.append(old_len + 1)
            last_tokens.append(self._req_token_ids[rid][-1])

        ctx = PagedAttentionContext(
            kv_pool=self._kv_pool,
            cu_seqlens=cu_seqlens,
            offsets=offsets,
            slot_mapping=new_slots,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        set_context(ctx)
        try:
            # Pass 1D sequence for packed inference (varlen)
            batched_input = mx.array([last_tokens], dtype=mx.int32)
            shim_cache = [None] * self._num_layers
            model_output = self.model(batched_input, cache=shim_cache)
            logits = self._extract_logits(model_output)

            next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            
            # Flatten the output to 1D to ensure tolist() returns a flat list
            next_tokens_mlx = next_tokens_mlx.reshape(-1)

            eval_targets = [next_tokens_mlx]
            eval_targets.extend(self._kv_pool.all_buffers())
            mx.eval(*eval_targets)
        finally:
            clear_context()

        next_tokens = next_tokens_mlx.tolist()
        
        if not isinstance(next_tokens, list):
            next_tokens = [next_tokens]
        elif len(next_tokens) == 1 and isinstance(next_tokens[0], list):
            pass
            
        # Ensure it's a list of ints
        flat_tokens = []
        for x in next_tokens:
            if isinstance(x, list):
                flat_tokens.extend(x)
            else:
                flat_tokens.append(x)
        next_tokens = flat_tokens
        
        # Pad or truncate if somehow lengths don't match, though they should
        if len(next_tokens) < len(req_ids):
            next_tokens.extend([next_tokens[-1]] * (len(req_ids) - len(next_tokens)))

        for i, rid in enumerate(req_ids):
            self._req_token_ids[rid].append(next_tokens[i])

        return next_tokens

    def has_request(self, req_id: str) -> bool:
        """Check if a request has active state."""
        return req_id in self._req_slot_ids

    def remove_request(self, req_id: str):
        """Clean up state for a completed request."""
        if not self.disable_radix_cache:
            last_node = self._req_last_node.pop(req_id, None)
            if last_node is not None:
                self._radix_trie.dec_ref(last_node)

        self._req_slot_ids.pop(req_id, None)
        self._req_token_ids.pop(req_id, None)
        self._req_prefix_len.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._req_slot_ids.clear()
        self._req_token_ids.clear()
        self._req_last_node.clear()
        self._req_prefix_len.clear()
        if self._radix_trie is not None:
            freed = self._radix_trie.reset()
            if freed and self._kv_pool is not None:
                self._kv_pool.allocator.free(freed)
        if self._kv_pool is not None:
            self._kv_pool.clear()
