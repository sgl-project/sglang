"""Engram: a gated n-gram hash memory written into the hc residual stream at a few layers.

Predecessor tokens come from the per-request n-gram token table the scheduler maintains
(forward_batch.ngram_embedding_info). The token map, hash multipliers and prime layout are
mirrored by the pure-torch oracle in test/manual/deepseek_v41, which checks them against the
released reference; the two copies must stay bit-identical or the hashed ids diverge.
"""

from __future__ import annotations

import logging
from typing import Optional

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_global_dp_buffer_len,
)
from sglang.srt.layers.engram_offload import get_engram_offload_manager
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_model,
    get_schedule,
    get_serving,
)
from sglang.srt.utils import add_prefix, is_npu
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

# Heartbeat interval for the per-forward offload lookup logs (first call is
# always logged); keeps the decode hot path quiet while still proving liveness.
# All engram-offload log lines share the [engram_offload] prefix for grepping.
_OFFLOAD_LOG_EVERY = 1000


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id, plus the compressed vocab size. The size feeds every
    hash multiplier, so a mismatch rehashes the whole table."""
    from tokenizers import Regex, normalizers

    # A private-use sentinel keeps a lone-space token from collapsing to "" under Strip().
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token has nothing to normalize; key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback) from a per-layer RNG, bounded so that
    token_id * multiplier fits in int64."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout(msgspec.Struct, frozen=True):
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head]
    n_heads: int
    head_dim: int

    @classmethod
    def build(
        cls,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
    ) -> EngramLayout:
        """Primes are drawn in (layer, n-gram size, head) order from one shared
        ascending sequence starting above vocab_size - 1."""
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=head_dim,
        )


def build_engram_layout(config) -> Optional[EngramLayout]:
    layer_ids = tuple(config.engram_layer_ids)
    if not layer_ids:
        return None
    return EngramLayout.build(
        layer_ids=layer_ids,
        num_embeddings=tuple(config.engram_num_embeddings),
        max_ngram_size=config.engram_max_ngram_size,
        n_heads=config.engram_n_heads,
        head_dim=config.engram_head_dim,
        vocab_size=config.engram_vocab_size,
    )


def compute_engram_hash_ids(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """tokens [T, n]: column 0 is the token itself, column s its s-th predecessor;
    blocked [T, n] marks look-back that ran off the sequence start. Returns
    [T, n_engram_layers, (n - 1) * n_heads] row ids into each layer's table."""
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed.unsqueeze(1) * multipliers
    # XOR the multiplied ids one look-back at a time: after step i the running value
    # is the (i + 1)-gram hash, bucketed by that n-gram size's primes.
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHasher(nn.Module):
    """Hash ids for every token of a forward batch, [T, n_engram_layers, n_hash_cols]."""

    def __init__(
        self,
        layout: EngramLayout,
        tokenizer,
        pad_id: int,
        compressed_vocab_size: int,
    ):
        super().__init__()
        self.max_ngram_size = layout.max_ngram_size
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == compressed_vocab_size, (
            f"the tokenizer normalizes to {vocab_size} distinct tokens but the config "
            f"expects {compressed_vocab_size}; every hash multiplier depends on it"
        )
        self.pad_id = token_map[pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self._idle_logged = 0

    @classmethod
    def from_config(cls, config, layout: EngramLayout) -> EngramHasher:
        # The compressed token map is built with the HF normalizers, so the HF
        # tokenizer backend is used here whatever the serving backend is.
        tokenizer = get_tokenizer(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            tokenizer_backend="huggingface",
        )
        result = cls(
            layout,
            tokenizer,
            config.engram_pad_id,
            config.engram_compressed_vocab_size,
        )
        result.image_token_id = (
            config.image_token_id
            if config.model_type == "deepseek_v4.1" and config.vision_n_layers > 0
            else None
        )
        return result

    def forward(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        info = forward_batch.ngram_embedding_info
        if info is None:
            # DP attention idle ranks run a dummy forward: ForwardBatch.init_new
            # returns early for IDLE batches before _init_ngram_embedding_info,
            # and prepare_mlp_sync_batch may further convert the mode to
            # EXTEND/TARGET_VERIFY with padded dummy tokens. Emit row id 0
            # (valid in every layer's table) so the dummy tokens flow through;
            # their outputs are sliced off in post_forward_mlp_sync_batch.
            self._idle_logged += 1
            if self._idle_logged == 1 or self._idle_logged % _OFFLOAD_LOG_EVERY == 0:
                logger.info(
                    "[engram_offload] hasher: ngram_embedding_info is None "
                    "(DP-attention idle rank), returning zero row ids "
                    "(#%d, %d dummy tokens)",
                    self._idle_logged,
                    len(input_ids),
                )
            return self.primes.new_zeros(
                (
                    len(input_ids),
                    self.primes.shape[0],
                    self.primes.shape[1] * self.primes.shape[2],
                )
            )
        req = forward_batch.req_pool_indices.to(torch.int64)
        if not forward_batch.forward_mode.is_decode():
            req = torch.repeat_interleave(
                req, forward_batch.extend_seq_lens.to(torch.int64)
            )
        positions = forward_batch.positions.to(torch.int64)
        # TP/DP-attention padding aligns token-level tensors (positions,
        # input_ids) to attn_tp_size while request-level req_pool_indices keeps
        # the real batch size. Align req to the token count; padded rows hash
        # from table row 0 and are discarded with the dummy tokens after the
        # forward (post_forward_mlp_sync_batch slices them off).
        num_tokens = positions.shape[0]
        if req.shape[0] < num_tokens:
            req = torch.cat([req, req.new_zeros(num_tokens - req.shape[0])])
        elif req.shape[0] > num_tokens:
            req = req[:num_tokens]
        shifts = torch.arange(self.max_ngram_size, device=positions.device)
        lookback = positions.unsqueeze(-1) - shifts
        tokens = info.token_table[req.unsqueeze(-1), lookback.clamp_min(0)].to(
            torch.int64
        )
        tokens[:, 0] = input_ids
        blocked = lookback < 0
        if self.image_token_id is not None:
            # Previous chunks remain hashed in the scheduler's token table.
            tokens = tokens.masked_fill(
                tokens >= MM_PAD_SHIFT_VALUE, self.image_token_id
            )
            # Once a lookback hits an image, every older predecessor is PAD.
            blocked = (
                (blocked | (tokens == self.image_token_id))
                .to(torch.int32)
                .cummax(-1)
                .values.bool()
            )
        # Belt-and-suspenders against OOB gathers in token_map[tokens] below:
        # unfilled token-table cells can hold arbitrary bits (freed rows are
        # stale; DP-attention padding hashes from reserved row 0), and the
        # gather runs unconditionally even where `blocked` would discard the
        # result. Clamp into the compressed vocab range AFTER the image mask
        # so genuine shifted image ids still get masked above.
        tokens = tokens.clamp(0, self.token_map.shape[0] - 1)
        return compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )


class _OffloadForwardBuffers:
    """Fixed-address device buffers for the offloaded lookup path: NPU graph
    capture only rewrites their contents, never their addresses. The staging
    slice is carved so its data_ptr is entry-aligned, as entry_gather requires."""

    def __init__(self, capacity: int, entry_bytes: int, device: torch.device):
        self.capacity = capacity
        self.ids = torch.empty(capacity, dtype=torch.int64, device=device)
        self.count = torch.empty(1, dtype=torch.int32, device=device)
        # entry_gather packs the gathered rows at dst + i * entry_bytes.
        self._backing = torch.empty(
            capacity * entry_bytes + entry_bytes - 1, dtype=torch.uint8, device=device
        )
        self.staging = self._backing[-self._backing.data_ptr() % entry_bytes :]


class EngramEmbedding(nn.Module):
    """One layer's hash table. Two residency modes:

    Device-resident (default): rows are sharded across the FULL TP group
    (minimizes per-rank memory). Under DP attention this follows the
    moe_dense_tp convention: gather indices into the uniform global DP buffer,
    all_reduce the row-shard partials over the full TP group, then scatter
    back to the local token segment. Storage follows the chip: arch35 keeps
    fp8 rows + e8m0 block scales (dequantized on the fly), everything else
    stores bf16 rows directly.

    Host-offload (SGLANG_OPT_ENGRAM_HOST_OFFLOAD, NPU): the fp8 weight table
    lives in the node-local acc_offload GVA pool as slot-segmented chunks
    (rows back to back from each slot start, row pitch = head dim, no
    padding; slot tail gaps strided over by the registered-layout
    addressing), row-sharded over the ranks of one node; forward
    entry-gathers the selected rows into a device staging buffer and every
    rank computes the full value itself (no all-reduce, so DP-attention's
    differing per-rank token counts never meet a collective). The 128x
    smaller scale table stays resident on the device.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        layer_hash_index: Optional[int] = None,
        config=None,
    ):
        super().__init__()
        # Offload mode never reads tp_size (no collective in _forward_offload);
        # the device path below shards over the FULL TP group.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.offload = None
        if (
            layer_hash_index is not None
            and config is not None
            and envs.SGLANG_OPT_ENGRAM_HOST_OFFLOAD.get()
            and is_npu()
        ):
            manager = get_engram_offload_manager(config)
            self.offload = manager.table(layer_hash_index)
            self._offload_manager = manager
            self._offload_finalized = False
            self._offload_buffers = None
            self._offload_calls = 0
            self._n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
            # The device keeps no weight table: the param stays meta so
            # load_weights still routes through it; the checkpoint slice lands
            # in pinned staging that finalize_offload() flushes into the GVA
            # pool, rows packed back to back at the registered block offset.
            self.rows = self.offload.rows
            # The chunk index is the node-local pool rank — every scheduler
            # on the node, dp replicas included — not the global tp_rank;
            # each node of a multi-node run holds its own replica's chunks.
            self.row_start = self._offload_manager.rank * self.rows
            self._is_arch35 = is_npu_arch35()
            if self._is_arch35:
                # A5 (arch35): fp8 weight + e8m0 block scales in the pool;
                # the 128x smaller scale table stays device-resident.
                self.weight = nn.Parameter(
                    torch.empty(self.rows, dim, dtype=torch.float8_e4m3fn, device="meta"),
                    requires_grad=False,
                )
                self.weight.weight_loader = self._load_weight_rows
                self._staging_weight = torch.empty(
                    self.rows, dim, dtype=torch.float8_e4m3fn, device="cpu", pin_memory=True
                )
                self.scale = nn.Parameter(
                    torch.empty(
                        num_embeddings, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu
                    ),
                    requires_grad=False,
                )
                self.scale.weight_loader = self._load_scale_rows
            else:
                # A3: bf16 weight in the pool, no scale table, no dequant.
                self.weight = nn.Parameter(
                    torch.empty(self.rows, dim, dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                )
                self.weight.weight_loader = self._load_weight_rows
                self._staging_weight = torch.empty(
                    self.rows, dim, dtype=torch.bfloat16, device="cpu", pin_memory=True
                )
                # Keep the attribute surface identical to the arch35 branch;
                # None is a plain attribute, not a registered parameter.
                self.scale = None
            logger.info(
                "[engram_offload] rank %d: layer %d OFFLOAD mode engaged "
                "(E=%d chunk_rows=%d entry=%dB block_off=%d table_id=%d; "
                "weight meta + pinned staging, %s)",
                self._offload_manager.rank,
                layer_hash_index,
                self.offload.num_embeddings,
                self.rows,
                self.offload.entry_bytes,
                self.offload.block_offset,
                self.offload.table_id,
                "fp8 weight + scale device-resident" if self._is_arch35 else "bf16 weight, no scale",
            )
            return
        assert num_embeddings % self.tp_size == 0, (num_embeddings, self.tp_size)
        self.rows = num_embeddings // self.tp_size
        self.row_start = get_tensor_model_parallel_rank() * self.rows
        if not is_npu_arch35():
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.weight.weight_loader = self._load_rows
        else:
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(self.rows, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu),
                requires_grad=False,
            )
            self.weight.weight_loader = self._load_rows
            self.scale.weight_loader = self._load_rows

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight[self.row_start : self.row_start + self.rows])

    def _load_weight_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        rows = loaded_weight[self.row_start : self.row_start + self.rows]
        # The last chunk can be shorter than self.rows (ceil sharding).
        self._staging_weight[: rows.shape[0]].copy_(rows)

    def _load_scale_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def finalize_offload(self):
        """Flush the pinned staging chunk into the GVA pool and pre-size the
        fixed forward buffers; called once after all checkpoint weights are
        loaded."""
        if self.offload is None or self._offload_finalized:
            return
        self.offload.pool_weight.copy_(self._staging_weight.view(torch.uint8))
        self._staging_weight = None
        self._offload_finalized = True
        logger.info(
            "[engram_offload] rank %d: layer %d weight chunk flushed to GVA "
            "pool (%d rows x %dB @ slot off %d)",
            self._offload_manager.rank,
            self.offload.layer_hash_index,
            self.offload.rows,
            self.offload.entry_bytes,
            self.offload.block_offset,
        )
        # Size the fixed buffers ONCE for the largest batch any phase can
        # produce. Graphs captured at any bs bake these addresses into their
        # launches; a later realloc for a bigger eager prefill would leave
        # every smaller captured graph with dangling pointers (the caching
        # allocator recycles the old block), so growth is forbidden after
        # this point.
        try:
            max_tokens = max(
                get_schedule().chunked_prefill_size or 0,
                get_exec().graph.cuda_graph_config.decode.max_bs or 0,
            )
        except Exception:
            # Standalone tests without the runtime context: fall back to
            # sizing on first use (still frozen afterwards).
            max_tokens = 0
        if max_tokens:
            # A3 keeps no device-resident param (bf16 rows, no scale), so take
            # the device from the NPU runtime — this branch is is_npu()-gated.
            device = (
                self.scale.device
                if self.scale is not None
                else torch.device("npu", torch.npu.current_device())
            )
            self._offload_buffers = _OffloadForwardBuffers(
                max_tokens * self._n_hash_cols,
                self.offload.entry_bytes,
                device,
            )
            logger.info(
                "[engram_offload] rank %d: layer %d forward buffers "
                "pre-sized and frozen (capacity=%d ids for max %d tokens x %d cols)",
                self._offload_manager.rank,
                self.offload.layer_hash_index,
                self._offload_buffers.capacity,
                max_tokens,
                self._n_hash_cols,
            )

    def _ensure_offload_buffers(self, count: int, device: torch.device):
        buffers = self._offload_buffers
        if buffers is None:
            # Preallocation was skipped (no runtime context): size on first
            # use, then freeze.
            self._offload_buffers = _OffloadForwardBuffers(
                count, self.offload.entry_bytes, device
            )
            return self._offload_buffers
        if buffers.capacity >= count:
            return buffers
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"engram offload buffers (capacity {buffers.capacity}) cannot "
                f"serve a capture of {count} ids; preallocation undersized "
                "the graph batch"
            )
        # Oversized eager call (outside capture): a throwaway buffer keeps the
        # frozen one intact — address stability is only required for replays.
        logger.warning(
            "[engram_offload] rank %d: layer %d eager lookup of %d ids exceeds "
            "frozen capacity %d; using a throwaway buffer",
            self._offload_manager.rank,
            self.offload.layer_hash_index,
            count,
            buffers.capacity,
        )
        return _OffloadForwardBuffers(count, self.offload.entry_bytes, device)

    def _forward_offload(self, indices: torch.Tensor) -> torch.Tensor:
        # entry_gather executes any uint32 row count (no batch cap); the
        # id/staging buffers are pre-sized once (finalize_offload) and frozen,
        # so graph-captured addresses stay valid across every batch size.
        assert self._offload_finalized, "engram offload rows are not flushed yet"
        num_tokens, num_cols = indices.shape
        table = self.offload
        count = indices.numel()
        # indices are already the layer-table row numbers: the kernel maps
        # row -> (slot, in-slot row) from the registered layout itself. The
        # per-layer slice of hash_ids is strided ([T, L, H] cut on dim 1), so
        # it always lands in the fixed buffers.ids first — the kernel reads
        # ids as a flat array. copy_ needs matching shapes (it broadcasts, it
        # does not reinterpret), hence viewing the flat prefix as [T, H]; the
        # strided source is still handled in one d2d pass.
        if count:
            # Only touch the buffers on non-empty calls: an empty lookup must
            # not size-on-first-use a zero-capacity buffer (it would freeze
            # the whole layer onto throwaway buffers afterwards).
            buffers = self._ensure_offload_buffers(count, indices.device)
            buffers.ids[:count].view(indices.shape).copy_(indices)
            buffers.count.fill_(count)
            self._offload_calls += 1
            if (
                self._offload_calls == 1
                or self._offload_calls % _OFFLOAD_LOG_EVERY == 0
            ):
                logger.info(
                    "[engram_offload] rank %d: layer %d entry_gather call #%d "
                    "(%d ids, table_id=%d, capacity=%d, staging=%d x %dB)",
                    self._offload_manager.rank,
                    self.offload.layer_hash_index,
                    self._offload_calls,
                    count,
                    table.table_id,
                    buffers.capacity,
                    count,
                    table.entry_bytes,
                )
            assert (
                self._offload_manager.entry_gather(
                    buffers.staging,
                    buffers.ids,
                    buffers.count,
                    table.table_id,
                    indices.device,
                )
                == 0
            )
            staged = buffers.staging[: count * table.entry_bytes]
        else:
            staged = indices.new_empty(0, dtype=torch.uint8)
        if not self._is_arch35:
            # A3: bf16 rows gathered verbatim — no dequant, no scale.
            return staged.view(torch.bfloat16).reshape(
                num_tokens, num_cols, table.head_dim
            )
        # A5 (arch35): dequantize the staged fp8 rows with the same math as
        # the device path; each staged entry holds exactly one unpadded fp8
        # row. The row width is spelled out (not -1) so empty batches
        # (num_tokens 0, e.g. a DP-attention idle dummy forward) reshape
        # unambiguously, and no gather is launched for zero ids.
        rows = (
            staged.view(torch.float8_e4m3fn)
            .reshape(num_tokens, num_cols, table.head_dim)
            .float()
            .unflatten(-1, (-1, FP8_BLOCK_SIZE))
        )
        # NPU aclnnIndex does not support float8_e8m0fnu; view as uint8 first
        # (same trick as _lookup), index, then cast — identical math, working op.
        # e8m0 stores the raw exponent; the actual scale factor is 2^(e-127),
        # matching _lookup's conversion.
        scales = torch.pow(2.0, self.scale.view(torch.uint8)[indices].to(torch.float32) - 127.0).unsqueeze(-1)
        return (rows * scales).flatten(-2).to(torch.bfloat16)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [..., n_cols] -> bf16 [.., n_cols, head_dim], zeroed on rows
        this rank's shard does not own."""
        local = indices - self.row_start
        owned = (local >= 0) & (local < self.rows)
        local = local.masked_fill(~owned, 0)
        if not is_npu_arch35():
            return self.weight[local].to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        else:
            w = self.weight.view(torch.uint8)[local].view(torch.float8_e4m3fn)
            s_raw = self.scale.view(torch.uint8)[local].to(torch.float32)
            s = torch.pow(2.0, s_raw - 127.0)
            rows = w.float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
            values = (rows * s.unsqueeze(-1)).flatten(-2)
            return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)

    def forward(
        self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        if self.offload is not None:
            # Offload path gathers from the node-local pool and computes the
            # full value on every rank: no collective, so it must bypass the
            # DP gather/scatter + all_reduce below entirely.
            return self._forward_offload(indices)
        if (
            self.tp_size > 1
            and get_attention_dp_size() > 1
        ):
            # moe_dense_tp style: allgather the DP domain first so the
            # all_reduce below has the same [global_tokens, ...] shape on
            # every rank (a bare all_reduce on the local tensor would
            # mismatch across DP groups and deadlock HCCL).
            global_indices = indices.new_zeros(
                (get_global_dp_buffer_len(), *indices.shape[1:])
            )
            # Clone: the MAX_LEN gather may zero its local input in place.
            dp_gather_replicate(global_indices, indices.clone(), forward_batch)
            values = tensor_model_parallel_all_reduce(self._lookup(global_indices))
            out = values.new_zeros((indices.shape[0], *values.shape[1:]))
            dp_scatter(out, values, forward_batch)
            return out
        values = self._lookup(indices)
        if self.tp_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values


def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    """x [T, hc_mult, dim]; kv [T, (hc_mult + 1) * dim] holds one key per hc copy
    followed by the shared value. Adds the gated value to every copy."""
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    # Normalized per (token, hc copy) over dim, not jointly over the copies.
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    # Signed square root before the sigmoid, matching the training kernel.
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        dim, hc_mult = config.hidden_size, config.hc_mult
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            layer_hash_index=self.layer_hash_index,
            config=config,
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            dim * (hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] for this layer."""
        kv, _ = self.wkv(self.embed(hash_ids, forward_batch).flatten(-2))
        return engram_gate(
            x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value
        )
