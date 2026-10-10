"""Engram: gated n-gram hash memory added to the hc residual stream.

The token map, hash multipliers and prime layout must match
sglang.kernels.ops.embeddings.engram_hash to produce identical hash ids.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.kernels.ops.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    MODE_EXTEND,
    MODE_VERIFY,
    engram_commit_history,
    engram_hash_ids,
    engram_hash_ids_and_commit,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import inplace_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_global_dp_buffer_len,
    is_dp_gatherv_active,
)
from sglang.srt.layers.engram_table import EngramTableLayout, create_engram_table
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_model, get_parallel, get_serving
from sglang.srt.utils import add_prefix, is_cuda, is_gfx95_supported, is_hip
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

_is_hip = is_hip()

_MILLER_RABIN_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _cuda_kernels(t: torch.Tensor) -> bool:
    """True where the Triton kernels apply (CUDA and gfx950); other ROCm GPUs and CPU
    take the torch paths."""
    return t.is_cuda and (is_cuda() or is_gfx95_supported())


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin; exact for n < 3.3e24 with these witnesses."""
    if n < 2:
        return False
    for p in _MILLER_RABIN_WITNESSES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in _MILLER_RABIN_WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
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
    def from_config(cls, config) -> Optional[EngramLayout]:
        """Primes are drawn in (layer, n-gram size, head) order from one shared
        ascending sequence starting above engram_vocab_size - 1."""
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = config.engram_max_ngram_size, config.engram_n_heads
        vocab_size = config.engram_vocab_size
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
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=config.engram_head_dim,
        )

    def column_bounds(self, hash_index: int) -> tuple[int, ...]:
        """Prefix sums of one layer's column segments, in the hash's (n-gram size, head)
        column order: hash column c owns rows [bounds[c], bounds[c + 1])."""
        bounds = [0]
        for per_ngram in self.primes[hash_index]:
            for prime in per_ngram:
                bounds.append(bounds[-1] + prime)
        return tuple(bounds)


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
    # After step i the running xor is the (i + 1)-gram hash, bucketed by that
    # n-gram size's primes.
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
        self.image_token_id: Optional[int] = None
        self.history: Optional[torch.Tensor] = None
        self.pad_row = 0
        # typing only
        self.token_map: torch.Tensor
        self.multipliers: torch.Tensor
        self.primes: torch.Tensor
        self.offsets: torch.Tensor

    def init_history(self, num_req_slots: int, device) -> None:
        """Allocate oldest-first history with a spare row for graph padding."""
        self.history = torch.zeros(
            num_req_slots + 1,
            self.max_ngram_size - 1,
            dtype=torch.int32,
            device=device,
        )
        self.pad_row = num_req_slots

    @classmethod
    def from_config(
        cls, config, layout: EngramLayout, *, image_token_id: Optional[int] = None
    ) -> EngramHasher:
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
            config.engram_pad_token_id,
            config.engram_compressed_vocab_size,
        )
        result.image_token_id = image_token_id
        return result

    def forward(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        assert self.history is not None, "EngramHasher.init_history was not called"
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        if num_tokens == 0:
            return torch.empty(
                (0, self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
                device=input_ids.device,
            )
        mode = forward_batch.forward_mode
        req_slots = forward_batch.req_pool_indices
        bs = req_slots.shape[0]
        device = input_ids.device
        # Tokens at or past num_real are graph padding. History comes from
        # self.history via req_slots unless the scheduler supplied this extend's rows.
        num_real, block, row, starts = num_tokens, 1, None, None
        history, hist_via_slots = self.history, True
        if mode.is_decode():
            kmode = MODE_DECODE
            commit_rows, commit_last = req_slots, None
        elif mode.is_target_verify():
            block = int(forward_batch.spec_info.draft_token_num)
            assert num_tokens == bs * block, (
                "engram target-verify expects one equal block per request, got "
                f"{num_tokens} tokens for {bs} requests of {block}"
            )
            kmode = MODE_VERIFY
            commit_rows = commit_last = None
        else:
            assert mode.is_extend(), (
                f"engram serves extend, target-verify and decode, not {mode}"
            )
            lens = forward_batch.extend_seq_lens.to(torch.int64)
            starts = forward_batch.extend_start_loc.to(torch.int64)
            lens_cpu = forward_batch.extend_seq_lens_cpu
            row = torch.repeat_interleave(
                torch.arange(bs, device=device),
                lens,
                # The host total skips the device sum's sync.
                output_size=sum(lens_cpu) if lens_cpu is not None else None,
            )
            num_real = row.shape[0]
            kmode = MODE_EXTEND
            if forward_batch.engram_history is not None:
                history, hist_via_slots = forward_batch.engram_history, False
            commit_rows = torch.where(lens > 0, req_slots, self.pad_row)
            commit_last = (starts + lens - 1).clamp(0, num_tokens - 1)

        if _cuda_kernels(input_ids):
            if kmode == MODE_DECODE:
                # out_cache_loc 0 marks the CUDA-graph padded rows that must not commit.
                assert forward_batch.out_cache_loc is not None
                return engram_hash_ids_and_commit(
                    input_ids,
                    forward_batch.positions,
                    history=self.history,
                    req_slots=req_slots,
                    out_cache_loc=forward_batch.out_cache_loc,
                    token_map=self.token_map,
                    multipliers=self.multipliers,
                    primes=self.primes,
                    offsets=self.offsets,
                    pad_id=self.pad_id,
                    image_token_id=self.image_token_id,
                    mm_pad_shift=MM_PAD_SHIFT_VALUE,
                )
            hash_ids, tokens = engram_hash_ids(
                input_ids,
                forward_batch.positions,
                mode=kmode,
                history=history,
                token_map=self.token_map,
                multipliers=self.multipliers,
                primes=self.primes,
                offsets=self.offsets,
                pad_id=self.pad_id,
                num_real=num_real,
                req_slots=req_slots if hist_via_slots else None,
                block=block,
                row=row,
                starts=starts,
                image_token_id=self.image_token_id,
                mm_pad_shift=MM_PAD_SHIFT_VALUE,
            )
        else:
            hash_ids, tokens = self._torch_hash_ids(
                input_ids,
                forward_batch.positions,
                kmode,
                history[req_slots] if hist_via_slots else history,
                num_real,
                block,
                row,
                starts,
            )
        if commit_rows is not None:
            # Padded rows must not overwrite a live request's history.
            last_tokens = tokens if commit_last is None else tokens[commit_last]
            out_loc = forward_batch.out_cache_loc
            if out_loc is not None:
                if commit_last is not None:
                    out_loc = out_loc[commit_last]
                commit_rows = torch.where(out_loc == 0, self.pad_row, commit_rows)
            self.history[commit_rows] = (
                last_tokens[:, : n - 1].flip(-1).to(self.history.dtype)
            )
        return hash_ids

    def _torch_hash_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kmode: int,
        history: torch.Tensor,
        num_real: int,
        block: int,
        row: Optional[torch.Tensor],
        starts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-CUDA fallback of the hash kernel: predecessor table [T, n] and hash ids.
        ``history`` is already the per-request [bs, n - 1] rows of this batch."""
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        device = input_ids.device
        input_ids = input_ids.to(torch.int64)
        shifts = torch.arange(n, device=device)
        if kmode == MODE_DECODE:
            tokens = torch.cat(
                [input_ids.unsqueeze(-1), history.to(torch.int64).flip(-1)], dim=-1
            )
        else:
            t = torch.arange(num_real, device=device)
            if kmode == MODE_VERIFY:
                row = t // block
                offset = t - row * block
            else:
                offset = t - starts[row]
            # Predecessor at shift s of token t: an earlier token of the same run
            # when s <= offset, else history[row, -(s - offset)].
            from_batch = shifts.unsqueeze(0) <= offset.unsqueeze(-1)
            in_batch = input_ids[(t.unsqueeze(-1) - shifts).clamp_min(0)]
            hist_col = (n - 2 - (shifts.unsqueeze(0) - offset.unsqueeze(-1) - 1)).clamp(
                0, n - 2
            )
            from_hist = history.to(torch.int64)[row].gather(1, hist_col)
            tokens = torch.where(from_batch, in_batch, from_hist)
        if num_real < num_tokens:
            tokens = torch.cat([tokens, tokens.new_zeros(num_tokens - num_real, n)])
        positions = positions.to(torch.int64)
        blocked = positions.unsqueeze(-1) < shifts
        if num_real < num_tokens:
            blocked[num_real:] = True
        if self.image_token_id is not None:
            # Scheduler-provided history still carries the multimodal pad ids.
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
        hash_ids = compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )
        return hash_ids, tokens

    def commit_after_verify(
        self,
        verify_ids_2d: torch.Tensor,
        req_pool_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit anchor + accepted drafts; the bonus is the next block's anchor."""
        assert self.history is not None, "EngramHasher.init_history was not called"
        if _cuda_kernels(self.history):
            engram_commit_history(
                self.history, verify_ids_2d, req_pool_indices, commit_lens
            )
            return
        n1 = self.max_ngram_size - 1
        req = req_pool_indices.to(torch.int64)
        window = torch.cat(
            [self.history[req], verify_ids_2d.to(self.history.dtype)], dim=1
        )
        cols = commit_lens.to(torch.int64).unsqueeze(-1) + torch.arange(
            n1, device=window.device
        )
        self.history[req] = window.gather(1, cols)


class EngramEmbedding(nn.Module):
    """One layer's fp8 hash table with e8m0 block scales, dequantized on lookup.

    Storage and layout come from engram_table: device or host
    (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE), and per SGLANG_DSV41_ENGRAM_TABLE_LAYOUT
    either one full_shared copy or row_sharded column shards. Loading is sharded
    in every layout: a rank writes only its own row range.
    """

    def __init__(
        self, num_embeddings: int, dim: int, layer_id: int, bounds: tuple[int, ...]
    ):
        assert 0 == bounds[0] and num_embeddings == bounds[-1]
        assert dim % FP8_BLOCK_SIZE == 0
        super().__init__()
        self.dim = dim
        self.tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.use_host = envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()
        layout = EngramTableLayout.parse(
            self.use_host, envs.SGLANG_DSV41_ENGRAM_TABLE_LAYOUT.get()
        )
        self.row_start, self.row_end = layout.shard(bounds, self.tp_size, tp_rank)
        self.num_rows = self.row_end - self.row_start
        # Loading is sharded in every layout: a full_shared table is one buffer
        # the ranks fill together, each writing its row_sharded slice.
        self._load_slice = slice(
            *EngramTableLayout.ROW_SHARDED.shard(bounds, self.tp_size, tp_rank)
        )
        self.table = create_engram_table(
            use_host=self.use_host,
            layout=layout,
            nbytes=self.num_rows * (dim + (dim // FP8_BLOCK_SIZE)),
            name=f"sglang_engram_{layer_id}",
            group=get_parallel().tp_group,
        )
        raw = self.table.bytes
        n, d = self.num_rows, dim
        weight = raw[: n * d].view(torch.float8_e4m3fn).view(n, d)
        scale = raw[n * d :].view(torch.float8_e8m0fnu).view(n, d // FP8_BLOCK_SIZE)
        device_ptr = self.table.device_ptr
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows
        self._table_ptrs = (device_ptr, device_ptr + n * d)

    @property
    def _shared(self) -> bool:
        return self.table.layout == EngramTableLayout.FULL_SHARED

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        rows = self._load_slice
        if self._shared:
            param.data[rows].copy_(loaded_weight[rows])
        else:
            param.data.copy_(loaded_weight[rows])
        self.table.mark_loaded()

    def finish_load(self, label: str):
        """Collective in the shared layout: every rank calls it after loading."""
        self.table.finish_load(label)

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        if self._shared:
            if indices.shape[0] == 0:
                return self._empty(indices)
            out = self._empty(indices)
            engram_gather(
                *self._table_ptrs,
                indices.reshape(-1),
                out.view(-1, self.dim),
                self.dim,
                FP8_BLOCK_SIZE,
            )
            return out
        if cp_all_tokens and self.tp_size > 1:
            # Prefill CP: gather the hash ids over the CP group first so every
            # TP rank looks up the same indices, then keep this rank's slice.
            parallel = get_parallel()
            local_rows = indices.shape[0]
            all_indices = indices.new_empty(
                (parallel.attn_cp_size * local_rows, *indices.shape[1:])
            )
            attn_cp_all_gather_into_tensor(all_indices, indices.contiguous())
            start = parallel.attn_cp_rank * local_rows
            return self._lookup(all_indices)[start : start + local_rows]
        if self.tp_size > 1 and get_parallel().attn_dp_size > 1:
            return self._dp_sharded_lookup(indices, forward_batch)
        return self._lookup(indices)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup when every TP rank holds the same indices: the device and
        per-rank host shards zero unowned rows and the all-reduce reassembles."""
        if indices.shape[0] == 0:
            return self._empty(indices)
        values = self._owned_rows(indices)
        if self.tp_size > 1:
            values = self._reduce_owned_rows(values)
        return values

    def _reduce_owned_rows(self, values: torch.Tensor) -> torch.Tensor:
        if _is_hip and values.is_cuda:
            # Integer addition preserves all BF16 bits because exactly one shard owns each row.
            inplace_all_reduce(
                values.view(torch.int32), group_name=get_parallel().tp_group.unique_name
            )
            return values
        return tensor_model_parallel_all_reduce(values)

    def _empty(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            *indices.shape, self.dim, dtype=torch.bfloat16, device=indices.device
        )

    def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Rows of `indices` this rank's shard holds, zero for the rest."""
        if self.num_rows == 0:
            return self._empty(indices).zero_()
        if not _cuda_kernels(indices):
            local = indices - self.row_start
            owned = (local >= 0) & (local < self.num_rows)
            local = local.masked_fill(~owned, 0)
            rows = self.weight[local].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
            values = (rows * self.scale[local].float().unsqueeze(-1)).flatten(-2)
            return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        out = self._empty(indices)
        engram_gather(
            *self._table_ptrs,
            indices.reshape(-1),
            out.view(-1, self.dim),
            self.dim,
            FP8_BLOCK_SIZE,
            row_lo=self.row_start,
            row_hi=self.row_end,
        )
        return out

    def _dp_sharded_lookup(
        self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch]
    ) -> torch.Tensor:
        """Gather DP ranks' indices before looking up TP-sharded rows."""
        assert forward_batch is not None, "the DP engram lookup needs the batch"
        rows = get_global_dp_buffer_len()
        ids_global = torch.empty(
            (rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device
        )
        # The MAX_LEN gather may zero its local input in place, hence the clone.
        dp_gather_replicate(ids_global, indices.clone(), forward_batch)
        if rows == 0:
            return self._empty(indices)
        values = self._owned_rows(ids_global).view(rows, -1)
        local = torch.empty(
            (indices.shape[0], values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        padding = forward_batch.dp_padding_mode
        if (
            padding is not None
            and padding.is_max_len()
            and self.tp_size == get_parallel().attn_dp_size
            and rows == self.tp_size * local.shape[0]
        ) or is_dp_gatherv_active():
            if _is_hip and values.is_cuda:
                dp_reduce_scatter_tensor(
                    local.view(torch.int32), values.view(torch.int32)
                )
            else:
                dp_reduce_scatter_tensor(local, values)
        else:
            dp_scatter(local, self._reduce_owned_rows(values), forward_batch)
        return local.view(*indices.shape, self.dim)


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
    if (
        _cuda_kernels(x)
        and x.ndim == 3
        and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and q_weight.dtype in (torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
    ):
        return fused_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
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
            layer_id,
            layout.column_bounds(self.layer_hash_index),
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
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] for this layer."""
        # The lookup runs first even for an idle DP-attention batch: under DP
        # attention it is a collective every rank has to join.
        emb = self.embed(hash_ids, forward_batch, cp_all_tokens=cp_all_tokens)
        if x.shape[0] == 0:
            # Nothing to gate, and the MXFP8 quantize behind wkv rejects an
            # empty M.
            return x
        kv, _ = self.wkv(emb.flatten(-2))
        return engram_gate(
            x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value
        )
