"""Engram: n-gram hash lookups written into the residual stream at a few layers.

A position is hashed as max_ngram_size - 1 n-grams (2-gram .. max_ngram_size-gram), each
split over n_heads heads; every (n-gram size, head) pair owns a prime-sized bucket range
of the layer's table. Hashing runs over a compressed token id space where tokens that
normalize alike (case, accents, whitespace) collapse together.
"""

import msgspec
import numpy as np
import torch
import torch.nn.functional as F
from dsv41_args import DeepseekV41Args
from dsv41_linear import Linear
from torch import nn

from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE


def find_next_prime(start: int, seen_primes: set[int]) -> int:
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
    def from_args(cls, args: DeepseekV41Args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        return cls.build(
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            max_ngram_size=args.engram_max_ngram_size,
            n_heads=args.engram_n_heads,
            head_dim=args.engram_head_dim,
            vocab_size=args.engram_vocab_size,
        )

    @classmethod
    def build(
        cls,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
    ) -> "EngramLayout":
        """Primes are drawn in (layer, n-gram size, head) order from one shared
        ascending sequence starting above vocab_size - 1."""
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
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


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there. Look-back stops at
    the sequence start and at any dead token (image span); the cache carries compressed
    ids across the prefill / decode split."""

    DEAD = -1

    def __init__(self, args: DeepseekV41Args, layout: EngramLayout, tokenizer):
        super().__init__()
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == args.engram_compressed_vocab_size, (
            vocab_size,
            args.engram_compressed_vocab_size,
        )
        self.pad_id = token_map[args.engram_pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer(
            "offsets", torch.tensor(np.array(offsets)), persistent=False
        )
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer(
            "cache",
            torch.empty(args.max_batch_size, args.max_seq_len, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        token_mask: torch.Tensor | None = None,
    ):
        """token_mask [B, L]: False for tokens that take no part in an n-gram.
        Returns hash ids [B, L, n_engram_layers, n_hash_cols]."""
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(
            start_pos, start_pos + seqlen, device=input_ids.device
        ).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)

        # XOR the multiplied ids one lookback at a time: after step i the running value is
        # the (i + 1)-gram hash, placed in its own prime-sized bucket range.
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class EngramEmbedding(nn.Module):
    """The hash table stays fp8 as stored; rows are dequantized with `scale` on lookup."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.block_size = FP8_BLOCK_SIZE
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, dim, dtype=torch.float8_e4m3fn)
        )
        self.scale = nn.Parameter(
            torch.empty(
                num_embeddings, dim // self.block_size, dtype=torch.float8_e8m0fnu
            )
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        values = F.embedding(indices, self.weight)
        scales = F.embedding(indices, self.scale)
        values = values.float().unflatten(
            -1, (-1, self.block_size)
        ) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


class Engram(nn.Module):
    """Adds a gated n-gram lookup to every hc copy of the residual stream. wkv turns the
    fetched rows into one key per copy plus one shared value; the gate is a normalized
    dot product of stream against key."""

    def __init__(self, args: DeepseekV41Args, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = Linear(
            n_hash_cols * layout.head_dim,
            args.dim * (args.hc_mult + 1),
            dtype=torch.float8_e4m3fn,
        )
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ):
        """x [B, L, hc_mult, dim]; hash_ids [B, L, n_hash_cols]; token_mask [B, L], False
        shuts the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h, eps = x.float(), self.eps
        # Normalized per (token, hc copy) over dim, not jointly over the copies.
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
            key.square().mean(-1) + eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # Signed square root before the sigmoid, matching the training kernel.
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)
        )
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
