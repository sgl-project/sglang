"""Engram module — n-gram hash-based conditional memory for LLMs.

This is a prototype implementation demonstrating the core Engram architecture.
Standard Transformer components (attention, MoE, hyper-connections) are mocked
to focus on the Engram-specific logic. Production use requires further
optimisation (custom CUDA kernels, distributed training support, etc.).

Dependencies: torch numpy transformers sympy tokenizers
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer

from sglang.srt.mem_cache.engram.local_engram_store import LocalEngramStore

logger = logging.getLogger(__name__)


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [1024, 1024])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    store_backend: str = "local"
    enable_prefetch: bool = True


@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30


engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


def build_engram_config_from_cli() -> EngramConfig:
    parser = argparse.ArgumentParser(description="Engram demo config")
    parser.add_argument(
        "--store-backend",
        type=str,
        default=None,
        choices=["local"],
    )
    parser.add_argument("--engram-vocab-size", type=int, default=None)
    parser.add_argument("--engram-emb-size", type=int, default=None)
    parser.add_argument("--engram-head", type=int, default=None)
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.store_backend:
        cfg.store_backend = args.store_backend
    if args.engram_vocab_size:
        for i in range(len(cfg.engram_vocab_size)):
            cfg.engram_vocab_size[i] = args.engram_vocab_size
    if args.engram_emb_size:
        cfg.n_embed_per_ngram = args.engram_emb_size
    if args.engram_head:
        cfg.n_head_per_ngram = args.engram_head
    return cfg


class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()
        self._lookup_table_torch = {}

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        if valid_ids.size == 0:
            return out
        vocab_size = self.lookup_table.shape[0]
        if valid_ids.max(initial=-1) >= vocab_size:
            if engram_cfg.pad_id is not None:
                valid_ids = np.where(
                    valid_ids < vocab_size, valid_ids, int(engram_cfg.pad_id)
                )
            else:
                valid_ids = np.clip(valid_ids, 0, vocab_size - 1)
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def _get_lookup_table_torch(self, device: torch.device) -> torch.Tensor:
        table = self._lookup_table_torch.get(device)
        if table is None:
            table = torch.from_numpy(self.lookup_table).to(device=device)
            self._lookup_table_torch[device] = table
        return table

    def compress_torch(self, input_ids: torch.Tensor) -> torch.Tensor:
        arr = input_ids.to(dtype=torch.long)
        pos_mask = arr >= 0
        out = arr.clone()
        if not pos_mask.any():
            return out
        lookup_table = self._get_lookup_table_torch(arr.device)
        valid_ids = arr[pos_mask]
        vocab_size = lookup_table.shape[0]
        if valid_ids.max().item() >= vocab_size:
            if engram_cfg.pad_id is not None:
                pad_id = int(engram_cfg.pad_id)
                valid_ids = torch.where(
                    valid_ids < vocab_size,
                    valid_ids,
                    torch.full_like(valid_ids, pad_id),
                )
            else:
                valid_ids = valid_ids.clamp(0, vocab_size - 1)
        out[pos_mask] = lookup_table[valid_ids]
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


class ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norm = nn.RMSNorm(total_channels, eps=norm_eps)

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input/Output: (B, T, HC_MULT, D)"""
        B, T, G, C = x.shape
        x = x.view(B, T, G * C)
        x = self.norm(x)
        x = x.transpose(1, 2)
        y = self.conv(x)
        y = y[..., :T]
        if self.activation:
            y = self.act_fn(y)
        y = y.transpose(1, 2).view(B, T, G, C)
        return y


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()
        self._layer_multipliers_torch = {}

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def _get_layer_multipliers_torch(
        self, layer_id: int, device: torch.device
    ) -> torch.Tensor:
        key = (layer_id, device)
        multipliers = self._layer_multipliers_torch.get(key)
        if multipliers is None:
            multipliers = torch.as_tensor(
                self.layer_multipliers[layer_id],
                dtype=torch.long,
                device=device,
            )
            self._layer_multipliers_torch[key] = multipliers
        return multipliers

    def _get_ngram_hashes_torch(
        self,
        input_ids: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        x = input_ids.to(dtype=torch.long)
        batch_size, seq_len = x.shape
        multipliers = self._get_layer_multipliers_torch(layer_id, x.device)

        base_shifts = []
        for k in range(self.max_ngram_size):
            if k == 0:
                shifted = x
            else:
                shifted = F.pad(x, (k, 0), value=int(self.pad_id))
                shifted = shifted[:, :seq_len]
            base_shifts.append(shifted)

        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for mod in head_vocab_sizes:
                head_hash = torch.remainder(mix, int(mod))
                all_hashes.append(head_hash)

        return torch.stack(all_hashes, dim=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                input_ids, layer_id=layer_id
            )
        return hash_ids_for_all_layers

    def hash_torch(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        input_ids = self.compressed_tokenizer.compress_torch(input_ids)
        return self._get_ngram_hashes_torch(input_ids, layer_id=layer_id)


class MultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        list_of_N: List[int],
        layer_id: int,
        D: int,
        vocab_table: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float16,
        store=None,
    ):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.layer_id = layer_id

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)

        if store is not None:
            self.store = store
        else:
            self.store = LocalEngramStore(
                embedding_dim=D,
                vocab_size=total_N,
                layer_id=layer_id,
                dtype=dtype,
                device=torch.device("cpu"),
            )
        if vocab_table is None:
            raise ValueError("vocab_table must be provided for put_sharded")
        self.store.put_sharded(vocab_table)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        offsets = self.offsets.to(device=input_ids.device)
        shifted_input_ids = input_ids + offsets
        output = self.store.get_many(
            shifted_input_ids, self.layer_id, device=input_ids.device
        )
        return output


class Engram(nn.Module):
    def __init__(
        self,
        layer_id: int,
        vocab_table: Optional[torch.Tensor] = None,
        store_manager=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.enable_prefetch = engram_cfg.enable_prefetch
        self._prefetch_embeddings: Optional[torch.Tensor] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self._prefetch_shape: Optional[tuple[int, int]] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        logger.info(
            "Engram init: layer_id=%d, hidden_size=%d, hc_mult=%d, "
            "vocab_size=%d, num_layers=%d, max_ngram_size=%d, "
            "n_embed_per_ngram=%d, n_head_per_ngram=%d, "
            "engram_vocab_size=%s, kernel_size=%d, store_backend=%s",
            layer_id,
            backbone_config.hidden_size,
            backbone_config.hc_mult,
            backbone_config.vocab_size,
            backbone_config.num_layers,
            engram_cfg.max_ngram_size,
            engram_cfg.n_embed_per_ngram,
            engram_cfg.n_head_per_ngram,
            engram_cfg.engram_vocab_size,
            engram_cfg.kernel_size,
            engram_cfg.store_backend,
        )
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
        )

        list_of_N = [
            x
            for y in self.hash_mapping.vocab_size_across_layers[self.layer_id]
            for x in y
        ]
        logger.info("Engram layer %d vocab sizes: %s", self.layer_id, list_of_N)

        total_N = sum(list_of_N)
        embedding_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        vocab_table = (
            torch.arange(total_N, dtype=torch.float16)
            .unsqueeze(1)
            .expand(-1, embedding_dim)
        )

        engram_store = None
        if store_manager is not None:
            engram_store = store_manager.get_or_create_store(
                layer_id=self.layer_id,
                vocab_size=total_N,
                embedding_dim=embedding_dim,
            )

        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            layer_id=self.layer_id,
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
            vocab_table=vocab_table,
            store=engram_store,
        )
        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )
        engram_hidden_size = (
            engram_cfg.max_ngram_size - 1
        ) * engram_cfg.n_embed_per_ngram
        self.hc_mult = backbone_config.hc_mult
        self.hidden_size = backbone_config.hidden_size
        self.eps = 1e-6

        self.value_proj = nn.Linear(engram_hidden_size, self.hidden_size)
        self.key_projs_all = nn.Linear(
            engram_hidden_size, self.hidden_size * self.hc_mult
        )
        self.norm1_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size))
        self.norm2_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size))

    def start_prefetch(
        self,
        input_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not self.enable_prefetch:
            return

        if input_ids.device != device:
            input_ids = input_ids.to(device=device)

        if self._prefetch_stream is not None and device.type == "cuda":
            stream = self._prefetch_stream
            with torch.cuda.stream(stream):
                embeddings = self.compute_embeddings(input_ids, dtype=dtype)
            event = torch.cuda.Event()
            event.record(stream)
            self._prefetch_event = event
            self._prefetch_embeddings = embeddings
        else:
            embeddings = self.compute_embeddings(input_ids, dtype=dtype)
            self._prefetch_event = None
            self._prefetch_embeddings = embeddings

        self._prefetch_shape = tuple(input_ids.shape[:2])

    def _consume_prefetch(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_prefetch:
            return None
        if self._prefetch_embeddings is None:
            return None
        if (
            self._prefetch_shape is not None
            and tuple(input_ids.shape[:2]) != self._prefetch_shape
        ):
            self._prefetch_embeddings = None
            self._prefetch_event = None
            self._prefetch_shape = None
            return None

        if self._prefetch_event is not None:
            torch.cuda.current_stream().wait_event(self._prefetch_event)
        embeddings = self._prefetch_embeddings
        self._prefetch_embeddings = None
        self._prefetch_event = None
        self._prefetch_shape = None
        return embeddings

    def parallel_rms_norm(self, x, weight):
        # x shape: (batch, seq, hc_mult, hidden_size)
        # weight shape: (hc_mult, hidden_size)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * weight

    # ------------------------------------------------------------------
    # Stage A: hash + embedding lookup (input_ids → embeddings)
    # Depends only on input_ids; safe to run ahead of time or on a
    # different device backend.  Subclasses may override to use custom
    # kernels (e.g. CPU-offloaded or fused CUDA implementations).
    # ------------------------------------------------------------------
    def compute_embeddings(
        self,
        input_ids: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Stage A: token compression → n-gram hashing → embedding lookup.

        Args:
            input_ids: ``[B, L]`` token ids.
            dtype: cast output to this dtype when provided.

        Returns:
            embeddings: ``[B, L, (max_ngram_size-1) * n_embed_per_ngram]``
        """
        hash_input_ids = self.hash_mapping.hash_torch(
            input_ids, layer_id=self.layer_id
        )
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        if dtype is not None and embeddings.dtype != dtype:
            embeddings = embeddings.to(dtype=dtype)
        return embeddings

    # ------------------------------------------------------------------
    # Stage B: key/value projections + norms (embeddings → keys, values)
    # Depends only on embeddings (Stage A output); independent of the
    # current hidden_states.  Subclasses may override for fused or
    # quantised projection kernels.
    # ------------------------------------------------------------------
    def compute_kv_projections(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage B: linear projections and RMS-norm for keys and values.

        Args:
            embeddings: ``[B, L, engram_hidden_size]``

        Returns:
            normed_key: ``[B, L, HC_MULT, D]`` — normalised key tensor.
            v_projected: ``[B, L, 1, D]``       — projected value tensor.
        """
        keys = self.key_projs_all(embeddings).view(
            *embeddings.shape[:2], self.hc_mult, self.hidden_size
        )
        normed_key = self.parallel_rms_norm(keys, self.norm1_weight)
        v_projected = self.value_proj(embeddings).unsqueeze(2)
        return normed_key, v_projected

    # ------------------------------------------------------------------
    # Stage C: gated mixing + short convolution (hidden_states → output)
    # Depends on hidden_states (previous layer) and Stage A/B outputs.
    # Subclasses may override to use custom attention or conv kernels.
    # ------------------------------------------------------------------
    def compute_mixing(
        self,
        hidden_states: torch.Tensor,
        normed_key: torch.Tensor,
        v_projected: torch.Tensor,
    ) -> torch.Tensor:
        """Stage C: scaled dot-product gating and short convolution.

        Args:
            hidden_states: ``[B, L, HC_MULT, D]``
            normed_key:    ``[B, L, HC_MULT, D]``
            v_projected:   ``[B, L, 1, D]``

        Returns:
            output: ``[B, L, HC_MULT, D]``
        """
        normed_query = self.parallel_rms_norm(hidden_states, self.norm2_weight)

        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)

        value = gate * v_projected
        return value + self.short_conv(value)

    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        embeddings = self._consume_prefetch(input_ids)
        if embeddings is None:
            if input_ids.device != hidden_states.device:
                input_ids = input_ids.to(device=hidden_states.device)
            embeddings = self.compute_embeddings(input_ids)
        if embeddings.dtype != hidden_states.dtype:
            embeddings = embeddings.to(dtype=hidden_states.dtype)

        normed_key, v_projected = self.compute_kv_projections(embeddings)
        return self.compute_mixing(hidden_states, normed_key, v_projected)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.attn = lambda x: x
        self.moe = lambda x: x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)

    def forward(self, input_ids, hidden_states):
        if self.engram is not None:
            hidden_states = (
                self.engram(hidden_states=hidden_states, input_ids=input_ids)
                + hidden_states
            )
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


if __name__ == "__main__":
    engram_cfg = build_engram_config_from_cli()
    backbone_config = BackBoneConfig()

    LLM = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        *[TransformerBlock(layer_id=i) for i in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size),
    ]

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path, trust_remote_code=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    B, L = input_ids.shape

    for idx, layer in enumerate(LLM):
        if idx == 0:
            hidden_states = LLM[0](input_ids)
            # mock hyper-connection: expand to (B, L, HC_MULT, D)
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, backbone_config.hc_mult, -1
            )
        elif idx == len(LLM) - 1:
            # mock hyper-connection: collapse back to (B, L, D)
            hidden_states = hidden_states[:, :, 0, :]
            output = layer(hidden_states)
        else:
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)
