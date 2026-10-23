"""Scheduler-side staging of DeepSeek-V4.1 engram rows from the shared host table.

Without ATS the GPU maps the host table in its own page tables, and a gather of
random rows from a table far larger than its translation reach runs at a fraction
of PCIe bandwidth, on the forward stream. The scheduler knows every token of an
extend batch before its forward, so it hashes the n-grams on the CPU, gathers
each engram layer's fp8 rows and e8m0 scales into pinned memory, and copies them
to the GPU on a side stream while the previous forward runs. The layer then only
dequantizes contiguous device rows (EngramEmbedding.dequant_staged).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.engram import (
    Engram,
    EngramEmbedding,
    EngramHasher,
    compute_engram_hash_ids,
    mask_image_lookback,
)
from sglang.srt.model_executor.encoder_swa_replay import replay_len
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

# Per engram layer: device fp8 rows, device e8m0 scales, and the event of their copy.
StagedRows = list[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]]

_GATHER_THREADS = 16


class EngramRowPrefetcher:
    def __init__(
        self,
        hasher: EngramHasher,
        embeds: list[EngramEmbedding],
        device: torch.device,
    ):
        self.n = hasher.max_ngram_size
        self.pad_id = hasher.pad_id
        self.image_token_id = hasher.image_token_id
        self.token_map = hasher.token_map.cpu()
        self.multipliers = hasher.multipliers.cpu()
        self.primes = hasher.primes.cpu()
        self.offsets = hasher.offsets.cpu()
        # fp8 and e8m0 have no CPU index_select; the gather moves bytes.
        self.tables = [
            (e.weight.data.view(torch.uint8), e.scale.data.view(torch.uint8))
            for e in embeds
        ]
        self.device = device
        self.pool = ThreadPoolExecutor(_GATHER_THREADS)
        # The schedule stream waits on the running forward (the WAR barrier), so
        # copies there would not overlap it; each layer waits on its copy's event.
        self.copy_stream = torch.cuda.Stream(device)

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, hasher: EngramHasher, device: torch.device
    ) -> Optional[EngramRowPrefetcher]:
        """A prefetcher when every engram table is the shared host layout on a GPU
        without ATS: only that layout maps every row in this process, and with
        ATS the forward-stream gather already runs at full speed."""
        if not envs.SGLANG_OPT_DSV41_ENGRAM_HOST_PREFETCH.get():
            return None
        engrams = sorted(
            (m for m in model.modules() if isinstance(m, Engram)),
            key=lambda m: m.layer_hash_index,
        )
        embeds = [m.embed for m in engrams]
        if (
            not embeds
            # Prefill CP hashes the whole prompt and keeps this rank's rows.
            or get_parallel().attn_cp_size > 1
            or not all(e._shared and e._sort_host_reads for e in embeds)
        ):
            return None
        for e in embeds:
            e.host_prefetch = True
        return cls(hasher, embeds, device)

    def prefetch(self, batch: ScheduleBatch) -> Optional[StagedRows]:
        """Stage every engram layer's rows for an extend batch, None to leave the
        lookup to the forward."""
        ids = self._hash_ids(batch)
        if ids is None:
            return None
        staged = []
        for layer, (weight, scale) in enumerate(self.tables):
            layer_ids = ids[:, layer].reshape(-1)
            rows = self._gather(weight, layer_ids)
            scales = self._gather(scale, layer_ids)
            with torch.cuda.stream(self.copy_stream):
                rows = rows.to(self.device, non_blocking=True)
                scales = scales.to(self.device, non_blocking=True)
                copied = torch.cuda.Event()
                copied.record()
            staged.append((rows, scales, copied))
        return staged

    def _hash_ids(self, batch: ScheduleBatch) -> Optional[torch.Tensor]:
        """The forward's hash ids [T, n_engram_layers, n_hash_cols], on the CPU."""
        if not batch.forward_mode.is_extend_without_speculative():
            return None
        windows = self._windows(batch)
        if windows is None:
            return None
        tokens, blocked = windows
        if self.image_token_id is not None:
            tokens, blocked = mask_image_lookback(tokens, blocked, self.image_token_id)
        return compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )

    def _windows(
        self, batch: ScheduleBatch
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """tokens [T, n], column s the s-th predecessor, and blocked [T, n], the
        look-back past the sequence start. None unless every extend token is on
        the host."""
        n1 = self.n - 1
        tokens, positions = [], []
        # The forward's rows: an encoder SWA replay hit extends from its replay
        # start (fold_encoder_swa_replay).
        resets = batch.encoder_swa_reset or []
        replay = [replay_len(p) if r else 0 for r, p in zip(resets, batch.prefix_lens)]
        replay += [0] * (len(batch.reqs) - len(replay))
        for req, prefix, extend, head in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens, replay, strict=True
        ):
            start, length = prefix - head, extend + head
            fill_ids = req.full_untruncated_fill_ids
            if start + length > len(fill_ids):
                # A decode row's newest token can still be on the device.
                return None
            if length == 0:
                continue
            lo = max(0, start - n1)
            seg = np.asarray(fill_ids[lo : start + length], dtype=np.int64)
            seg = np.concatenate([np.zeros(n1 - (start - lo), np.int64), seg])
            tokens.append(
                np.lib.stride_tricks.sliding_window_view(seg, self.n)[:, ::-1]
            )
            positions.append(np.arange(start, start + length))
        if not tokens:
            return None
        positions = torch.from_numpy(np.concatenate(positions))
        blocked = positions.unsqueeze(-1) < torch.arange(self.n)
        return torch.from_numpy(np.concatenate(tokens)), blocked

    def _gather(self, table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (ids.numel(), table.shape[1]), dtype=table.dtype, pin_memory=True
        )
        step = -(-ids.numel() // _GATHER_THREADS)
        chunks = [
            self.pool.submit(
                torch.index_select, table, 0, ids[i : i + step], out=out[i : i + step]
            )
            for i in range(0, ids.numel(), step)
        ]
        for chunk in chunks:
            chunk.result()
        return out
