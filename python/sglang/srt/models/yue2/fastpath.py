# SPDX-License-Identifier: Apache-2.0
"""Fast sampler for YuE2 AR decoding, backed by the DeepSelect TopK kernel.

`distribution()` in `.sampling` masks the full vocabulary, sorts it, and 
multinomial-samples over 184,704 categories per step. 


1. scores live in one padded fp32 row ``[1, V_pad]`` where ``V_pad`` rounds 
   the vocabulary up to a multiple of 256 floats

2. ``deep_select.topk(scores, k, sorted=True)`` returns the descending top-k
   values and global indices in one kernel.

3. top-p filtering and ``torch.multinomial`` run on the ``[1, k]`` support.

"""
from __future__ import annotations

import os

import torch

from .protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END

_IN_TREE = None
_PIP = None


# NOTE (yiakwy) : deepseek-select is slightly fast than extended flash-float-jit-kernel in this workload
def _load_topk_backend():
    """Prefer the in-tree JIT DeepSelect wrapper, fall back to the pip package."""
    global _IN_TREE, _PIP
    if _IN_TREE is None:
        try:
            from sglang.kernels.ops import deep_select as module

            _IN_TREE = module
        except ImportError:
            _IN_TREE = False
    if not _IN_TREE and _PIP is None:
        try:
            import deep_select as module

            _PIP = module
        except ImportError:
            _PIP = False
    return _IN_TREE or _PIP


def fast_sampler_available() -> bool:
    if os.environ.get("SGLANG_YUE2_FAST_SAMPLER", "1") != "1":
        return False
    backend = _load_topk_backend()
    if not backend:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


STRIDE_FLOATS = 256  # DeepSelect row alignment: 1024B = 256 fp32


class FastSampler:
    """Reused per-model state for the accelerated decode loop."""

    def __init__(self, vocab_size: int, device: torch.device):
        topk_backend = _load_topk_backend()
        if not topk_backend:
            raise ImportError(
                "DeepSelect/flash-float-jit-kernel are required for the fast sampler."
            )
        
        self.backend = topk_backend

        self.vocab = int(vocab_size)
        self.device = device
        self.pad = (-self.vocab) % STRIDE_FLOATS
        self.total = self.vocab + self.pad
        self.scores = torch.empty(1, self.total, dtype=torch.float32, device=device)
        self.scores[0, self.vocab:] = float("-inf")
        self._masks: dict[str, torch.Tensor] = {}
        self._ends: dict[str, int] = {}

    def _phase_mask(self, phase: str) -> torch.Tensor:
        mask = self._masks.get(phase)
        if mask is None:
            row = torch.full((self.total,), float("-inf"), device=self.device,
                             dtype=torch.float32)
            if phase == "abc":
                row[:EOD] = 0.0
                row[ABC_END] = 0.0
                end = ABC_END
            else:
                # MUSIC_END sits directly below the contiguous codec range.
                row[MUSIC_END:CODEC_OFFSET + CODEC_SIZE] = 0.0
                end = MUSIC_END
            row[self.vocab:] = float("-inf")
            mask = row.unsqueeze(0)
            self._masks[phase] = mask
            self._ends[phase] = end
        return mask

    def allowed_end(self, phase: str) -> int:
        self._phase_mask(phase)
        return self._ends[phase]

    @torch.inference_mode()
    def sample_token(
        self,
        logits: torch.Tensor,
        phase: str,
        temperature: float,
        top_p: float,
        top_k: int,
        history: torch.Tensor,
        history_len: int,
        repetition_penalty: float,
        penalty_window: int,
        min_tokens: int,
        step: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Return the next token as a GPU ``[1, 1]`` int64 tensor."""
        mask = self._phase_mask(phase)
        end = self._ends[phase]
        torch.add(logits, mask[:, : self.vocab], out=self.scores[:, : self.vocab])
        if step < min_tokens:
            self.scores[0, end] = float("-inf")
        if repetition_penalty != 1.0 and history_len > 0:
            _window_penalty_(self.scores[:, : self.vocab], history[:history_len],
                             repetition_penalty, penalty_window)
        if temperature == 0:
            return self.scores.argmax(dim=-1, keepdim=True).to(torch.int64)
        if temperature != 1:
            self.scores.div_(temperature)
        k = min(int(top_k), self.vocab)

        values, idx = self.backend.topk(self.scores, k, sorted=True,
                                        indices_type=torch.int64, return_value=True)
        if top_p < 1:
            probabilities = values.softmax(-1)
            removed = probabilities.cumsum(-1) - probabilities > top_p
            removed[..., :1] = False
            values = values.masked_fill(removed, float("-inf"))
        probabilities = values.softmax(-1)
        position = torch.multinomial(probabilities, 1, generator=generator)
        return idx.gather(1, position)


MAX_GRAPH_TOPK = 100  # capture budget: max top_k across phases (protocol caps at 100)


class GraphSampler:
    """Decode-graph-resident sampler: the whole ``distribution()`` chain runs
    inside the AR step CUDA graph.
    """

    def __init__(self, vocab_size: int, device: torch.device, generator: torch.Generator):
        topk_backend = _load_topk_backend()

        if not topk_backend:
            raise ImportError("GraphSampler requires the DeepSelect backend")
        
        self.backend = topk_backend

        self.vocab = int(vocab_size)

        self.device = device

        self.window_buf_cap = 100

        self.pad = (-self.vocab) % STRIDE_FLOATS

        self.total = self.vocab + self.pad

        self.scores = torch.empty(1, self.total, dtype=torch.float32, device=device)

        self.scores[0, self.vocab:] = float("-inf")

        self.mask = torch.empty(1, self.total, dtype=torch.float32, device=device)

        self.temp_buf = torch.ones(1, dtype=torch.float32, device=device)

        self.top_p_buf = torch.ones(1, dtype=torch.float32, device=device)

        self.topk_buf = torch.full((1,), MAX_GRAPH_TOPK, dtype=torch.int32, device=device)

        self.penalty_buf = torch.ones(1, dtype=torch.float32, device=device)

        self.window_len_buf = torch.full((1,), self.window_buf_cap, dtype=torch.int32,
                                         device=device)
        
        self.window_buf = torch.full((self.window_buf_cap,), self.total - 1,
                                     dtype=torch.int64, device=device)
        
        self.slot_seq = torch.full((self.window_buf_cap,), -(1 << 20),
                                   dtype=torch.int32, device=device)
        
        self.step_buf = torch.zeros(1, dtype=torch.int32, device=device)

        self.end_buf = torch.zeros(1, 1, dtype=torch.int64, device=device)

        self.cfg_buf = torch.ones(1, dtype=torch.float32, device=device)

        self.token_buf = torch.zeros(1, 1, dtype=torch.int64, device=device)

        self.done_buf = torch.zeros(1, 1, dtype=torch.bool, device=device)

        self._ages = torch.empty(self.window_buf_cap, dtype=torch.int32, device=device)

        self._window_weights = torch.ones(self.window_buf_cap, dtype=torch.float32, device=device)

        self._topk_ranks = torch.arange(MAX_GRAPH_TOPK, device=device, dtype=torch.int32)

        self._freq = torch.zeros(self.total, dtype=torch.float32, device=device)

        self._ones_w = torch.ones(self.window_buf_cap, dtype=torch.float32, device=device)

        self.generator = generator

        self._k = MAX_GRAPH_TOPK

        # Output buffers allocated during capture.
        self.values_buf = None
        self.idx_buf = None

    def configure_phase(self, phase: str, temperature: float, top_p: float, top_k: int,
                        repetition_penalty: float, penalty_window: int, end: int,
                        min_tokens: int) -> None:
        """Refill the per-phase input buffers (called outside the graph)."""
        self.temp_buf.fill_(max(float(temperature), 1e-5))
        self.top_p_buf.fill_(float(top_p))
        self.topk_buf.fill_(int(min(max(top_k, 1), MAX_GRAPH_TOPK)))
        self.penalty_buf.fill_(float(repetition_penalty))
        self.window_len_buf.fill_(int(penalty_window))
        self.end_buf.fill_(int(end))
        self.mask.zero_()
        self.mask[0, self.vocab:] = float("-inf")

        if phase == "abc":
            self.mask[0, :EOD] = 0.0
            self.mask[0, ABC_END] = 0.0
        elif phase == "semantic":
            self.mask[0, MUSIC_END:CODEC_OFFSET + CODEC_SIZE] = 0.0
        else:  # forced single-token mask for transition steps
            self.mask.fill_(float("-inf"))
            self.mask[0, int(end)] = 0.0

        self.mask[0, self.vocab:] = float("-inf")
        self.window_buf.fill_(self.total - 1)
        self.slot_seq.fill_(-(1 << 20))
        self.step_buf.zero_()
        self.done_buf.zero_()

    def force_token(self, token: int) -> None:
        """Forced transition step: mask everything except ``token``."""
        self.mask.fill_(float("-inf"))
        self.mask[0, self.vocab:] = float("-inf")
        self.mask[0, int(token)] = 0.0
        self.end_buf.fill_(-1)

    def sample_in_graph(self, logits: torch.Tensor, cfg_pair=None):
        """Graph body: mask -> penalty -> top-k -> top-p -> multinomial.

        When ``cfg_pair`` is given (cond, uncond) the historical BF16 CFG
        combine runs first. Returns (token_buf, done_buf) — both live in
        graph-owned storage and feed the next replay.
        """
        if cfg_pair is not None:
            cond, uncond = cfg_pair
            torch.add(uncond, torch.mul(cond - uncond, self.cfg_buf), out=logits)
        torch.add(logits, self.mask[:, : self.vocab], out=self.scores[:, : self.vocab])
        self._freq.zero_()
        torch.sub(self.step_buf, self.slot_seq, out=self._ages)
        self._window_weights.copy_(self._ages <= self.window_len_buf)
        self._freq.scatter_add_(0, self.window_buf, self._window_weights)
        alpha = self.penalty_buf ** self._freq
        scores = torch.where(self.scores < 0, self.scores * alpha, self.scores / alpha)
        scores.div_(self.temp_buf)

        values, idx = self.backend.topk(scores, self._k, sorted=True,
                                        indices_type=torch.int64, return_value=True)
        
        self.values_buf, self.idx_buf = values, idx
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > self.top_p_buf
        removed = removed | (self._topk_ranks >= self.topk_buf)
        removed[..., :1] = False
        p = values.masked_fill(removed, float("-inf")).softmax(-1)
        position = torch.multinomial(p, 1, generator=self.generator)
        torch.gather(idx, 1, position, out=self.token_buf)
        torch.eq(self.token_buf, self.end_buf, out=self.done_buf)
        return self.token_buf, self.done_buf

    def advance_in_graph(self):
        """Ring-buffer bookkeeping executed inside the graph after sampling."""
        slot = self.step_buf % self.window_buf_cap
        self.window_buf[slot] = self.token_buf[0, 0]
        self.slot_seq[slot] = self.step_buf[0]
        self.step_buf += 1

    def advance_eager(self):
        """First-sample ring update (runs outside the graph, same semantics)."""
        self.window_buf[0] = self.token_buf[0, 0]
        self.slot_seq[0] = 0
        self.step_buf.fill_(1)

    def reset_state(self):
        """Clear the ring/step/decision state (capture warmup pollution)."""
        self.window_buf.fill_(self.total - 1)
        self.slot_seq.fill_(-(1 << 20))
        self.step_buf.zero_()
        self.token_buf.zero_()
        self.done_buf.zero_()


def _window_penalty_(scores: torch.Tensor, recent: torch.Tensor, penalty: float,
                     window: int) -> None:
    """In-place version of ``sampling.window_penalty`` on the scores row."""
    if window < recent.numel():
        recent = recent[-window:]

    freq = torch.zeros_like(scores)
    freq.scatter_add_(-1, recent.reshape(1, -1),
                      torch.ones(1, recent.numel(), dtype=scores.dtype, device=scores.device))
    
    alpha = penalty ** freq
    scores.copy_(torch.where(scores < 0, scores * alpha, scores / alpha))
