"""Request-owned CUDA bad-word suffixes; kernels run outside model CUDA graphs.

Only committed generated tokens enter the ring. DFlash candidate column zero
is already committed, so row j matches against H + candidates[1:j+1].
Pointers are retained by DeviceBatch and recorded on each consuming stream.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.srt.sampling.custom_logit_processor import _BAD_WORDS_BACKEND

# Explicit limits bound allocations and compilation; never silently truncate.
MAX_WORDS = 1024
MAX_WORD_TOKENS = 16384
MAX_PREFIX = 256


def device_bad_words_enabled():
    return _BAD_WORDS_BACKEND == "cuda"


@triton.jit
def _match(
    Table, Logits, Draft, STRIDE: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    request = tl.program_id(0)
    pos = tl.program_id(1)
    word = tl.program_id(2)
    base = Table + request * 7
    state = tl.load(base).to(tl.pointer_type(tl.int32))
    tokens = tl.load(base + 1).to(tl.pointer_type(tl.int32))
    offsets = tl.load(base + 2).to(tl.pointer_type(tl.int32))
    row = tl.load(base + 3)
    count = tl.load(base + 4)
    capacity = tl.load(base + 5).to(tl.int32)
    if word < count:
        start = tl.load(offsets + word)
        end = tl.load(offsets + word + 1)
        length = end - start - 1
        valid = tl.load(state)
        head = tl.load(state + 1)
        x = tl.arange(0, BLOCK)
        # Logical index relative to the beginning of the retained history.
        index = valid + pos - length + x
        from_history = index < valid
        ring_index = (head - valid + index + capacity) % capacity
        history = tl.load(
            state + 2 + ring_index, (x < length) & (index >= 0) & from_history, other=0
        )
        candidate = tl.full((BLOCK,), 0, tl.int32)
        if WIDTH > 1:
            candidate = tl.load(
                Draft + row * WIDTH + index - valid + 1,
                (x < length) & (index >= valid),
                other=0,
            )
        actual = tl.where(from_history, history, candidate)
        expected = tl.load(tokens + start + x, x < length, other=0)
        matches = (valid + pos >= length) & (
            tl.sum(((x < length) & (actual != expected)).to(tl.int32), 0) == 0
        )
        if matches:
            terminal = tl.load(tokens + end - 1)
            # Multiple matching words may have the same terminal token.
            tl.atomic_xchg(
                Logits + (row * WIDTH + pos) * STRIDE + terminal,
                -float("inf"),
                sem="relaxed",
            )


@triton.jit
def _commit(
    Table,
    Tokens,
    Lengths,
    WIDTH: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    base = Table + request * 7
    state = tl.load(base).to(tl.pointer_type(tl.int32))
    row = tl.load(base + 3)
    capacity = tl.load(base + 5).to(tl.int32)
    enabled = tl.load(base + 6)
    if enabled:
        count = WIDTH
        if HAS_LENGTHS:
            count = tl.load(Lengths + row)
        head = tl.load(state + 1)
        valid = tl.load(state)
        # If count exceeds ring capacity, only the last capacity tokens write.
        x = tl.arange(0, BLOCK)
        start = tl.maximum(0, count - capacity)
        index = start + x
        value = tl.load(Tokens + row * WIDTH + index, index < count, other=0)
        tl.store(state + 2 + (head + index) % capacity, value, index < count)
        tl.store(state, tl.minimum(capacity, valid + count))
        tl.store(state + 1, (head + count) % capacity)


def _upload(values, dtype, device):
    # Pinned staging keeps metadata uploads from synchronizing the forward stream.
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


class DeviceState:
    def __init__(self, words, history, device, epoch=0):
        words = sorted(set(tuple(w) for w in words))
        self.count = len(words)
        self.prefix = max((len(w) - 1 for w in words), default=0)
        if (
            self.count > MAX_WORDS
            or sum(map(len, words)) > MAX_WORD_TOKENS
            or self.prefix > MAX_PREFIX
        ):
            raise ValueError(
                "CUDA bad_words exceeds limits: 1024 sequences, "
                "16384 total tokens, 257 tokens per sequence"
            )
        self.capacity = max(1, self.prefix)
        flat, offsets = [], [0]
        for word in words:
            flat.extend(word)
            offsets.append(len(flat))
        suffix = list(history[-self.capacity :])
        self.state = _upload(
            [len(suffix), len(suffix) % self.capacity]
            + suffix
            + [0] * (self.capacity - len(suffix)),
            dtype=torch.int32,
            device=device,
        )
        self.tokens = _upload(flat, torch.int32, device)
        self.offsets = _upload(offsets, torch.int32, device)
        self.epoch = epoch
        self.ready = torch.cuda.Event()
        self.ready.record()


@dataclass
class DeviceBatch:
    states: list
    table: torch.Tensor
    max_words: int
    max_prefix: int
    ready: torch.cuda.Event

    @classmethod
    def build(cls, params, device, commit_mask=None):
        entries, states = [], []
        for row, param in enumerate(params or []):
            if not param or not param.get("bad_words_token_ids"):
                continue
            req = param["__req__"]
            epoch = getattr(req, "retraction_count", 0)
            state = getattr(req, "_bad_words_device", None)
            if state is None or state.epoch != epoch:
                state = DeviceState(
                    param["bad_words_token_ids"], req.output_ids, device, epoch
                )
                req._bad_words_device = state
            states.append(state)
            entries.append(
                [
                    state.state.data_ptr(),
                    state.tokens.data_ptr(),
                    state.offsets.data_ptr(),
                    row,
                    state.count,
                    state.capacity,
                    int(commit_mask is None or commit_mask[row]),
                ]
            )
        if not entries:
            return None
        table = _upload(entries, torch.int64, device)
        ready = torch.cuda.Event()
        ready.record()
        return cls(
            states,
            table,
            max(s.count for s in states),
            max(s.prefix for s in states),
            ready,
        )

    def use(self):
        stream = torch.cuda.current_stream(self.table.device)
        self.table.record_stream(stream)
        events = {self.ready, *(state.ready for state in self.states)}
        for event in events:
            stream.wait_event(event)
        for state in self.states:
            for tensor in (state.state, state.tokens, state.offsets):
                tensor.record_stream(stream)

    def apply(self, logits, width=1, drafts=None):
        if logits.dtype != torch.float32 or logits.stride(1) != 1:
            raise ValueError(
                "CUDA bad_words requires float32 contiguous vocabulary logits"
            )
        if width > 1 and (drafts is None or not drafts.is_contiguous()):
            raise ValueError("CUDA bad_words requires contiguous DFlash candidates")
        self.use()
        _match[(len(self.states), width, self.max_words)](
            self.table,
            logits,
            drafts if drafts is not None else self.table,
            logits.stride(0),
            width,
            triton.next_power_of_2(max(1, self.max_prefix)),
        )
        self._record_ready()

    def _record_ready(self):
        # Order both reads and writes if a request moves to another CUDA stream.
        event = torch.cuda.Event()
        event.record()
        self.ready = event
        for state in self.states:
            state.ready = event

    def commit(self, tokens, lengths=None):
        if self.max_prefix == 0:
            return  # Single-token bans never depend on generated history.
        self.use()
        tokens = tokens.contiguous()
        width = 1 if tokens.ndim == 1 else tokens.shape[1]
        _commit[(len(self.states),)](
            self.table,
            tokens,
            lengths if lengths is not None else self.table,
            width,
            lengths is not None,
            triton.next_power_of_2(min(width, max(1, self.max_prefix))),
        )
        self._record_ready()


def apply_device_bad_words(logits, sampling_info, width=1, drafts=None):
    # Mapping is invocation-local; retained states belong to requests, not rows.
    batch = DeviceBatch.build(
        sampling_info.custom_params, logits.device, sampling_info.bad_words_commit_mask
    )
    sampling_info._bad_words_device_batch = batch
    if batch is not None:
        batch.apply(logits, width, drafts)


def commit_device_bad_words(sampling_info, tokens, lengths=None):
    if device_bad_words_enabled():
        batch = getattr(sampling_info, "_bad_words_device_batch", None)
        if batch is not None:
            batch.commit(tokens, lengths)
            sampling_info._bad_words_device_batch = None
