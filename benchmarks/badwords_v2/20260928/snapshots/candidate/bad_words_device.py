"""Request-owned CUDA bad-word suffixes; kernels run outside model CUDA graphs.

Only committed generated tokens enter the ring. DFlash candidate column zero
is already committed, so row j matches against H + candidates[1:j+1].
Pointers are retained by DeviceBatch and recorded on each consuming stream.
"""

from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class BadWordsRequestSpec:
    """Immutable tokenized constraints; constructed once per request generation."""

    words: tuple
    flat: tuple
    offsets: tuple
    prefix: int

    @classmethod
    def build(cls, words):
        words = tuple(sorted(set(tuple(w) for w in words)))
        if any(not w or any(type(t) is not int or t < 0 for t in w) for w in words):
            raise ValueError("bad_words requires nonempty sequences of token IDs")
        prefix = max((len(w) - 1 for w in words), default=0)
        if (
            len(words) > MAX_WORDS
            or sum(map(len, words)) > MAX_WORD_TOKENS
            or prefix > MAX_PREFIX
        ):
            raise ValueError(
                "CUDA bad_words exceeds limits: 1024 sequences, "
                "16384 total tokens, 257 tokens per sequence"
            )
        flat, offsets = [], [0]
        for word in words:
            flat.extend(word)
            offsets.append(len(flat))
        return cls(words, tuple(flat), tuple(offsets), prefix)


class DeviceState:
    """Request-owned state, independent of batch row and KV allocation slot.

    Retraction replaces this object. In-flight contexts keep the old generation
    alive; a late commit can therefore never modify the replacement generation.
    """

    def __init__(self, words, history, device, epoch=0):
        self.spec = (
            words
            if isinstance(words, BadWordsRequestSpec)
            else BadWordsRequestSpec.build(words)
        )
        self.count = len(self.spec.words)
        self.prefix = self.spec.prefix
        self.capacity = max(1, self.prefix)
        flat, offsets = self.spec.flat, self.spec.offsets
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
class _StreamFence:
    # Runtime ordering evolves; the request-to-row mapping remains immutable.
    event: torch.cuda.Event


@dataclass(frozen=True)
class DeviceBatch:
    states: tuple
    batch_size: int
    table: torch.Tensor
    max_words: int
    max_prefix: int
    ready: _StreamFence

    @classmethod
    def build(cls, params, device, commit_mask=None):
        entries, states = [], []
        for row, param in enumerate(params or []):
            if not param or not param.get("bad_words_token_ids"):
                continue
            req = param["__req__"]
            state = BadWordsStateManager.acquire(
                req, param["bad_words_token_ids"], device
            )
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
            tuple(states),
            len(params),
            table,
            max(s.count for s in states),
            max(s.prefix for s in states),
            _StreamFence(ready),
        )

    def use(self):
        stream = torch.cuda.current_stream(self.table.device)
        self.table.record_stream(stream)
        events = {self.ready.event, *(state.ready for state in self.states)}
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
        if width < 1 or logits.ndim != 2 or logits.shape[0] != self.batch_size * width:
            raise ValueError("bad_words logits rows do not match the sampling context")
        if width > 1 and (
            drafts is None
            or not drafts.is_contiguous()
            or tuple(drafts.shape) != (self.batch_size, width)
            or drafts.device != logits.device
            or drafts.dtype not in (torch.int32, torch.int64)
        ):
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
        # The completion event subsumes the table upload dependency. Keeping
        # both would insert an unnecessary stream wait on every commit.
        self.ready.event = event
        for state in self.states:
            state.ready = event

    def commit(self, tokens, lengths=None):
        if tokens.ndim not in (1, 2) or tokens.shape[0] != self.batch_size:
            raise ValueError("accepted tokens do not match the sampling context")
        if tokens.device != self.table.device or tokens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "accepted tokens must be integer IDs on the context device"
            )
        if lengths is not None and (
            lengths.shape != (self.batch_size,)
            or lengths.device != tokens.device
            or lengths.dtype not in (torch.int32, torch.int64)
            or not lengths.is_contiguous()
        ):
            raise ValueError("accepted lengths do not match the sampling context")
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


class BadWordsStateManager:
    """Admission/resume boundary for persistent request state.

    A request owns its handle, so cancellation/finish needs no global slot free
    list. Contexts retain state tensors until their last asynchronous GPU use.
    No pointer table is shared or overwritten between in-flight invocations.
    """

    @staticmethod
    def acquire(req, words, device):
        epoch = getattr(req, "retraction_count", 0)
        state = getattr(req, "bad_words_state", None)
        if state is None or state.epoch != epoch:
            state = DeviceState(words, req.output_ids, device, epoch)
            req.bad_words_state = state
        return state


@dataclass
class BadWordsSamplingContext:
    """One mask/accept transaction with an immutable request-to-row snapshot."""

    view: DeviceBatch
    verify_width: int
    committed: bool = False

    def commit_accepted(self, tokens, lengths=None):
        if self.committed:
            raise RuntimeError("bad_words accepted tokens were already committed")
        # Mark only after successful validation/launch. This guards even a caller
        # holding a second reference to this context, unlike clearing one field.
        self.view.commit(tokens, lengths)
        self.committed = True


def apply_device_bad_words(logits, sampling_info, width=1, drafts=None):
    if getattr(sampling_info, "bad_words_context", None) is not None:
        raise RuntimeError("previous bad_words sampling context was not consumed")
    view = DeviceBatch.build(
        sampling_info.custom_params, logits.device, sampling_info.bad_words_commit_mask
    )
    if view is not None:
        view.apply(logits, width, drafts)
        sampling_info.bad_words_context = BadWordsSamplingContext(view, width)


def take_bad_words_context(sampling_info) -> Optional[BadWordsSamplingContext]:
    """Move invocation state out of batch metadata before acceptance work."""
    if sampling_info is None:
        return None
    context = getattr(sampling_info, "bad_words_context", None)
    sampling_info.bad_words_context = None
    return context


def commit_accepted_bad_words(context, tokens, lengths=None):
    """Shared ordinary-decode / speculative-accept postprocessing boundary.

    lengths includes accepted drafts and the replacement/bonus token. Rejected
    drafts never enter history. Intermediate prefill chunks are disabled in the
    view's immutable commit mask. Counts are trusted device acceptance output.
    """
    if context is not None:
        context.commit_accepted(tokens, lengths)
