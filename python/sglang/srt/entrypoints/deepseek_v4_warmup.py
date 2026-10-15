"""DeepSeek-V4 serving-shape warm-up (``--warmups deepseek_v4_serving_shapes``).

Graph capture and the server's own warm-up run only short decode-shaped batches, so
prefill sizes, prefix hits, chunked prompts, multi-request extends, eager decode above
the decode graphs, mixed batches and prompts past the prefill graph's context bucket
compiled their Triton and JIT kernel variants inside the first requests that reached
them (about 8 s for the first long prompt on a cold cache). These internal requests take
the real serving path at each shape class before the server listens, alternating greedy
and the server's default sampling; the cache is flushed afterwards so no prefix or KV
state survives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

_RID_PREFIX = "deepseek-v4-warmup-"

# (concurrent, prompt lengths, max_new_tokens). "k+n" resends source prompt k plus n
# new tokens, a prefix hit with an n-token extend ("k+0" extends one token). Triton
# specializes integer arguments as 1, divisible by 16, or other, and the mHC prefill
# kernel switches at 4,096 rows; 20,011 tokens take several chunks with a prefix.
_PLAN = (
    (False, (1, 13, 16, 211, 4096, 4817, 20011), 8),
    (False, ("5+0", "5+7", "5+160", "6+4321", "6+0"), 8),
    (True, (300, 5000), 16),
    (True, (64, 2000, 9000), 16),
    (True, (17, 32, 1100, 4100), 16),
    (True, ("2+5", "3+40", "4+4099"), 16),
)
# A batched request reaches the scheduler as one message: one n-request extend, then
# decode at batch n. Kernels keyed on next_power_of_2(bs) or bs * verify width see
# every size, with a total token count divisible by 16 and not.
_GROUP_SIZES = range(1, 17)
# One eager verify step above the decode graphs, plus the second step at graph sizes.
_GROUP_NEW_TOKENS = 4
# Decodes that a prompt prefill joins (mixed batches): 1, then 2 decode rows. 256 tokens
# outlast both 8K chunks of the prompt by more than 10x.
_MIXED = ((1, 12000), (2, 9500))
_MIXED_DECODE_TOKENS = 256
# Past the prefill graph's context bucket prefill runs eager, where the mHC kernels keyed
# on the row count compile on first use: an aligned 8K chunk, an unaligned tail, and
# three extends on that prompt.
_ABOVE_GRAPH_TAIL = (8192, 1001)
_ABOVE_GRAPH_EXTENDS = (211, 2064, 5000)

Length = Union[int, str]


async def warm_serving_shapes(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
) -> None:
    from sglang.srt.runtime_context import get_schedule

    if disaggregation_mode != "null":
        logger.warning("deepseek_v4_serving_shapes is for aggregated serving; skipped.")
        return
    await run_serving_shapes(
        tokenizer_manager,
        mixed_chunk=get_schedule().enable_mixed_chunk,
        prefill_graph_context=_prefill_graph_context(),
    )


async def run_serving_shapes(
    tokenizer_manager: TokenizerManager,
    *,
    mixed_chunk: bool,
    prefill_graph_context: Optional[int],
) -> None:
    start = time.perf_counter()
    warm = _Warmup(tokenizer_manager)
    for concurrent, lengths, max_new_tokens in _PLAN:
        prompts = warm.prompts(lengths)
        if concurrent:
            await asyncio.gather(*(warm.run(p, max_new_tokens) for p in prompts))
        else:
            for prompt in prompts:
                await warm.run(prompt, max_new_tokens)
    for size in _GROUP_SIZES:
        for divisible in (True, False):
            await warm.run_group(_group_lengths(size, divisible), greedy=divisible)
    if mixed_chunk:
        for num_decodes, prefill_len in _MIXED:
            await warm.run_mixed(num_decodes, prefill_len)
    if prefill_graph_context is not None:
        await warm.run_above_graph(prefill_graph_context)
    flushed = await _flush(tokenizer_manager)
    logger.info(
        "DeepSeek-V4 serving-shape warm-up: %d requests in %.1f s; cache flushed: %s.",
        warm.count,
        time.perf_counter() - start,
        flushed,
    )
    if not flushed:
        raise RuntimeError("deepseek_v4_serving_shapes could not flush the cache.")


def _prefill_graph_context() -> Optional[int]:
    """Longest context the breakable prefill graph replays, or None when it is off."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec

    prefill = get_exec().graph.cuda_graph_config.prefill
    if prefill.backend != Backend.BREAKABLE:
        return None
    bounds = [b for b in (prefill.max_context_size, prefill.max_seq_len) if b]
    return min(bounds) if bounds else None


def _group_lengths(size: int, divisible_by_16: bool) -> List[int]:
    lengths = [64 + 37 * i for i in range(size)]
    total = sum(lengths)
    if divisible_by_16:
        lengths[-1] += -total % 16
    elif total % 16 == 0:
        lengths[-1] += 1
    return lengths


async def _flush(tokenizer_manager: TokenizerManager) -> bool:
    # The flush is refused while the last requests are still being released.
    for _ in range(50):
        if (await tokenizer_manager.flush_cache()).success:
            return True
        await asyncio.sleep(0.1)
    return False


async def _drain(gen) -> None:
    async for _ in gen:
        pass


class _Warmup:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.rng = np.random.default_rng(0)
        self.sources: List[List[int]] = []
        self.count = 0

    def prompts(self, lengths: Sequence[Length]) -> List[List[int]]:
        out = []
        for length in lengths:
            if isinstance(length, str):
                base, extra = (int(x) for x in length.split("+"))
                out.append(self.sources[base] + self._tokens(extra))
            else:
                out.append(self._tokens(length))
                self.sources.append(out[-1])
        return out

    def _tokens(self, n: int) -> List[int]:
        return self.rng.integers(1000, 60000, size=n).tolist()

    def _request(self, prompt, max_new_tokens: int, *, stream: bool = False):
        self.count += 1
        params = {"max_new_tokens": max_new_tokens, "ignore_eos": True}
        if self.count % 2 == 0:
            params["temperature"] = 0.0
        return GenerateReqInput(
            input_ids=prompt,
            sampling_params=params,
            rid=f"{_RID_PREFIX}{self.count}",
            stream=stream,
        )

    async def run(self, prompt: List[int], max_new_tokens: int) -> None:
        req = self._request(prompt, max_new_tokens)
        await _drain(self.tokenizer_manager.generate_request(req, None))

    async def run_group(self, lengths: List[int], *, greedy: bool) -> None:
        prompts = self.prompts(lengths)
        self.count += len(prompts)
        params = {"max_new_tokens": _GROUP_NEW_TOKENS, "ignore_eos": True}
        if greedy:
            params["temperature"] = 0.0
        req = GenerateReqInput(
            input_ids=prompts,
            sampling_params=params,
            rid=[f"{_RID_PREFIX}{self.count}-{i}" for i in range(len(prompts))],
        )
        await _drain(self.tokenizer_manager.generate_request(req, None))

    async def run_mixed(self, num_decodes: int, prefill_len: int) -> None:
        streams = []
        for prompt in self.prompts((500,) * num_decodes):
            req = self._request(prompt, _MIXED_DECODE_TOKENS, stream=True)
            gen = self.tokenizer_manager.generate_request(req, None)
            await gen.__anext__()  # decoding from here on
            streams.append(asyncio.ensure_future(_drain(gen)))
        await self.run(self.prompts((prefill_len,))[0], 8)
        await asyncio.gather(*streams)

    async def run_above_graph(self, graph_context: int) -> None:
        length = graph_context + sum(_ABOVE_GRAPH_TAIL)
        if length + max(_ABOVE_GRAPH_EXTENDS) > self.tokenizer_manager.context_len:
            return
        base = len(self.sources)
        await self.run(self.prompts((length,))[0], 2)
        for prompt in self.prompts([f"{base}+{n}" for n in _ABOVE_GRAPH_EXTENDS]):
            await self.run(prompt, 2)
