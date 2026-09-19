"""Build a ``bench_serving`` agentic-trace file from the public AgentX corpus.

InferenceX's AgentX benchmark replays SemiAnalysis' Claude Code proxy traces
(the ``semianalysisai/cc-traces-weka-*`` datasets on the Hub) through aiperf.
Those traces carry no text at all: every recorded model call is a
``{"t", "in", "out", "hash_ids"}`` record, where ``in``/``out`` are token counts
and ``hash_ids`` are the ids of the 64-token KV blocks the prompt covered.
aiperf synthesises prompts from that skeleton at replay time.

sglang already replays multi-turn agentic conversations natively --
``bench_serving --dataset-name agentic-trace`` walks a conversation round by
round and feeds the server's own reply back into the next round's history --
but it wants a text trace. This module bridges the two: it turns the recorded
per-request token counts into synthetic conversations whose *replayed* prompt
lengths track the recorded ones turn by turn, so the sequence of prefill sizes,
cache hits and decode lengths the engine sees follows the real workload.

What is preserved
-----------------
* Per-turn prompt growth of the main agent stream. Turn ``k`` contributes only
  the tokens the trace added between request ``k-1`` and request ``k``, minus
  the reply the server is about to generate, so the accumulated history at turn
  ``k`` equals the recorded ``in`` of request ``k``.
* Per-turn decode lengths. Each turn carries its recorded ``out`` as
  ``output_tokens``, which ``AgenticTraceDataset`` replays round by round. This
  is also what keeps prompt growth exact: replay appends the same number of
  reply tokens the trace did.
* Within-conversation prefix reuse, which is where essentially all of an
  agentic workload's cache hits come from (in the corpus a typical turn reuses
  1310 of 1325 blocks).
* One unique prefix per conversation. AgentX's scenario pins
  ``--cache-bust first_turn_prefix``, so cross-trajectory reuse is deliberately
  broken there too, and synthesising independent text per conversation
  reproduces that.

Replay can only append, so a replayed prompt is never *shorter* than the
recorded one, and :data:`DEFAULT_CONTEXT_RESET_RATIO` bounds how much longer it
can get before the trajectory is split. Against the published corpus that holds
the per-turn prompt length within 2% at p95 and 10% at worst, on top of about
0.3% of tokenizer round-trip slack in the synthetic text.

What is not
-----------
* Recorded inter-request think time. sglang's replay is closed-loop: a lane
  issues the next turn as soon as the previous one returns, so the run is
  concurrency-driven rather than timestamp-driven.
* Sub-agent fan-out. Sub-agent groups run concurrently with their parent in the
  real trace; ``include_subagents`` emits them as independent conversations,
  which keeps their token volume but not their overlap with the parent.
* Prompt *content*. Only lengths and reuse structure are reproduced; the text
  itself is deterministic filler, so this measures serving behaviour, not
  anything about what the model generates.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import requests

# v7 corpus with the 256k per-request cap, the default AgentX picks for model
# families whose native context is below 1M tokens (Qwen3.5 among them).
DEFAULT_WEKA_TRACE_REPO = "semianalysisai/cc-traces-weka-062126-256k"
DEFAULT_WEKA_TRACE_FILENAME = "traces.jsonl"

# Main-agent requests; sub-agent groups are nested under "subagent" entries.
MAIN_AGENT_REQUEST_TYPE = "s"
SUBAGENT_GROUP_TYPE = "subagent"

# A turn always has to say something, even where the recorded prompt grew by
# less than this (or shrank slightly, which replay cannot follow).
DEFAULT_MIN_DELTA_TOKENS = 32

# A recorded prompt this much smaller than the history replay has accumulated is
# a context reset (agent compaction, or a new session on the same trajectory),
# not noise, and is replayed as a fresh conversation. Clamping through one
# instead would leave replay stuck tens of thousands of tokens above the trace
# for the rest of the trajectory.
DEFAULT_CONTEXT_RESET_RATIO = 0.10

_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_CHUNK_SIZE = 1 << 20


@dataclass
class RecordedTurn:
    """One model call as the corpus recorded it."""

    prompt_tokens: int
    output_tokens: int


@dataclass
class PlannedTurn:
    """One replay round: what to send, how much to generate, and the result.

    ``prompt_tokens`` is the history the server will have accumulated once
    ``new_tokens`` are appended, and is what the plan makes track the corpus.
    """

    new_tokens: int
    output_tokens: int
    prompt_tokens: int


@dataclass
class TraceStats:
    """Aggregate shape of a converted trace, for the benchmark report."""

    num_conversations: int
    num_turns: int
    total_prompt_tokens: int
    max_prompt_tokens: int
    total_output_tokens: int
    mean_output_len: float

    def as_markdown_rows(self) -> str:
        return (
            f"| conversations | {self.num_conversations} |\n"
            f"| turns | {self.num_turns} |\n"
            f"| mean turns / conversation | "
            f"{self.num_turns / max(self.num_conversations, 1):.1f} |\n"
            f"| mean output len / turn | {self.mean_output_len:.0f} |\n"
            f"| total output tokens | {self.total_output_tokens} |\n"
            f"| max context (tokens) | {self.max_prompt_tokens} |\n"
        )


def weka_trace_url(
    repo: str = DEFAULT_WEKA_TRACE_REPO,
    filename: str = DEFAULT_WEKA_TRACE_FILENAME,
) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{repo}/resolve/main/{filename}"


def iter_weka_records(source: str, limit: Optional[int] = None) -> Iterator[dict]:
    """Yield WekaTrace records from a local JSONL file or an https URL.

    Streams and stops at ``limit``: the published corpus is ~570 MB of which a
    CI run reads a few tens of conversations, so it is never fully downloaded.
    """
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as f:
            yield from _iter_json_lines(f, limit)
        return

    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(
        source, stream=True, timeout=_DOWNLOAD_TIMEOUT, headers=headers
    ) as response:
        response.raise_for_status()
        lines = response.iter_lines(
            chunk_size=_DOWNLOAD_CHUNK_SIZE, decode_unicode=True
        )
        yield from _iter_json_lines(lines, limit)


def _iter_json_lines(lines: Iterable[str], limit: Optional[int]) -> Iterator[dict]:
    count = 0
    for line in lines:
        if not line or not line.strip():
            continue
        yield json.loads(line)
        count += 1
        if limit is not None and count >= limit:
            return


def conversation_turns(
    record: dict, include_subagents: bool = False
) -> List[List[RecordedTurn]]:
    """Recorded turns, one list per conversation in ``record``."""

    def turn(request: dict) -> RecordedTurn:
        return RecordedTurn(int(request["in"]), int(request["out"]))

    main: List[RecordedTurn] = []
    subagents: List[List[RecordedTurn]] = []
    for request in record.get("requests", []):
        if request.get("type") == MAIN_AGENT_REQUEST_TYPE:
            main.append(turn(request))
        elif request.get("type") == SUBAGENT_GROUP_TYPE and include_subagents:
            inner = [turn(r) for r in request.get("requests", [])]
            if inner:
                subagents.append(inner)
    return ([main] if main else []) + subagents


def plan_replay(
    turns: Sequence[RecordedTurn],
    output_len: Optional[int] = None,
    max_turns: Optional[int] = None,
    min_delta_tokens: int = DEFAULT_MIN_DELTA_TOKENS,
    context_reset_ratio: float = DEFAULT_CONTEXT_RESET_RATIO,
) -> List[List[PlannedTurn]]:
    """Turn recorded ``(in, out)`` pairs into replayable conversation segments.

    Replay accumulates the tokens a round contributes plus the reply it
    generates, so ``new_k = in_k - (in_{k-1} + out_{k-1})`` reproduces the
    corpus exactly -- the same arithmetic the corpus itself follows, and why
    generating the recorded reply length matters.

    Replay can only append, so a recorded prompt below the accumulated history
    cannot be followed within one conversation. A small shortfall (the trace
    grew by less than ``min_delta_tokens``, or trimmed a message) clamps to
    ``min_delta_tokens`` and re-syncs on the next round that grows past the
    offset. A drop past ``context_reset_ratio`` is a context reset and starts a
    new segment, which is what the agent did.

    ``output_len`` overrides the recorded reply lengths with a uniform value.
    """
    if max_turns is not None:
        turns = turns[:max_turns]

    segments: List[List[PlannedTurn]] = []
    segment: List[PlannedTurn] = []
    replayed = 0
    for recorded in turns:
        if segment and recorded.prompt_tokens < replayed * (1 - context_reset_ratio):
            segments.append(segment)
            segment = []
            replayed = 0

        new_tokens = max(recorded.prompt_tokens - replayed, min_delta_tokens)
        out_tokens = recorded.output_tokens if output_len is None else output_len
        replayed += new_tokens
        segment.append(PlannedTurn(new_tokens, out_tokens, replayed))
        replayed += out_tokens

    if segment:
        segments.append(segment)
    return segments


def recommended_output_len(records: Iterable[dict], include_subagents: bool = False):
    """Mean recorded completion length, rounded, or ``None`` for an empty corpus."""
    total = 0
    count = 0
    for record in records:
        for conversation in conversation_turns(record, include_subagents):
            for recorded in conversation:
                total += recorded.output_tokens
                count += 1
    return round(total / count) if count else None


class SyntheticTurnText:
    """Deterministic filler text of an exact token length.

    Token ids are sampled from the tokenizer's own vocabulary and the result is
    re-encoded and corrected, because decoding random ids and re-tokenizing the
    text does not round-trip: adjacent pieces merge. Prompt length is the thing
    this whole conversion exists to reproduce, so it is worth the extra passes.
    """

    def __init__(
        self,
        tokenizer,
        seed: int = 42,
        max_correction_rounds: int = 3,
        tolerance: float = 0.005,
    ):
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_correction_rounds = max_correction_rounds
        self.tolerance = tolerance
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.vocab = sorted(
            token_id
            for token_id in tokenizer.get_vocab().values()
            if isinstance(token_id, int) and token_id not in special
        )
        if not self.vocab:
            raise ValueError("Tokenizer exposes no usable vocabulary ids")

    def _encoded_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, num_tokens: int, key: str) -> str:
        num_tokens = max(int(num_tokens), 1)
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        # Correction is bounded rather than exact: re-encoding a 64k-token turn
        # is not free, and a fraction of a percent of prompt drift does not
        # change what the engine is being measured on.
        slack = max(1, int(num_tokens * self.tolerance))
        ids = rng.choices(self.vocab, k=num_tokens)
        text = self.tokenizer.decode(ids)
        for _ in range(self.max_correction_rounds):
            actual = self._encoded_len(text)
            if abs(actual - num_tokens) <= slack:
                break
            if actual < num_tokens:
                ids.extend(rng.choices(self.vocab, k=num_tokens - actual))
            else:
                del ids[max(num_tokens - actual, 1 - len(ids)) :]
            text = self.tokenizer.decode(ids)
        return text


def build_agentic_trace(
    records: Iterable[dict],
    tokenizer,
    output_len: Optional[int] = None,
    max_turns: Optional[int] = None,
    max_conversations: Optional[int] = None,
    include_subagents: bool = False,
    min_delta_tokens: int = DEFAULT_MIN_DELTA_TOKENS,
    seed: int = 42,
    source: str = DEFAULT_WEKA_TRACE_REPO,
):
    """Convert WekaTrace records into an agentic-trace document plus its stats.

    ``output_len`` overrides the recorded per-turn reply lengths with a uniform
    value; leaving it unset replays the corpus' own lengths.
    """
    filler = SyntheticTurnText(tokenizer, seed=seed)

    conversations: List[List[Dict[str, Any]]] = []
    num_turns = 0
    total_prompt_tokens = 0
    max_prompt_tokens = 0
    total_output_tokens = 0

    for record_index, record in enumerate(records):
        if max_conversations is not None and len(conversations) >= max_conversations:
            break

        streams = [
            segment
            for stream in conversation_turns(record, include_subagents)
            for segment in plan_replay(
                stream,
                output_len=output_len,
                max_turns=max_turns,
                min_delta_tokens=min_delta_tokens,
            )
        ]
        for stream_index, plan in enumerate(streams):
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            conversation: List[Dict[str, Any]] = []
            for turn_index, planned in enumerate(plan):
                key = f"{record.get('id', record_index)}:{stream_index}:{turn_index}"
                # Every round is a plain user turn: the chat template is applied
                # server-side, and a uniform role keeps the conversion portable
                # across templates that treat system blocks specially.
                conversation.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": filler.generate(planned.new_tokens, key),
                            }
                        ],
                        "prompt_tokens": planned.prompt_tokens,
                        "output_tokens": planned.output_tokens,
                    }
                )
                max_prompt_tokens = max(max_prompt_tokens, planned.prompt_tokens)
                total_prompt_tokens += planned.prompt_tokens
                total_output_tokens += planned.output_tokens

            num_turns += len(conversation)
            conversations.append(conversation)

    if not conversations:
        raise ValueError(f"No usable conversations found in {source}")

    document = {
        "metadata": {
            "source": source,
            "generator": "sglang.test.agentic_trace_utils",
            "output_len": output_len,
            "max_turns": max_turns,
            "include_subagents": include_subagents,
            "seed": seed,
        },
        "conversations": conversations,
    }
    stats = TraceStats(
        num_conversations=len(conversations),
        num_turns=num_turns,
        total_prompt_tokens=total_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
        total_output_tokens=total_output_tokens,
        mean_output_len=total_output_tokens / num_turns if num_turns else 0.0,
    )
    return document, stats


def ensure_agentic_trace_file(
    tokenizer,
    output_dir: str,
    num_conversations: int,
    output_len: Optional[int] = None,
    max_turns: Optional[int] = None,
    include_subagents: bool = False,
    seed: int = 42,
    source: Optional[str] = None,
):
    """Return ``(path, stats)`` for the converted trace, building it if absent.

    ``source`` is a local WekaTrace JSONL or an https URL; it defaults to the
    published corpus. The conversion is tokenizer-specific and slow enough
    (millions of tokens re-encoded) to be worth caching between runs, so a cache
    hit also avoids re-downloading the corpus.
    """
    source = source or weka_trace_url()
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "source": source,
                "tokenizer": getattr(tokenizer, "name_or_path", str(type(tokenizer))),
                "num_conversations": num_conversations,
                "output_len": output_len,
                "max_turns": max_turns,
                "include_subagents": include_subagents,
                "seed": seed,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"agentic_trace_{fingerprint}.json")
    stats_path = f"{path}.stats.json"
    if os.path.isfile(path) and os.path.isfile(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return path, TraceStats(**json.load(f))

    # Sub-agent groups turn one record into several conversations, so cap the
    # records read by the conversation budget rather than assuming one each.
    records = list(iter_weka_records(source, limit=num_conversations))
    document, stats = build_agentic_trace(
        records,
        tokenizer,
        output_len=output_len,
        max_turns=max_turns,
        max_conversations=num_conversations,
        include_subagents=include_subagents,
        seed=seed,
        source=source,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f)
    return path, stats
