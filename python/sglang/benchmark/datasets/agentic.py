"""Agentic multi-turn benchmark dataset.

Replays multi-turn agentic conversations (OpenHands trajectories by default;
SWE-smith selectable via ``--agentic-source-*``) padded to exact per-turn
bare-token shapes so KV-cache behavior is controlled and reproducible.
Unique per-conversation prefixes keep cross-conversation prefix-cache reuse
from inflating cache-hit numbers. Built datasets are cached as JSON
(``{"metadata": {...}, "conversations": [[{messages, prompt_tokens}, ...]]}``)
and reload unchanged via ``--dataset-path``.
"""

import json
import os
import re
import string
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from sglang.benchmark.datasets.common import (
    MULTI_TURN_BACKENDS,
    BaseDataset,
    DatasetRow,
)

# A sized text never exceeds its bare-token target and may fall short by at
# most this many tokens (decode -> re-encode can merge tokens at slice
# boundaries on byte-fallback tokenizers).
PAD_SIZING_MAX_DEFICIT_TOKENS = 8

_PAD_ALPHABET = np.array(list(string.ascii_letters + string.digits + " .,;:?!\n"))

# Reasoning / chat-control markup spellings, stripped from pad text so the
# pad stays free of structurally-interpreted tokens.
_R1_THINK_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
_R1_CTRL_RE = re.compile(r"<\|[^>\n]{0,64}?\|>|<｜[^｜\n]{0,64}?｜>")


def _sanitize_r1(text: str) -> str:
    return _R1_CTRL_RE.sub(" ", _R1_THINK_RE.sub(" ", text))


def _flatten_content(content: Any) -> str:
    """Flatten message ``content``: OpenHands stores user content as a list
    of ``{type: text}`` blocks, other roles as plain strings."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _extract_messages(trajectory: List[Dict]) -> List[Dict[str, str]]:
    """Normalize a trajectory to ``[{role, content}, ...]``: ``tool`` roles
    become ``user``, empty/unknown roles are dropped. Assistant messages are
    discarded downstream — the live server reply replaces them."""
    out: List[Dict[str, str]] = []
    for msg in trajectory:
        role = msg.get("role", "")
        content = _flatten_content(msg.get("content", ""))
        if not content or not role:
            continue
        if role == "tool":
            role = "user"
        if role not in ("system", "user", "assistant"):
            continue
        out.append({"role": role, "content": content})
    return out


def _encode_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_bare_tokens(
    text: str, target_bare_tokens: int, tokenizer
) -> Tuple[str, int]:
    """Trim ``text`` to at most ``target_bare_tokens`` bare tokens and
    return ``(text, realized_len)``.

    Byte-fallback tokenizers can re-encode a decoded slice to a different
    count, so verify and re-trim until the realized length is at most the
    target (it may fall short; see ``PAD_SIZING_MAX_DEFICIT_TOKENS``).
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_bare_tokens:
        return text, len(ids)
    slice_len = target_bare_tokens
    while slice_len > 0:
        candidate = tokenizer.decode(ids[:slice_len], skip_special_tokens=True)
        realized = _encode_len(candidate, tokenizer)
        if realized <= target_bare_tokens:
            return candidate, realized
        # Overshot after re-encoding: shrink the slice by the overshoot.
        slice_len -= realized - target_bare_tokens
    return "", 0


def _generate_random_text(
    target_bare_tokens: int,
    rng: np.random.RandomState,
    chars_per_token: float = 1.7,
) -> str:
    """Random-ASCII candidate encoding to at least ``target_bare_tokens``
    tokens; the caller trims it to the exact target."""
    if target_bare_tokens <= 0:
        return ""
    n_chars = int(target_bare_tokens * chars_per_token) + 8
    idx = rng.randint(0, len(_PAD_ALPHABET), size=n_chars)
    return "".join(_PAD_ALPHABET[idx].tolist())


def _control_token_ids(tokenizer) -> set:
    """Special/added-vocab token ids to keep out of the pad, plus the
    reasoning markers even when not registered as special tokens."""
    ids = {i for i in tokenizer.all_special_ids if i is not None}
    ids.update(tokenizer.get_added_vocab().values())
    for marker in ("<think>", "</think>"):
        ids.update(tokenizer.encode(marker, add_special_tokens=False))
    return ids


def _stream_hf_rows(dataset_name: str, split: str) -> Iterable[Dict]:
    """Stream rows of a HF dataset. Isolated for test injection."""
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, streaming=True)


class SciencePadPool:
    """A special-token-free pool of natural-language pad tokens.

    Streams the pad dataset, strips control markup and token ids, and vends
    disjoint contiguous spans so every conversation gets unique filler.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        split: str,
        text_field: str,
        needed_tokens: int,
        rows: Optional[Iterable[Dict]] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._cursor = 0
        control_ids = _control_token_ids(tokenizer)

        # Margin so per-slot boundary trims never run the pool dry.
        target_tokens = max(int(needed_tokens * 1.15) + 4096, 8192)
        print(
            f"Building pad pool from {dataset_name} (split={split}, "
            f"field={text_field}); need ~{needed_tokens} tokens, "
            f"collecting ~{target_tokens}..."
        )
        if rows is None:
            rows = _stream_hf_rows(dataset_name, split)

        chunks: List[np.ndarray] = []
        collected = 0
        num_rows = 0
        control_arr = (
            np.array(sorted(control_ids), dtype=np.int64) if control_ids else None
        )
        pbar = tqdm(total=target_tokens, desc="Collecting pad tokens", unit="tok")
        for row in rows:
            num_rows += 1
            text = row.get(text_field, "") if isinstance(row, dict) else ""
            if not text:
                continue
            text = _sanitize_r1(text)
            ids = np.asarray(
                tokenizer.encode(text, add_special_tokens=False), dtype=np.int64
            )
            if control_arr is not None and ids.size:
                ids = ids[~np.isin(ids, control_arr)]
            if not ids.size:
                continue
            chunks.append(ids)
            collected += ids.size
            pbar.update(min(ids.size, max(0, target_tokens - pbar.n)))
            if collected >= target_tokens:
                break
        pbar.close()

        self._ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        print(
            f"  Pad pool ready: {self._ids.size} tokens from {num_rows} rows "
            f"(needed ~{needed_tokens})."
        )
        if self._ids.size < needed_tokens:
            print(
                f"  WARNING: pad pool ({self._ids.size} tok) is smaller than "
                f"needed (~{needed_tokens} tok); slots that outrun it fall "
                f"back to random padding."
            )

    def next_pad(self, target_bare_tokens: int) -> Optional[str]:
        """Vend a disjoint span of ~``target_bare_tokens`` tokens, or
        ``None`` once the pool is exhausted."""
        if target_bare_tokens <= 0:
            return ""
        if self._ids.size - self._cursor < target_bare_tokens:
            return None
        span = self._ids[self._cursor : self._cursor + target_bare_tokens]
        self._cursor += target_bare_tokens
        return self._tokenizer.decode(span.tolist(), skip_special_tokens=True)


class _PadProvider:
    """Produces pad text of exact bare-token sizes from the selected source."""

    def __init__(self, tokenizer, pad_source: str, pool: Optional[SciencePadPool]):
        self._tokenizer = tokenizer
        self._pad_source = pad_source
        self._pool = pool
        self._fallback_warned = False

    def make_pad(self, target_bare_tokens: int, conv_rng: np.random.RandomState) -> str:
        if target_bare_tokens <= 0:
            return ""
        text = None
        if self._pad_source == "openscience" and self._pool is not None:
            text = self._pool.next_pad(target_bare_tokens)
            if text is None and not self._fallback_warned:
                print(
                    "WARNING: pad pool exhausted; remaining pad slots fall "
                    "back to random padding."
                )
                self._fallback_warned = True
        if text is None:
            text = _generate_random_text(target_bare_tokens, conv_rng)
        # Enforce the never-over contract even when decode -> re-encode drifts.
        pad, realized = _truncate_to_bare_tokens(
            text, target_bare_tokens, self._tokenizer
        )
        if target_bare_tokens - realized > PAD_SIZING_MAX_DEFICIT_TOKENS:
            print(
                f"WARNING: pad realized {realized} bare tokens for target "
                f"{target_bare_tokens} (deficit exceeds the "
                f"{PAD_SIZING_MAX_DEFICIT_TOKENS}-token tolerance)."
            )
        return pad


def build_agentic_conversations(
    tokenizer,
    *,
    model_path: str,
    source_dataset: str,
    source_split: str,
    source_field: str,
    only_resolved: bool,
    first_turn_len: int,
    subsequent_turn_len: int,
    num_turns: int,
    num_conversations: int,
    pad_source: str,
    pad_dataset: str,
    pad_split: str,
    pad_text_field: str,
    max_real_first_turn_frac: float,
    output_len: int,
    seed: int,
    source_rows: Optional[Iterable[Dict]] = None,
    pad_rows: Optional[Iterable[Dict]] = None,
) -> Dict[str, Any]:
    """Build padded multi-turn conversations in the prebuilt-dataset JSON
    schema. ``source_rows`` / ``pad_rows`` override the HF streams for
    offline tests."""
    pool: Optional[SciencePadPool] = None
    if pad_source == "openscience":
        # Upper bound: every pad slot fully synthetic; real content shrinks it.
        needed_tokens = num_conversations * (
            first_turn_len + (num_turns - 1) * subsequent_turn_len
        )
        pool = SciencePadPool(
            tokenizer,
            dataset_name=pad_dataset,
            split=pad_split,
            text_field=pad_text_field,
            needed_tokens=needed_tokens,
            rows=pad_rows,
        )
    provider = _PadProvider(tokenizer, pad_source, pool)

    if source_rows is None:
        source_rows = _stream_hf_rows(source_dataset, source_split)

    conversations: List[List[Dict]] = []
    skipped_empty = 0

    pbar = tqdm(total=num_conversations, desc="Building padded conversations")
    for row in source_rows:
        if len(conversations) >= num_conversations:
            break
        if not isinstance(row, dict):
            skipped_empty += 1
            continue
        if only_resolved and row.get("resolved") != 1:
            continue
        traj = row.get(source_field)
        if isinstance(traj, str):
            # Some sources (e.g. SWE-smith) store the trajectory as JSON text.
            try:
                traj = json.loads(traj)
            except json.JSONDecodeError:
                skipped_empty += 1
                continue
        if not traj or not isinstance(traj, list):
            skipped_empty += 1
            continue
        msgs = _extract_messages(traj)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        if not user_msgs:
            skipped_empty += 1
            continue

        # Per-conversation RNG: synthetic content unique but reproducible.
        conv_rng = np.random.RandomState(seed + len(conversations))

        first_user_real = user_msgs[0]["content"]
        first_user_real_bare = _encode_len(first_user_real, tokenizer)
        max_real_in_turn1 = max(1, int(first_turn_len * max_real_first_turn_frac))
        real_in_turn1 = min(first_user_real_bare, max_real_in_turn1)
        if first_user_real_bare > real_in_turn1:
            first_user_msg_content, real_in_turn1 = _truncate_to_bare_tokens(
                first_user_real, real_in_turn1, tokenizer
            )
        else:
            first_user_msg_content = first_user_real
        system_target = max(0, first_turn_len - real_in_turn1)
        system_content = provider.make_pad(system_target, conv_rng)

        turn_1_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": first_user_msg_content},
        ]
        # Render the chat template only for turn 1 (the slow path); later
        # turns grow by subsequent_turn_len + output_len, matching the
        # per-turn growth the engine actually pays.
        prompt_tokens_t = len(
            tokenizer.apply_chat_template(
                turn_1_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        turns: List[Dict] = [
            {"messages": turn_1_messages, "prompt_tokens": prompt_tokens_t}
        ]

        # One user message per later turn, synthesized from pad text alone
        # when the trajectory runs dry — turns are never skipped.
        msg_idx = 1
        for _ in range(1, num_turns):
            real_part = ""
            if msg_idx < len(user_msgs):
                real_part = user_msgs[msg_idx]["content"]
                msg_idx += 1
            real_bare = _encode_len(real_part, tokenizer)
            if real_bare > subsequent_turn_len:
                content, _ = _truncate_to_bare_tokens(
                    real_part, subsequent_turn_len, tokenizer
                )
            else:
                # make_pad(0) returns "" (no pool/RNG use), so the exact-fit
                # case appends nothing and keeps the real content unchanged.
                synth_part = provider.make_pad(
                    subsequent_turn_len - real_bare, conv_rng
                )
                content = real_part + synth_part
            user_msg = {"role": "user", "content": content}
            prompt_tokens_t = prompt_tokens_t + subsequent_turn_len + output_len
            turns.append({"messages": [user_msg], "prompt_tokens": prompt_tokens_t})

        conversations.append(turns)
        pbar.update(1)
    pbar.close()

    print(
        f"Built {len(conversations)} conversations "
        f"(skipped {skipped_empty} empty trajectories)"
    )
    if len(conversations) < num_conversations:
        raise ValueError(
            f"Only built {len(conversations)} of the requested "
            f"{num_conversations} conversations; the source dataset "
            f"{source_dataset} (split={source_split}, field={source_field}) "
            "ran out of usable trajectories."
        )

    metadata = {
        "model_path": model_path,
        "dataset_name": source_dataset,
        "split": source_split,
        "only_resolved": bool(only_resolved),
        "padded": True,
        "pad_source": pad_source,
        "pad_dataset_name": pad_dataset if pad_source == "openscience" else None,
        "first_turn_length": first_turn_len,
        "subsequent_turn_length": subsequent_turn_len,
        "num_turns": num_turns,
        "num_conversations": len(conversations),
        # Absent in files built by earlier tooling; compared only when present.
        "tokenizer_path": tokenizer.name_or_path,
        "seed": seed,
        "source_field": source_field,
        "pad_split": pad_split,
        "pad_text_field": pad_text_field,
        "max_real_first_turn_frac": max_real_first_turn_frac,
    }
    return {"metadata": metadata, "conversations": conversations}


# Metadata fields that must match for a cached build to be reused. A field
# absent from the file metadata (older files) is compatible: the file wins.
_CACHE_COMPAT_FIELDS = (
    "model_path",
    "dataset_name",
    "split",
    "only_resolved",
    "pad_source",
    "pad_dataset_name",
    "first_turn_length",
    "subsequent_turn_length",
    "num_turns",
    "tokenizer_path",
    "seed",
    "source_field",
    "pad_split",
    "pad_text_field",
    "max_real_first_turn_frac",
)


def _cache_is_compatible(file_metadata: Dict, expected: Dict) -> bool:
    for key in _CACHE_COMPAT_FIELDS:
        if key in file_metadata and file_metadata[key] != expected[key]:
            return False
    return True


def get_agentic_cache_path(expected_metadata: Dict) -> Path:
    """Configuration-keyed cache path under ~/.cache/sglang/benchmark. The
    conversation count is deliberately not part of the key: growing a cache
    (auto-expand) rewrites the same file."""
    import hashlib

    compat = {k: expected_metadata[k] for k in _CACHE_COMPAT_FIELDS}
    digest = hashlib.sha1(
        json.dumps(compat, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    name = (
        f"agentic_{compat['first_turn_length']}_"
        f"{compat['subsequent_turn_length']}_{compat['num_turns']}_"
        f"{compat['pad_source']}_{digest}.json"
    )
    return Path.home() / ".cache" / "sglang" / "benchmark" / name


def _write_cache_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _load_json_file(path: Path) -> Dict:
    with open(path) as f:
        payload = json.load(f)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("metadata"), dict)
        or not isinstance(payload.get("conversations"), list)
    ):
        raise ValueError(
            f"{path} is not an agentic dataset file: expected a JSON object "
            'with "metadata" and "conversations" keys.'
        )
    return payload


@dataclass
class AgenticDataset(BaseDataset):
    num_prompts: int
    dataset_path: str
    source_dataset: str
    source_split: str
    source_field: str
    only_resolved: bool
    first_turn_len: int
    subsequent_turn_len: int
    num_turns: int
    num_conversations: int
    pad_source: str
    pad_dataset: str
    pad_split: str
    pad_text_field: str
    max_real_first_turn_frac: float
    offset: int
    output_len: int
    cache_path: str
    rebuild: bool
    seed: int

    @classmethod
    def from_args(cls, args: Namespace) -> "AgenticDataset":
        # The CLI hook in serving.py enforces the same rules at parse time;
        # revalidate here for in-process callers.
        if args.backend not in MULTI_TURN_BACKENDS:
            raise ValueError(
                f"--dataset-name agentic requires a multi-turn chat backend "
                f"({', '.join(sorted(MULTI_TURN_BACKENDS))}); got --backend "
                f"{args.backend}."
            )
        if args.agentic_pad_source not in ("openscience", "random"):
            raise ValueError(
                f"--agentic-pad-source must be 'openscience' or 'random', "
                f"got {args.agentic_pad_source!r}"
            )
        if args.agentic_offset < 0:
            raise ValueError(
                f"--agentic-offset must be >= 0, got {args.agentic_offset}"
            )
        return cls(
            num_prompts=args.num_prompts,
            dataset_path=args.dataset_path,
            source_dataset=args.agentic_source_dataset,
            source_split=args.agentic_source_split,
            source_field=args.agentic_source_field,
            only_resolved=args.agentic_only_resolved,
            first_turn_len=args.agentic_first_turn_len,
            subsequent_turn_len=args.agentic_subsequent_turn_len,
            num_turns=args.agentic_num_turns,
            num_conversations=args.agentic_num_conversations,
            pad_source=args.agentic_pad_source,
            pad_dataset=args.agentic_pad_dataset,
            pad_split=args.agentic_pad_split,
            pad_text_field=args.agentic_pad_text_field,
            max_real_first_turn_frac=args.agentic_max_real_first_turn_frac,
            offset=args.agentic_offset,
            output_len=args.agentic_output_len,
            cache_path=args.agentic_cache_path,
            rebuild=args.agentic_rebuild,
            seed=args.seed,
        )

    def load(self, tokenizer, model_id=None) -> List[DatasetRow]:
        needed = self.offset + self.num_prompts
        if self.dataset_path:
            payload = self._load_prebuilt(needed)
        else:
            payload = self._load_or_build(tokenizer, model_id, needed)
        return self._to_rows(payload, needed)

    def _load_prebuilt(self, needed: int) -> Dict:
        payload = _load_json_file(Path(self.dataset_path))
        num_available = len(payload["conversations"])
        if needed > num_available:
            raise ValueError(
                f"Prebuilt dataset {self.dataset_path} has {num_available} "
                f"conversations but --agentic-offset {self.offset} + "
                f"--num-prompts {self.num_prompts} requires {needed}. "
                "Conversations are consumed as a slice (never recycled); "
                "rebuild the file with more conversations or lower the "
                "offset/num-prompts."
            )
        return payload

    def _expected_metadata(self, tokenizer, model_id) -> Dict:
        return {
            "model_path": model_id or tokenizer.name_or_path,
            "dataset_name": self.source_dataset,
            "split": self.source_split,
            "only_resolved": bool(self.only_resolved),
            "padded": True,
            "pad_source": self.pad_source,
            "pad_dataset_name": (
                self.pad_dataset if self.pad_source == "openscience" else None
            ),
            "first_turn_length": self.first_turn_len,
            "subsequent_turn_length": self.subsequent_turn_len,
            "num_turns": self.num_turns,
            "tokenizer_path": tokenizer.name_or_path,
            "seed": self.seed,
            "source_field": self.source_field,
            "pad_split": self.pad_split,
            "pad_text_field": self.pad_text_field,
            "max_real_first_turn_frac": self.max_real_first_turn_frac,
        }

    def _load_or_build(self, tokenizer, model_id, needed: int) -> Dict:
        expected = self._expected_metadata(tokenizer, model_id)
        cache_path = (
            Path(self.cache_path)
            if self.cache_path
            else get_agentic_cache_path(expected)
        )

        if not self.rebuild and cache_path.exists():
            try:
                payload = _load_json_file(cache_path)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                print(f"WARNING: ignoring unreadable cache {cache_path}: {exc}")
                payload = None
            if payload is not None:
                if not _cache_is_compatible(payload["metadata"], expected):
                    print(
                        f"Cache {cache_path} was built with different "
                        "settings; rebuilding."
                    )
                elif len(payload["conversations"]) < needed:
                    print(
                        f"Cache {cache_path} has "
                        f"{len(payload['conversations'])} conversations but "
                        f"{needed} are needed (offset + num-prompts); "
                        "rebuilding with more conversations."
                    )
                else:
                    print(f"Loading cached agentic dataset from {cache_path}")
                    return payload

        build_size = max(self.num_conversations, needed)
        payload = build_agentic_conversations(
            tokenizer,
            model_path=expected["model_path"],
            source_dataset=self.source_dataset,
            source_split=self.source_split,
            source_field=self.source_field,
            only_resolved=self.only_resolved,
            first_turn_len=self.first_turn_len,
            subsequent_turn_len=self.subsequent_turn_len,
            num_turns=self.num_turns,
            num_conversations=build_size,
            pad_source=self.pad_source,
            pad_dataset=self.pad_dataset,
            pad_split=self.pad_split,
            pad_text_field=self.pad_text_field,
            max_real_first_turn_frac=self.max_real_first_turn_frac,
            output_len=self.output_len,
            seed=self.seed,
        )
        print(f"Caching agentic dataset to {cache_path}")
        _write_cache_atomic(cache_path, payload)
        return payload

    def _to_rows(self, payload: Dict, needed: int) -> List[DatasetRow]:
        metadata = payload["metadata"]
        conversations = payload["conversations"]
        # File metadata wins over flags so prebuilt files stay self-consistent.
        subsequent_turn_len = metadata.get(
            "subsequent_turn_length", self.subsequent_turn_len
        )

        rows: List[DatasetRow] = []
        for conversation in conversations[self.offset : needed]:
            turn_messages = [turn["messages"] for turn in conversation]
            # Stored turns-2+ prompt_tokens bake in the build-time output
            # length; recompute from turn 1 so any prebuilt file stays valid
            # under any --agentic-output-len.
            first_turn_prompt_tokens = conversation[0]["prompt_tokens"]
            prompt_lens = [first_turn_prompt_tokens]
            for _ in range(1, len(conversation)):
                prompt_lens.append(
                    prompt_lens[-1] + subsequent_turn_len + self.output_len
                )
            rows.append(
                DatasetRow(
                    prompt=turn_messages,
                    prompt_len=first_turn_prompt_tokens,
                    output_len=self.output_len,
                    prompt_lens=prompt_lens,
                )
            )
        print(
            f"Loaded {len(rows)} agentic conversations "
            f"(offset {self.offset}, {len(conversations)} available)."
        )
        return rows
