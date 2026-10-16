"""Build the padded OpenHands multi-turn dataset consumed by evalscope's ``swe_smith`` perf plugin
(``--dataset swe_smith --dataset-path <output>``).

Public workload reference: sgl-project/sglang@35b99238c23f benchmark/muse_spark_openhands/build_openhands_padded_dataset.py
(blob 7dec8aa0). The workload is the reference one. Every conversation has 13 turns:

* turn 1 is a synthetic system prompt plus the trajectory's first user message, 74160 bare tokens together. The
  system prompt is a span of nvidia/OpenScienceReasoning-2 text that no other conversation uses, so no two
  conversations share a prefix;
* each later turn is the next user message, padded or truncated to 753 bare tokens.

Differences from the reference builder:

* Both public datasets are read at pinned revisions, straight from their parquet files, and each conversation
  records its source row.
* Budgets are exact. Pads are refitted until the message encodes to its target, every message is re-encoded at
  the end, and a miss fails the build.
* Every spelling of an added-vocab token (chat control tokens, ``<think>``, tool-call tags) is removed from both
  real and pad text. The tokenizer would otherwise encode such text as a control token.
* The chat template is rendered with the ``chat_template_kwargs`` that the requests send.
* The tokenizer comes from a tokenizer-only view of the served checkpoint.
* Metadata records the revisions, file hashes, library versions and per-conversation stats.
* The output is written atomically and never over an existing file.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import string
import sys
from typing import Dict, List, Optional

# Parallel batch encoding of the pad corpus; it does not change any token id.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
from tqdm import tqdm

FORMAT = "openhands-padded/v2"
# Taken at import, so it names the code that ran even if the file is replaced during a long build.
with open(os.path.abspath(__file__), "rb") as _f:
    BUILDER_SHA256 = hashlib.sha256(_f.read()).hexdigest()
REFERENCE = {
    "repository": "sgl-project/sglang",
    "commit": "35b99238c23f873a61bbeed69e9a277cfa5a533d",
    "path": "benchmark/muse_spark_openhands/build_openhands_padded_dataset.py",
    "blob": "7dec8aa0c9ba732b70e99d9be0767b8f1ecef0e0",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
# Dataset name -> parquet files read; filled by _pinned_parquet_rows.
SOURCE_FILES: Dict[str, List[str]] = {}
# Arbitrary cap; each refit moves the pad length by the observed miss.
MAX_FIT_ROUNDS = 12

_PAD_ALPHABET = np.array(list(string.ascii_letters + string.digits + " .,;:?!\n"))
# R1 markup and chat-control spellings in any tokenizer's convention (the reference's two patterns).
_R1_THINK_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
_R1_CTRL_RE = re.compile(r"<\|[^>\n]{0,64}?\|>|<｜[^｜\n]{0,64}?｜>")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        help="tokenizer-only view of the served checkpoint",
    )
    p.add_argument(
        "--tokenizer-manifest",
        required=True,
        help="sweep_tool.py tokenizer-view output",
    )
    p.add_argument(
        "--chat-template-kwargs",
        default="{}",
        help="JSON; must equal what the requests send",
    )
    p.add_argument(
        "--dataset-name", default="nebius/SWE-rebench-openhands-trajectories"
    )
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--only-resolved", action="store_true")
    p.add_argument("--first-turn-length", type=int, default=74160)
    p.add_argument("--subsequent-turn-length", type=int, default=753)
    p.add_argument("--num-turns", type=int, default=13)
    p.add_argument(
        "--osl",
        type=int,
        default=220,
        help="output tokens per turn (evalscope --max-tokens)",
    )
    p.add_argument("--context-length", type=int, default=131073)
    p.add_argument("--number", type=int, default=1024)
    p.add_argument("--output-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pad-source", choices=["openscience", "random"], default="openscience"
    )
    p.add_argument("--pad-dataset-name", default="nvidia/OpenScienceReasoning-2")
    p.add_argument("--pad-dataset-revision", required=True)
    p.add_argument("--pad-split", default="train")
    p.add_argument("--pad-text-field", default="output")
    p.add_argument(
        "--max-real-first-turn-frac",
        type=float,
        default=0.5,
        help="upper bound on the share of the turn-1 budget taken by the real first message",
    )
    p.add_argument(
        "--max-rows", type=int, default=None, help="cap source rows scanned (debug)"
    )
    return p.parse_args()


# --------------------------------------------------------------------------- text helpers


def _flatten_content(content) -> str:
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
    """OpenHands trajectory -> [{role, content}], tool results as user messages, as in the reference."""
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


def make_sanitizer(tokenizer):
    """A function that blanks every added-vocab spelling plus the reference's R1/control patterns."""
    spellings = sorted(
        {t for t in tokenizer.get_added_vocab() if t.strip()}
        | set(tokenizer.all_special_tokens),
        key=len,
        reverse=True,
    )
    added_re = (
        re.compile("|".join(re.escape(s) for s in spellings)) if spellings else None
    )

    def sanitize(text: str) -> str:
        text = _R1_THINK_RE.sub(" ", text)
        text = _R1_CTRL_RE.sub(" ", text)
        return added_re.sub(" ", text) if added_re is not None else text

    return sanitize


def control_token_ids(tokenizer) -> np.ndarray:
    ids = {i for i in tokenizer.all_special_ids if i is not None}
    ids.update(tokenizer.get_added_vocab().values())
    return np.array(sorted(ids), dtype=np.int64)


class Codec:
    """Bare (no special tokens) encode/decode with the served tokenizer."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return (
            self._tokenizer(texts, add_special_tokens=False)["input_ids"]
            if texts
            else []
        )

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, ids) -> str:
        ids = ids.tolist() if isinstance(ids, np.ndarray) else list(ids)
        return self._tokenizer.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------------------- pad sources


def _pinned_parquet_rows(name, split, revision, columns):
    """Rows of a Hugging Face dataset split at a pinned revision, read in this thread one row group at a time.

    The split's parquet files are the ones `datasets` resolves for load_dataset(name, split=..., revision=...).
    Streaming through `datasets` (with or without parquet pre-buffering) wrote the output of a 3-conversation
    build and then aborted at interpreter exit: an HTTP read on the trajectories file still running (EBADF, then
    PyGILState_Release). Here every file is closed before the generator returns or is closed.
    """
    import pyarrow.parquet as pq
    from datasets import load_dataset_builder
    from huggingface_hub import HfFileSystem

    files = list(load_dataset_builder(name, revision=revision).config.data_files[split])
    if not files or any(f"@{revision}/" not in url for url in files):
        raise SystemExit(
            f"{name} split {split} does not resolve to files at revision {revision}: {files}"
        )
    SOURCE_FILES[name] = files
    fs = HfFileSystem()
    for url in files:
        with (
            fs.open(url.removeprefix("hf://"), "rb") as f,
            pq.ParquetFile(f, pre_buffer=False) as pf,
        ):
            for i in range(pf.num_row_groups):
                yield from pf.read_row_group(
                    i, columns=columns, use_threads=False
                ).to_pylist()


class SciencePadPool:
    """A contiguous, control-token-free id stream from an R1 corpus, vended in disjoint spans.

    As in the reference, each pad slot takes the next unread span, so no two slots (and no two conversations) share
    pad text. Spans are cut by token count and refitted by the caller, so the pool needs a small margin.
    """

    def __init__(
        self,
        codec,
        sanitize,
        control_ids,
        dataset_name,
        revision,
        split,
        text_field,
        needed_tokens,
    ):
        self._cursor = 0
        target_tokens = max(int(needed_tokens * 1.15) + 4096, 8192)
        print(
            f"Pad pool: reading {dataset_name}@{revision} ({split}/{text_field}), collecting ~{target_tokens} tokens"
        )
        stream = _pinned_parquet_rows(
            dataset_name, split, revision, columns=[text_field]
        )
        chunks: List[np.ndarray] = []
        collected = rows = 0
        batch: List[str] = []
        pbar = tqdm(total=target_tokens, desc="pad tokens", unit="tok", mininterval=10)

        def flush():
            nonlocal collected
            for ids in codec.encode_batch(batch):
                arr = np.asarray(ids, dtype=np.int64)
                arr = arr[~np.isin(arr, control_ids)]
                if arr.size and collected < target_tokens:
                    chunks.append(arr)
                    collected += arr.size
                    pbar.update(arr.size)
            batch.clear()

        rows_iter = iter(stream)
        for row in rows_iter:
            rows += 1
            text = row.get(text_field) or ""
            if text:
                batch.append(sanitize(text))
            if len(batch) >= 512:
                flush()
                if collected >= target_tokens:
                    break
        rows_iter.close()
        flush()
        pbar.close()
        self._ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        self.rows_scanned = rows
        print(f"Pad pool ready: {self._ids.size} tokens from {rows} rows")

    @property
    def size(self) -> int:
        return int(self._ids.size)

    @property
    def used(self) -> int:
        return self._cursor

    def peek(self, n: int) -> Optional[np.ndarray]:
        if self._cursor + n > self._ids.size:
            return None
        return self._ids[self._cursor : self._cursor + n]

    def consume(self, n: int) -> None:
        self._cursor += n


def _random_ids(n_tokens: int, rng: np.random.RandomState, codec: Codec) -> np.ndarray:
    """Ids of random ASCII text long enough for n_tokens; the reference's fallback padder."""
    idx = rng.randint(0, len(_PAD_ALPHABET), size=int(n_tokens * 1.7) + 64)
    return np.asarray(
        codec.encode("".join(_PAD_ALPHABET[idx].tolist())), dtype=np.int64
    )


class PadMaker:
    """Pads of an exact bare-token size appended to a prefix text, from the pool or (when it runs dry) random text."""

    # Arbitrary small bounds; a boundary merge moves the count by one or two tokens.
    MAX_PREFIX_TRIM = 3
    MAX_PAD_SKIP = 3

    def __init__(self, codec, sanitize, pool: Optional[SciencePadPool]):
        self._codec = codec
        self._sanitize = sanitize
        self._pool = pool
        self.random_slots = 0
        self.boundary_adjustments = 0

    def fit(self, prefix: str, target: int, rng: np.random.RandomState) -> str:
        """prefix + pad, encoding to exactly `target` bare tokens; prefix must already be within the target.

        When the prefix's last token merges with the pad's first so that no pad length is exact, the prefix loses up
        to MAX_PREFIX_TRIM trailing tokens and/or the pad starts up to MAX_PAD_SKIP tokens later (skipped pool tokens
        are consumed, so pads stay disjoint).
        """
        base = self._codec.count(prefix)
        if base > target:
            raise ValueError(f"prefix has {base} tokens > target {target}")
        if base == target:
            return prefix
        margin = 64 + self.MAX_PREFIX_TRIM + self.MAX_PAD_SKIP
        source = (
            self._pool.peek(target - base + margin) if self._pool is not None else None
        )
        from_pool = source is not None
        if not from_pool:
            source = _random_ids(target - base + margin, rng, self._codec)
        prefix_ids = self._codec.encode(prefix) if prefix else []
        for trim in range(min(self.MAX_PREFIX_TRIM, len(prefix_ids)) + 1):
            head = (
                prefix
                if trim == 0
                else self._codec.decode(prefix_ids[: len(prefix_ids) - trim])
            )
            head_len = self._codec.count(head)
            if head_len > target:
                continue
            for skip in range(self.MAX_PAD_SKIP + 1):
                found = self._fit_span(head, head_len, target, source[skip:])
                if found is not None:
                    text, k = found
                    self.boundary_adjustments += trim > 0 or skip > 0
                    if from_pool:
                        self._pool.consume(skip + k)
                    else:
                        self.random_slots += 1
                    return text
        raise ValueError(
            f"could not fit a pad to {target} tokens after a prefix of {base}"
        )

    def _fit_span(self, prefix, base, target, source):
        """(text, pad length) with prefix + decode(source[:k]) encoding to `target`, or None."""
        estimate = target - base
        tried = {}
        k = estimate
        for _ in range(MAX_FIT_ROUNDS):
            k = min(max(k, 0), len(source))
            if k in tried:
                break
            tried[k] = self._try(prefix, source, k)
            if tried[k][1] == target:
                return tried[k][0], k
            k += target - tried[k][1]
        # The steps cycled across a boundary merge; any nearby exact length will do.
        window = range(max(0, estimate - 8), min(len(source), estimate + 8) + 1)
        for k in sorted(window, key=lambda k: abs(k - estimate)):
            text, n = tried[k] if k in tried else self._try(prefix, source, k)
            if n == target:
                return text, k
        return None

    def _try(self, prefix, source, k):
        text = self._sanitize(prefix + self._codec.decode(source[:k]))
        return text, self._codec.count(text)


def truncate_to(text: str, limit: int, codec: Codec) -> str:
    """Longest decoded prefix of `text` that re-encodes to at most `limit` bare tokens."""
    ids = codec.encode(text)
    if len(ids) <= limit:
        return text
    m = limit
    while m > 0:
        cut = codec.decode(ids[:m])
        over = codec.count(cut) - limit
        if over <= 0:
            return cut
        m -= over
    return ""


# --------------------------------------------------------------------------- conversations


def build_conversation(user_msgs, args, codec, pads, rng) -> List[List[Dict[str, str]]]:
    """Per-turn delta messages: [[system, user], [user], ..., [user]] with exact bare budgets."""
    cap = max(1, int(args.first_turn_length * args.max_real_first_turn_frac))
    first_user = truncate_to(user_msgs[0]["content"], cap, codec)
    system = pads.fit("", args.first_turn_length - codec.count(first_user), rng)
    turns = [
        [{"role": "system", "content": system}, {"role": "user", "content": first_user}]
    ]
    for t in range(1, args.num_turns):
        real = user_msgs[t]["content"] if t < len(user_msgs) else ""
        real = truncate_to(real, args.subsequent_turn_length, codec)
        turns.append(
            [
                {
                    "role": "user",
                    "content": pads.fit(real, args.subsequent_turn_length, rng),
                }
            ]
        )
    return turns


def check_budgets(turns, args, codec, control_ids) -> List[str]:
    """Re-encode every message: turn 1 sums to the first-turn budget, later turns hit theirs, no control ids."""
    contents = [m["content"] for turn in turns for m in turn]
    encoded = codec.encode_batch(contents)
    problems = []
    first = len(encoded[0]) + len(encoded[1])
    if first != args.first_turn_length:
        problems.append(f"turn 1 has {first} bare tokens")
    for t, ids in enumerate(encoded[2:], start=2):
        if len(ids) != args.subsequent_turn_length:
            problems.append(f"turn {t} has {len(ids)} bare tokens")
    if any(
        np.isin(np.asarray(ids, dtype=np.int64), control_ids).any() for ids in encoded
    ):
        problems.append("a message encodes to a control token")
    return problems


def render_len(tokenizer, messages, chat_template_kwargs) -> int:
    """Prompt tokens as the server counts them: render the template to text, then encode the text."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        **chat_template_kwargs,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def assistant_placeholder(codec: Codec, osl: int) -> str:
    """Text of exactly `osl` bare tokens standing in for a replayed assistant turn in the context estimate."""
    text = " ok" * osl
    if codec.count(text) != osl:
        raise SystemExit(
            f"placeholder does not encode to {osl} tokens with this tokenizer"
        )
    return text


def final_turn_messages(turns, placeholder):
    messages = list(turns[0])
    for turn in turns[1:]:
        messages += [{"role": "assistant", "content": placeholder}] + turn
    return messages


# --------------------------------------------------------------------------- provenance


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions():
    versions = {}
    for dist in (
        "transformers",
        "tokenizers",
        "datasets",
        "huggingface-hub",
        "numpy",
        "jinja2",
        "pyarrow",
        "evalscope",
    ):
        versions[dist] = importlib.metadata.version(dist)
    direct = importlib.metadata.distribution("evalscope").read_text("direct_url.json")
    versions["evalscope_commit"] = (
        json.loads(direct).get("vcs_info", {}).get("commit_id", "") if direct else ""
    )
    return versions


def _tokenizer_provenance(args, tokenizer):
    with open(args.tokenizer_manifest) as f:
        manifest = json.load(f)
    files = {
        name: _sha256_file(os.path.join(args.tokenizer, name))
        for name in TOKENIZER_FILES
    }
    if files != manifest["files"]:
        raise SystemExit(
            "tokenizer files changed since sweep_tool.py tokenizer-view recorded them"
        )
    return {
        "path": os.path.abspath(args.tokenizer),
        "model_path": manifest["model_path"],
        "revision": manifest["revision"],
        "revision_source": manifest["revision_source"],
        "files": files,
        "class": type(tokenizer).__name__,
    }


# --------------------------------------------------------------------------- main


def _source_rows(args):
    print(f"Reading {args.dataset_name}@{args.dataset_revision} ({args.split})")
    columns = ["trajectory_id", "instance_id", "trajectory", "resolved"]
    stream = _pinned_parquet_rows(
        args.dataset_name, args.split, args.dataset_revision, columns=columns
    )
    try:
        for ridx, row in enumerate(stream):
            if args.max_rows is not None and ridx >= args.max_rows:
                return
            if args.only_resolved and row.get("resolved") != 1:
                continue
            yield ridx, row
    finally:
        stream.close()


def main():
    args = parse_args()
    if os.path.exists(args.output_path):
        raise SystemExit(f"{args.output_path} exists; refusing to overwrite it")
    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = Codec(tokenizer)
    sanitize = make_sanitizer(tokenizer)
    control_ids = control_token_ids(tokenizer)
    tokenizer_info = _tokenizer_provenance(args, tokenizer)
    placeholder = assistant_placeholder(codec, args.osl)

    pool = None
    if args.pad_source == "openscience":
        needed = args.number * (
            args.first_turn_length + (args.num_turns - 1) * args.subsequent_turn_length
        )
        pool = SciencePadPool(
            codec=codec,
            sanitize=sanitize,
            control_ids=control_ids,
            dataset_name=args.pad_dataset_name,
            revision=args.pad_dataset_revision,
            split=args.pad_split,
            text_field=args.pad_text_field,
            needed_tokens=needed,
        )
    pads = PadMaker(codec=codec, sanitize=sanitize, pool=pool)

    conversations, stats = [], []
    skipped_empty = sanitized_messages = 0
    pbar = tqdm(total=args.number, desc="conversations", mininterval=10)
    source_rows = _source_rows(args)
    for ridx, row in source_rows:
        if len(conversations) >= args.number:
            break
        msgs = _extract_messages(row.get("trajectory") or [])
        user_msgs = []
        for m in msgs:
            if m["role"] != "user":
                continue
            clean = sanitize(m["content"])
            sanitized_messages += clean != m["content"]
            user_msgs.append({"role": "user", "content": clean})
        if not user_msgs:
            skipped_empty += 1
            continue
        rng = np.random.RandomState(args.seed + len(conversations))
        turns = build_conversation(
            user_msgs=user_msgs, args=args, codec=codec, pads=pads, rng=rng
        )
        problems = check_budgets(
            turns=turns, args=args, codec=codec, control_ids=control_ids
        )
        if problems:
            raise SystemExit(f"source row {ridx}: budget check failed: {problems}")

        first_prompt = render_len(tokenizer, turns[0], chat_template_kwargs)
        final_prompt = render_len(
            tokenizer, final_turn_messages(turns, placeholder), chat_template_kwargs
        )
        if final_prompt + args.osl > args.context_length:
            raise SystemExit(
                f"source row {ridx}: last turn ~{final_prompt}+{args.osl} exceeds {args.context_length}"
            )
        # Reference semantics: turn 1 is rendered exactly; later turns add the turn budget plus the output length.
        prompt_tokens = [
            first_prompt + t * (args.subsequent_turn_length + args.osl)
            for t in range(args.num_turns)
        ]
        conversations.append(
            [
                {"messages": m, "prompt_tokens": p}
                for m, p in zip(turns, prompt_tokens, strict=True)
            ]
        )
        stats.append(
            {
                "source_row": ridx,
                "trajectory_id": row.get("trajectory_id"),
                "instance_id": row.get("instance_id"),
                "system_sha256": hashlib.sha256(
                    turns[0][0]["content"].encode()
                ).hexdigest(),
                "first_turn_prompt_tokens": first_prompt,
                "final_turn_prompt_tokens_est": final_prompt,
            }
        )
        pbar.update(1)
    pbar.close()
    source_rows.close()

    if len(conversations) < args.number:
        raise SystemExit(
            f"only {len(conversations)} conversations built, {args.number} requested"
        )
    first = [s["first_turn_prompt_tokens"] for s in stats]
    final = [s["final_turn_prompt_tokens_est"] for s in stats]
    print(
        f"Built {len(conversations)} conversations (skipped {skipped_empty} without user messages)"
    )
    print(f"  turn-1 prompt tokens: min={min(first)} max={max(first)}")
    print(
        f"  turn-{args.num_turns} prompt tokens (estimate): min={min(final)} max={max(final)}"
    )
    if pads.random_slots:
        print(
            f"  WARNING: {pads.random_slots} pad slots fell back to random text (pad pool exhausted)"
        )

    metadata = {
        "format": FORMAT,
        "reference": REFERENCE,
        "builder_sha256": BUILDER_SHA256,
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "model_path": tokenizer_info["model_path"],
        "tokenizer": tokenizer_info,
        "chat_template_kwargs": chat_template_kwargs,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "only_resolved": bool(args.only_resolved),
        "padded": True,
        "pad_source": args.pad_source,
        "pad_dataset_name": args.pad_dataset_name
        if args.pad_source == "openscience"
        else None,
        "pad_dataset_revision": args.pad_dataset_revision,
        "pad_split": args.pad_split,
        "pad_text_field": args.pad_text_field,
        "source_files": SOURCE_FILES,
        "first_turn_length": args.first_turn_length,
        "subsequent_turn_length": args.subsequent_turn_length,
        "num_turns": args.num_turns,
        "osl_estimate": args.osl,
        "context_length": args.context_length,
        "max_real_first_turn_frac": args.max_real_first_turn_frac,
        "seed": args.seed,
        "num_conversations": len(conversations),
        "pad_pool": {
            "rows_scanned": pool.rows_scanned if pool else 0,
            "tokens": pool.size if pool else 0,
            "tokens_used": pool.used if pool else 0,
            "random_fallback_slots": pads.random_slots,
            "boundary_adjustments": pads.boundary_adjustments,
        },
        "sanitized_user_messages": sanitized_messages,
        "budget_check": {
            "messages_checked": len(conversations) * (args.num_turns + 1),
            "violations": 0,
        },
        "library_versions": _library_versions(),
        "argv": sys.argv[1:],
        "conversation_stats": stats,
    }
    output = {"metadata": metadata, "conversations": conversations}
    partial = f"{args.output_path}.partial"
    print(f"Saving to {args.output_path}")
    with open(partial, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(partial, args.output_path)
    with open(f"{args.output_path}.meta.json", "w") as f:
        json.dump(
            {k: v for k, v in metadata.items() if k != "conversation_stats"},
            f,
            indent=1,
        )
    print("Done.")


if __name__ == "__main__":
    main()
