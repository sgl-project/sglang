"""Build a multi-turn benchmarking dataset by padding every OpenHands trajectory
to a fixed turn budget via a unique synthetic system prompt + per-turn padding.
The output JSON is consumed unchanged by evalscope's ``swe_smith`` perf dataset
plugin (``--dataset swe_smith --dataset-path <output>``).

Why padding
-----------
* The strict recipe (74160 first-turn bare tokens + 12 x 753) qualifies only ~4
  out of 67k OpenHands trajectories - the dataset distribution simply doesn't
  have many 80k-prompt trajectories.  With only 4 unique conversations, the
  evalscope sweep cycles them ~8x at --number 32 and the headline cache-hit
  number is inflated by replays.
* This builder pads every trajectory to exactly the recipe shape using a
  unique-per-conversation synthetic system prompt + unique-per-turn padding.
  All 67k rows qualify, so we can build any --number with no cycling and the
  recipe-derived 92% cache-hit math holds per conversation.
* Each conversation's synthetic content is unique so no incidental
  cross-conversation prefix-cache reuse happens.

Pad sources (``--pad-source``)
------------------------------
* ``openscience`` (default): fill pad slots with real reasoning traces from
  ``nvidia/OpenScienceReasoning-2`` (an R1 dataset) instead of random gibberish.
  Science is an *orthogonal* domain to OpenHands (coding), so it contributes
  little cross-domain prefix-cache reuse while keeping the context window filled
  with coherent text.  R1 markup (``<think>``/``</think>``) is stripped and any
  special/control token ids are filtered out so the pad stays compatible with
  the native serving tokenizer (GLM-5.2, where ``<think>``/``</think>`` are the
  model's own reasoning special tokens).  Each conversation is handed a disjoint
  contiguous span of the science corpus, so no two conversations share a prefix.
* ``random``: a per-turn random-ASCII padder, kept as a fallback.

Bit-identical shape across pad sources
--------------------------------------
Pad length is controlled by an *exact bare-token target* (encode -> slice ->
decode, which round-trips with zero drift on GLM-5.2's fast tokenizer), applied
identically to both sources.  The two sources therefore produce the same
per-turn token counts; only the pad *content* differs.
"""

import argparse
import json
import logging
import os
import re
import string
from typing import Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages
from tqdm import tqdm

# Random-text alphabet for the synthetic padder.  Mostly alphanumerics plus a
# light sprinkle of whitespace/punctuation.  Only used as a *candidate* string
# for the ``random`` pad source; the exact-token sizer trims it to the target,
# so the precise chars/token ratio does not have to be pinned (it just has to
# over-generate enough characters to cover the target).
_PAD_ALPHABET = np.array(list(string.ascii_letters + string.digits + " .,;:?!\n"))

# R1 reasoning markup to strip from OpenScience traces before they are used as
# padding.  ``<think>``/``</think>`` are GLM-5.2's native reasoning special
# tokens (ids 154841/154842); ``<|...|>`` / ``<｜...｜>`` are chat-control token
# spellings.  Stripping them keeps the pad free of control tokens so it stays a
# plain natural-language filler compatible with the serving tokenizer.
_R1_THINK_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
_R1_CTRL_RE = re.compile(r"<\|[^>\n]{0,64}?\|>|<｜[^｜\n]{0,64}?｜>")

logging.getLogger("transformers_modules").setLevel(logging.ERROR)


def _sanitize_r1(text: str) -> str:
    """Remove R1 / chat control markup so the pad is plain filler text."""
    text = _R1_THINK_RE.sub(" ", text)
    text = _R1_CTRL_RE.sub(" ", text)
    return text


def _flatten_content(content) -> str:
    """Flatten a message ``content`` into a single string.

    OpenHands trajectories store ``content`` as a string for ``system``,
    ``assistant`` and ``tool``, but ``user`` content is a list of
    ``{type: text, text: ...}`` blocks (with the issue text). Accept either.
    """
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
    """Normalise an OpenHands trajectory into ``[{role, content}, ...]``.

    * ``tool`` role rewritten to ``user`` (chat-template compatibility).
    * Empty / unrecognised roles dropped.
    * Returns system/user/assistant messages in original order.
    """
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


_DEFAULT_DATASET_NAME = "nebius/SWE-rebench-openhands-trajectories"
_DEFAULT_SPLIT = "train"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a padded multi-turn benchmarking dataset where every "
            "conversation is shaped to first_turn_length / subsequent_turn_length "
            "via a unique synthetic system prompt + per-turn random padding."
        )
    )
    p.add_argument(
        "--model", "--model-path", dest="model_path", default="nvidia/GLM-5.2-NVFP4"
    )
    p.add_argument("--dataset-name", default=_DEFAULT_DATASET_NAME)
    p.add_argument("--split", default=_DEFAULT_SPLIT)
    p.add_argument("--only-resolved", action="store_true")
    p.add_argument(
        "--first-turn-length",
        type=int,
        default=74160,
        help="Target bare-token size of turn-1 user content "
        "(synthetic system prompt + real first user msg combined).",
    )
    p.add_argument(
        "--subsequent-turn-length",
        type=int,
        default=753,
        help="Target bare-token size of each subsequent turn's user content "
        "(real + synthetic pad).",
    )
    p.add_argument(
        "--num-turns",
        type=int,
        default=13,
        help="Number of turns per conversation. Recipe-derived 92%% cache-hit "
        "assumes 13 at first/subsequent = 74160/753.",
    )
    p.add_argument("--number", type=int, default=128)
    p.add_argument("--output-path", default="openhand-dataset.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pad-source",
        choices=["openscience", "random"],
        default="openscience",
        help="What fills the synthetic pad slots.  'openscience' uses real "
        "R1 reasoning traces (orthogonal domain, coherent text); 'random' uses "
        "a random-ASCII padder.  Both are sized to the *same* exact "
        "bare-token targets, so the workload shape is identical either way.",
    )
    p.add_argument(
        "--pad-dataset-name",
        default="nvidia/OpenScienceReasoning-2",
        help="HF dataset providing pad text when --pad-source=openscience.",
    )
    p.add_argument(
        "--pad-split",
        default="train",
        help="Split of --pad-dataset-name to stream pad text from.",
    )
    p.add_argument(
        "--pad-text-field",
        default="output",
        help="Field of the pad dataset whose text is used as filler "
        "(OpenScienceReasoning-2 stores the R1 trace in 'output').",
    )
    p.add_argument(
        "--max-real-first-turn-frac",
        type=float,
        default=0.5,
        help="Upper bound on the fraction of turn-1 budget that real user "
        "content can occupy.  The rest is synthetic system prompt.  "
        "Caps how much the trajectory's natural first message can "
        "shrink the synthetic prefix (which is what makes each conv unique).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap dataset rows scanned (debug). Default = unlimited.",
    )
    return p.parse_args()


def _encode_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_bare_tokens(text: str, target_bare_tokens: int, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_bare_tokens:
        return text
    return tokenizer.decode(ids[:target_bare_tokens], skip_special_tokens=True)


def _generate_random_text(
    target_bare_tokens: int,
    rng: np.random.RandomState,
    chars_per_token: float = 1.7,
) -> str:
    """Generate a random-ASCII *candidate* string that encodes to at least
    ``target_bare_tokens`` bare tokens.

    This is only a candidate: the caller trims it to the exact target with
    ``_truncate_to_bare_tokens``.  The chars/token constant therefore only needs
    to be an *over-estimate* so the candidate is guaranteed long enough -- 1.7 is
    comfortably above GLM-5.2 (~1.373) on this alphabet, so the candidate always
    exceeds the target and the trim lands it exactly.
    """
    if target_bare_tokens <= 0:
        return ""
    n_chars = int(target_bare_tokens * chars_per_token) + 8
    idx = rng.randint(0, len(_PAD_ALPHABET), size=n_chars)
    return "".join(_PAD_ALPHABET[idx].tolist())


def _control_token_ids(tokenizer) -> set:
    """Best-effort set of special/added-vocab token ids to keep out of the pad.

    Padding must be plain content; control tokens (chat roles, R1
    ``<think>``/``</think>``, etc.) would otherwise be injected into the prompt
    and interpreted structurally by the serving model.
    """
    ids = set()
    try:
        ids.update(i for i in tokenizer.all_special_ids if i is not None)
    except Exception:
        pass
    try:
        ids.update(tokenizer.get_added_vocab().values())
    except Exception:
        pass
    # Belt-and-suspenders: explicitly include the R1 reasoning markers.
    for marker in ("<think>", "</think>"):
        try:
            ids.update(tokenizer.encode(marker, add_special_tokens=False))
        except Exception:
            pass
    return ids


class SciencePadPool:
    """A special-token-free pool of OpenScience reasoning tokens.

    Streams ``--pad-dataset-name`` (an R1 dataset), sanitises each trace of R1 /
    control markup, encodes it, drops any surviving control token ids, and
    accumulates the ids into one contiguous pool until at least
    ``needed_tokens`` are available.  Each :meth:`next_pad` call vends a
    *disjoint* contiguous span, so every conversation (and every pad slot within
    it) gets unique filler -- no two conversations share a prefix, mirroring the
    uniqueness the random padder gets from a per-conversation RNG.

    The decoded span re-encodes to exactly the requested token count on GLM-5.2's
    fast tokenizer (verified: zero round-trip drift), so science pads hit the
    same exact targets as random pads.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        split: str,
        text_field: str,
        needed_tokens: int,
        max_rows: Optional[int] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._cursor = 0
        control_ids = _control_token_ids(tokenizer)

        from datasets import load_dataset

        # A margin so per-slot boundary trims never run the pool dry, plus a
        # small floor for tiny --number smoke tests.
        target_tokens = max(int(needed_tokens * 1.15) + 4096, 8192)
        print(
            f"Building OpenScience pad pool from {dataset_name} (split={split}, "
            f"field={text_field}); need ~{needed_tokens} tokens, "
            f"collecting ~{target_tokens}..."
        )
        stream = load_dataset(dataset_name, split=split, streaming=True)

        chunks: List[np.ndarray] = []
        collected = 0
        rows = 0
        control_arr = np.array(sorted(control_ids), dtype=np.int64) if control_ids else None
        pbar = tqdm(total=target_tokens, desc="Collecting pad tokens", unit="tok")
        for row in stream:
            if max_rows is not None and rows >= max_rows:
                break
            rows += 1
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
            gained = ids.size
            collected += gained
            pbar.update(min(gained, max(0, target_tokens - pbar.n)))
            if collected >= target_tokens:
                break
        pbar.close()

        self._ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        self._rows_used = rows
        print(
            f"  Pad pool ready: {self._ids.size} tokens from {rows} rows "
            f"(needed ~{needed_tokens})."
        )
        if self._ids.size < needed_tokens:
            print(
                f"  WARNING: pad pool ({self._ids.size} tok) is smaller than "
                f"needed (~{needed_tokens} tok); slots that outrun it fall back "
                f"to random padding."
            )

    def available(self) -> int:
        return int(self._ids.size - self._cursor)

    def next_pad(self, target_bare_tokens: int) -> Optional[str]:
        """Vend a disjoint span decoding to ~target_bare_tokens tokens.

        Returns ``None`` when the pool is exhausted so the caller can fall back
        to random padding for the remaining slots.
        """
        if target_bare_tokens <= 0:
            return ""
        if self.available() < target_bare_tokens:
            return None
        span = self._ids[self._cursor : self._cursor + target_bare_tokens]
        self._cursor += target_bare_tokens
        return self._tokenizer.decode(span.tolist(), skip_special_tokens=True)


def _make_pad(
    target_bare_tokens: int,
    conv_rng: np.random.RandomState,
    tokenizer,
    pad_source: str,
    pool: Optional[SciencePadPool],
) -> str:
    """Produce pad text of exactly ``target_bare_tokens`` bare tokens.

    Both pad sources are sized to the same exact token target, so the workload
    shape is independent of which source is used.
    """
    if target_bare_tokens <= 0:
        return ""
    if pad_source == "openscience" and pool is not None:
        text = pool.next_pad(target_bare_tokens)
        if text is not None:
            return text
        # Pool exhausted -> fall back to random for the remainder.
    candidate = _generate_random_text(target_bare_tokens, conv_rng)
    return _truncate_to_bare_tokens(candidate, target_bare_tokens, tokenizer)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading tokenizer {args.model_path}...")
    # GLM-5.2 DSA checkpoints (e.g. nvidia/GLM-5.2-NVFP4) declare
    # layer_types=["deepseek_sparse_attention", ...], which the installed
    # transformers' strict config validation rejects. SGLang's get_tokenizer
    # carries the bypass for this (commit "Bypass legacy GLM DSA layer types
    # validation"), so prefer it and fall back to AutoTokenizer otherwise.
    try:
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(args.model_path, trust_remote_code=True)
    except Exception as exc:
        print(f"  sglang get_tokenizer unavailable ({exc}); falling back to AutoTokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

    from datasets import load_dataset

    print(f"Loading dataset {args.dataset_name} (split={args.split})...")
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.only_resolved:
        before = len(dataset)
        dataset = dataset.filter(lambda r: r.get("resolved") == 1)
        print(f"  --only-resolved: kept {len(dataset)}/{before} rows.")

    print(
        f"Target shape: first_turn={args.first_turn_length} bare tokens, "
        f"subsequent={args.subsequent_turn_length} bare tokens/turn, "
        f"num_turns={args.num_turns}, number={args.number}, "
        f"pad_source={args.pad_source}"
    )

    # Build the OpenScience pad pool up front (once) when that source is
    # selected.  Upper-bound the token need assuming every pad slot is fully
    # synthetic (real user content only shrinks the need); the pool adds its own
    # margin on top.
    pool: Optional[SciencePadPool] = None
    if args.pad_source == "openscience":
        needed_tokens = args.number * (
            args.first_turn_length
            + (args.num_turns - 1) * args.subsequent_turn_length
        )
        pool = SciencePadPool(
            tokenizer,
            dataset_name=args.pad_dataset_name,
            split=args.pad_split,
            text_field=args.pad_text_field,
            needed_tokens=needed_tokens,
        )

    conversations: List[List[Dict]] = []
    skipped_empty = 0

    # OSL the runtime will append per turn (evalscope --max-tokens, with ignore_eos).
    # Used to estimate prompt_tokens for subsequent turns without rendering the
    # chat template every turn (which is the expensive part).
    OSL_RUNTIME_ESTIMATE = 220

    pbar = tqdm(total=args.number, desc="Building padded conversations")
    for ridx, row in enumerate(dataset):
        if args.max_rows is not None and ridx >= args.max_rows:
            break
        if len(conversations) >= args.number:
            break

        try:
            traj = row.get("trajectory") if isinstance(row, dict) else row["trajectory"]
        except (KeyError, TypeError):
            skipped_empty += 1
            continue
        if not traj:
            skipped_empty += 1
            continue
        try:
            msgs = _extract_messages(traj)
        except (KeyError, TypeError):
            skipped_empty += 1
            continue
        user_msgs = [m for m in msgs if m["role"] == "user"]
        if not user_msgs:
            skipped_empty += 1
            continue

        # Per-conversation RNG so synthetic content is unique but reproducible.
        conv_rng = np.random.RandomState(args.seed + len(conversations))

        # ----- Turn 1: synthetic system prompt + first real user message -----
        first_user_real = user_msgs[0]["content"]
        first_user_real_bare = _encode_len(first_user_real, tokenizer)
        max_real_in_turn1 = max(
            1, int(args.first_turn_length * args.max_real_first_turn_frac)
        )
        real_in_turn1 = min(first_user_real_bare, max_real_in_turn1)
        if first_user_real_bare > real_in_turn1:
            first_user_msg_content = _truncate_to_bare_tokens(
                first_user_real, real_in_turn1, tokenizer
            )
        else:
            first_user_msg_content = first_user_real
        system_target = max(0, args.first_turn_length - real_in_turn1)
        system_content = _make_pad(
            system_target, conv_rng, tokenizer, args.pad_source, pool
        )

        turn_1_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": first_user_msg_content},
        ]
        accumulated = [m.copy() for m in turn_1_messages]
        # Render the chat template once for turn 1 so the metadata reflects the
        # real prompt_tokens the server will see.  For turns 2..N we increment
        # by subsequent_turn_length + OSL_RUNTIME_ESTIMATE which matches the
        # per-turn growth the engine actually pays.
        prompt_tokens_t = len(tokenize_chat_messages(tokenizer, accumulated))
        turns: List[Dict] = [
            {"messages": turn_1_messages, "prompt_tokens": prompt_tokens_t}
        ]

        # ----- Subsequent turns: 1 user message each, sized to subsequent_turn_length -----
        msg_idx = 1
        for _ in range(1, args.num_turns):
            real_part = ""
            if msg_idx < len(user_msgs):
                real_part = user_msgs[msg_idx]["content"]
                msg_idx += 1
            real_bare = _encode_len(real_part, tokenizer)
            if real_bare > args.subsequent_turn_length:
                content = _truncate_to_bare_tokens(
                    real_part, args.subsequent_turn_length, tokenizer
                )
            elif real_bare < args.subsequent_turn_length:
                synth_target = args.subsequent_turn_length - real_bare
                synth_part = _make_pad(
                    synth_target, conv_rng, tokenizer, args.pad_source, pool
                )
                content = real_part + synth_part
            else:
                content = real_part
            user_msg = {"role": "user", "content": content}
            # Approximate prompt_tokens for this turn: previous + (S + OSL).  This
            # matches the recipe's cache-hit formula and avoids re-rendering the
            # whole chat template per turn (which is the slow path).
            prompt_tokens_t = (
                prompt_tokens_t + args.subsequent_turn_length + OSL_RUNTIME_ESTIMATE
            )
            turns.append({"messages": [user_msg], "prompt_tokens": prompt_tokens_t})

        conversations.append(turns)
        pbar.update(1)

    pbar.close()
    print(
        f"\nBuilt {len(conversations)} conversations "
        f"(skipped {skipped_empty} empty trajectories)"
    )

    if not conversations:
        raise SystemExit(
            "Error: no conversations built. Try removing --only-resolved or --max-rows."
        )

    first_pt = [c[0]["prompt_tokens"] for c in conversations]
    last_pt = [c[-1]["prompt_tokens"] for c in conversations]
    print(
        f"  First-turn prompt tokens: min={min(first_pt)}, max={max(first_pt)}, "
        f"avg={int(sum(first_pt) / len(first_pt))}"
    )
    print(
        f"  Last-turn  prompt tokens: min={min(last_pt)}, max={max(last_pt)}, "
        f"avg={int(sum(last_pt) / len(last_pt))}"
    )

    output = {
        "metadata": {
            "model_path": args.model_path,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "only_resolved": bool(args.only_resolved),
            "padded": True,
            "pad_source": args.pad_source,
            "pad_dataset_name": (
                args.pad_dataset_name if args.pad_source == "openscience" else None
            ),
            "first_turn_length": args.first_turn_length,
            "subsequent_turn_length": args.subsequent_turn_length,
            "num_turns": args.num_turns,
            "num_conversations": len(conversations),
        },
        "conversations": conversations,
    }
    print(f"Saving to {args.output_path}...")
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
