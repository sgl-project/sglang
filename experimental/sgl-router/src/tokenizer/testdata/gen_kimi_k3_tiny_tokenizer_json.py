#!/usr/bin/env python3
"""Convert the tiny Kimi-K3 tiktoken vocabulary into a HuggingFace `tokenizer.json`.

`gen_kimi_k3_cases.py` builds `kimi_k3_tiny_vocab/` as a tiktoken rank file plus a
`tokenizer_config.json`, because the PYTHON reference encoder reads that format.
The router encodes through `dynamo_tokenizers::BasetenTokenizer`, which reads a
`tokenizer.json`. This script produces the second from the first so both sides
describe ONE vocabulary and the committed `token_ids` fixtures stay meaningful.

Deliberately dependency-free (no `tokenizers`, no `transformers`, no `tiktoken`):
it has to run in CI and in a bare checkout, and the whole point of the tiny vocab
is that its id-level assertions run without fetching the real 2.8 MB model.

Faithfulness is not asserted here — it is asserted in Rust, by the `kimi_k3`
fixtures, whose expected ids come from the reference encoder over the tiktoken
form of this same vocabulary. If this conversion is wrong, those tests fail.

Usage: python3 gen_kimi_k3_tiny_tokenizer_json.py
"""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TINY_DIR = os.path.join(HERE, "kimi_k3_tiny_vocab")

# The Kimi BPE pre-tokenizer, byte-for-byte the `pat_str` of
# `tokenization_kimi.TikTokenTokenizer` and identical to the `Split` pattern in
# `baseten/kimi-k3-tokenizer`'s own tokenizer.json. The `&&` inside the character
# classes is regex set INTERSECTION, not two literal ampersands.
#
# A constant of the model family rather than something derived from the vocabulary,
# so it is embedded here instead of copied out of a downloaded file — this script
# must not need the gitignored reference sources to regenerate a committed fixture.
KIMI_PAT = (
    r"[\p{Han}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable-codepoint map, as HF `ByteLevel` uses.

    tiktoken keys its ranks by raw bytes; a `tokenizer.json` vocab is JSON strings.
    This is the mapping that makes every byte representable without escapes.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_ENCODER = bytes_to_unicode()


def encode_token(token: bytes) -> str:
    return "".join(BYTE_ENCODER[b] for b in token)


def read_ranks(path):
    """Parse a tiktoken rank file: one `base64(token) rank` pair per line."""
    ranks = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            b64, rank = line.split()
            ranks[base64.b64decode(b64)] = int(rank)
    return ranks


def derive_merge(ranks, token):
    """The pair whose combination forms `token`, as tiktoken would build it.

    Runs tiktoken's own algorithm over `token`'s bytes — repeatedly merge the
    lowest-ranked ADJACENT PAIR present in the table — and returns the last merge
    performed. That pair is what the HF `merges` list needs.

    Note what this deliberately does NOT do: restrict itself to merges ranked
    below `token`. That restriction holds in a real vocabulary, where a token
    always outranks its parts, but this vocabulary is a hand-built subset with
    rank INVERSIONS (b"pon" is 281 while its part b"on" is 298). tiktoken is
    indifferent — it only ever asks "is this pair in the table", never "is it
    cheaper than where I'm headed" — so imposing the ordering here would drop real
    merges and silently tokenize differently from the reference.

    Returns None if the bytes cannot be merged all the way up to `token`.
    """
    parts = [bytes([b]) for b in token]
    last = None
    while len(parts) > 1:
        best_idx, best_rank = None, None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and (best_rank is None or rank < best_rank):
                best_idx, best_rank = i, rank
        if best_idx is None:
            return None
        last = (parts[best_idx], parts[best_idx + 1])
        parts = (
            parts[:best_idx]
            + [parts[best_idx] + parts[best_idx + 1]]
            + parts[best_idx + 2 :]
        )
    return last if parts[0] == token else None


def build(ranks, added_tokens_decoder):
    vocab = {encode_token(tok): rank for tok, rank in ranks.items()}

    merges = []
    whole_only = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) < 2:
            continue
        pair = derive_merge(ranks, token)
        if pair is None:
            # This vocabulary is a hand-built SUBSET, not a merge closure: it holds
            # whole words like b"message" without the intermediate pieces that
            # would build them pairwise, so no HF merge can express them. They stay
            # reachable anyway because `ignore_merges` takes a pre-token that is
            # already in `vocab` whole, without consulting `merges` at all — which
            # is also what tiktoken effectively does for an exact-match token.
            whole_only.append(token)
            continue
        merges.append([encode_token(pair[0]), encode_token(pair[1])])

    added = []
    for id_str, spec in sorted(added_tokens_decoder.items(), key=lambda kv: int(kv[0])):
        added.append(
            {
                "id": int(id_str),
                "content": spec["content"],
                "single_word": spec.get("single_word", False),
                "lstrip": spec.get("lstrip", False),
                "rstrip": spec.get("rstrip", False),
                "normalized": spec.get("normalized", False),
                "special": spec.get("special", True),
            }
        )

    print(
        f"  {len(merges)} merges derived; {len(whole_only)} whole-only tokens "
        f"(reachable via ignore_merges)"
    )

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": KIMI_PAT},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            # Matches the real vocabulary: a pre-token already present in `vocab`
            # is taken whole rather than re-derived through `merges`.
            "ignore_merges": True,
            "vocab": vocab,
            "merges": merges,
        },
    }


def main():
    ranks = read_ranks(os.path.join(TINY_DIR, "tiktoken.model"))
    with open(os.path.join(TINY_DIR, "tokenizer_config.json"), encoding="utf-8") as fh:
        config = json.load(fh)

    out = build(ranks, config.get("added_tokens_decoder", {}))
    path = os.path.join(TINY_DIR, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(
        f"wrote {path}: {len(out['model']['vocab'])} vocab, "
        f"{len(out['model']['merges'])} merges, {len(out['added_tokens'])} added tokens"
    )


if __name__ == "__main__":
    main()
