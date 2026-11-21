"""Build tests/fixtures/deepseek_v41_mini from a DeepSeek-V4.1 checkpoint.

Keeps the checkpoint's normalizer, pre-tokenizer, post-processor, decoder and
full added vocabulary, over a vocab of the byte-level alphabet plus the first
MERGES trained merges. A trained merge prefix is consistent by construction; a
hand-picked merge list is not, and fastokens and HF tokenizers disagree on
merges that ordinary BPE order can never reach.

Added tokens are renumbered to follow the vocab, as in the checkpoint (vocab
0..128000, added tokens from 128000). HF tokenizers assigns ids to added tokens
outside the model vocab from the vocab size, ignoring the ids in the file, while
fastokens honors them, so keeping the checkpoint's ids over a small vocab makes
the two disagree on a fixture artifact.

Usage: python3 tests/scripts/generate_deepseek_v41_mini.py <checkpoint dir>
"""

import json
import pathlib
import sys

MERGES = 2000
OUT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "deepseek_v41_mini"


def byte_level_alphabet():
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars, extra = printable[:], 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return [chr(c) for c in chars]


def main(checkpoint):
    src = pathlib.Path(checkpoint)
    tok = json.loads((src / "tokenizer.json").read_text())
    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"][:MERGES]

    keep = [k for k, i in sorted(vocab.items(), key=lambda kv: kv[1]) if i < 3]
    keep += byte_level_alphabet()
    keep += ["".join(m.split(" ")) for m in merges]
    mini = {}
    for token in keep:
        mini.setdefault(token, len(mini))
    for merge in merges:
        left, right = merge.split(" ")
        assert left in mini and right in mini, merge

    next_id = len(mini)
    for added in tok["added_tokens"]:
        if added["content"] in mini:
            added["id"] = mini[added["content"]]
        else:
            added["id"] = next_id
            next_id += 1

    tok["model"]["vocab"] = mini
    tok["model"]["merges"] = merges
    OUT.mkdir(parents=True, exist_ok=True)
    # Every file ends in exactly one newline, as the repo's end-of-file hook wants.
    (OUT / "tokenizer.json").write_text(
        json.dumps(tok, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    (OUT / "tokenizer_config.json").write_text(
        (src / "tokenizer_config.json").read_text().rstrip("\n") + "\n"
    )
    config = json.loads((src / "config.json").read_text())
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "model_type": config["model_type"],
                "architectures": config["architectures"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1])
