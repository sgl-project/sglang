import json
from collections.abc import Iterator
from pathlib import Path

# Must match SuffixAutomaton::kSeparatorToken in suffix_automaton.h.
SEPARATOR_TOKEN = -(2**31)

# Default chunk size for streaming tokenized documents into the SAM.
DEFAULT_CHUNK_SIZE = 4096


def iter_external_corpus_chunks(
    path: str, tokenizer, max_tokens: int, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[list[int]]:
    """Chunk documents and yield fixed-size token chunks from a JSONL corpus file."""
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise ValueError(f"External ngram corpus path does not exist: {path}")
    if tokenizer is None:
        raise ValueError("A tokenizer is required to load an external ngram corpus.")
    if max_tokens <= 0:
        raise ValueError("External ngram corpus max tokens must be positive.")

    total_tokens = 0
    has_previous_doc = False
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in external ngram corpus at line {line_no}: {e.msg}"
                ) from e

            if not isinstance(record, str):
                raise ValueError(
                    "Invalid external ngram corpus record at line "
                    f"{line_no}: expected a JSON string."
                )

            token_ids = list(tokenizer.encode(record, add_special_tokens=False))
            if not token_ids:
                continue

            separator_cost = 1 if has_previous_doc else 0
            next_total_tokens = total_tokens + separator_cost + len(token_ids)
            if next_total_tokens > max_tokens:
                raise ValueError(
                    "External ngram corpus exceeds the configured token limit "
                    f"({max_tokens}) at line {line_no} after loading "
                    f"{total_tokens} tokens."
                )
            total_tokens = next_total_tokens

            if has_previous_doc:
                token_ids = [SEPARATOR_TOKEN] + token_ids
            for i in range(0, len(token_ids), chunk_size):
                yield token_ids[i : i + chunk_size]
            has_previous_doc = True
