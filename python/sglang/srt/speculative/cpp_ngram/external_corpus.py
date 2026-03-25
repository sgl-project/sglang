import json
from pathlib import Path
from typing import List


def load_external_corpus_documents(path: str, tokenizer) -> List[List[int]]:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise ValueError(f"External ngram corpus path does not exist: {path}")
    if tokenizer is None:
        raise ValueError(
            "A tokenizer is required to load an external ngram corpus."
        )

    documents: List[List[int]] = []
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

            token_ids = tokenizer.encode(record, add_special_tokens=False)
            if token_ids:
                documents.append(list(token_ids))

    return documents
