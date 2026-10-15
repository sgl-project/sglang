"""Reconstruct the five original GLM TTFT texts from a pinned InfiniteBench sample."""
import argparse
import json
from pathlib import Path

REVISION = "90f0394333616266d9fe85824ceaf505093cbaa5"
PREFIXES = {131072: 183198, 262144: 364946, 524288: 728585,
            786432: 1092950, 1048569: 1459229}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Reuse an existing longbook_qa_chn.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("ttft-inputs"))
    args = parser.parse_args()
    dataset = args.dataset
    if dataset is None:
        from huggingface_hub import hf_hub_download
        dataset = Path(hf_hub_download(
            "xinrongzhang2022/InfiniteBench", "longbook_qa_chn.jsonl",
            repo_type="dataset", revision=REVISION,
        ))
    with dataset.open("rb") as stream:
        sample = next(row for row in map(json.loads, stream) if row["id"] == 123)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tokens, characters in PREFIXES.items():
        path = args.output_dir / f"prefix-{tokens}.txt"
        path.write_bytes(sample["context"][:characters].encode("utf-8"))
        print(f"{path}: {characters} characters; original GLM tokenizer length {tokens}")


if __name__ == "__main__":
    main()
