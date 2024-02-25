import argparse
import draftretriever
from tqdm import tqdm
from sglang.srt.hf_transformers_utils import get_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Build the reference for the speculative decoding."
    )
    parser.add_argument(
        "--index-file-path", "-i", type=str, help="The index file.", required=True
    )
    parser.add_argument(
        "--tokenizer-path", "-t", type=str, help="The tokenizer path.", required=True
    )
    parser.add_argument(
        "--reference-file-path",
        "-r",
        type=str,
        help="The reference file.",
        required=True,
    )
    args = parser.parse_args()
    print(args)

    tokenizer = get_tokenizer(args.tokenizer_path)

    writer = draftretriever.Writer(
        index_file_path=args.index_file_path, vocab_size=tokenizer.vocab_size
    )

    with open(args.reference_file_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            token_ids = tokenizer.encode(line)
            writer.add_entry(token_ids)

    writer.finalize()


if __name__ == "__main__":
    main()
