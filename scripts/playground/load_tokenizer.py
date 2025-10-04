import argparse
import code

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    args = parser.parse_args()

    t = get_tokenizer(args.name)
    code.interact(local=locals())
