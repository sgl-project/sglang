"""Compatibility entry point for the historical GSM8K smoke-test command."""

import argparse

from sglang.test.few_shot_gsm8k import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=128)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    run_eval(parser.parse_args())


if __name__ == "__main__":
    main()
