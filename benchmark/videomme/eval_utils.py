"""Response Parsing and Evaluation for LongVideoBench."""

import argparse
import dataclasses


@dataclasses.dataclass
class EvalArgs:
    seed: int = 42
    split: str = "lvb_val.json"
    result_filename: str = "./val_sglang.json"
    output_path: str = "."
    dataset_path: str = "longvideobench/LongVideoBench"
    concurrency: int = 1
    max_new_tokens: int = 30
    response_answer_regex: str = "(.*)"
    max_num_frames: int = 64
    batch_size: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--result-filename", type=str, default=EvalArgs.result_filename
        )
        parser.add_argument(
            "--output-path", type=str, default=EvalArgs.output_path
        )

        parser.add_argument("--dataset-path", type=str, default=EvalArgs.dataset_path)
        parser.add_argument("--seed", type=int, default=EvalArgs.seed)
        parser.add_argument("--split", type=str, default=EvalArgs.split)
        parser.add_argument("--concurrency", type=int, default=EvalArgs.concurrency)
        parser.add_argument(
            "--max-new-tokens", type=int, default=EvalArgs.max_new_tokens
        )
        parser.add_argument(
            "--response-answer-regex",
            type=str,
            default=EvalArgs.response_answer_regex,
        )
        parser.add_argument(
            "--max-num-frames", type=int, default=EvalArgs.max_num_frames
        )
        parser.add_argument(
            "--batch-size", type=int, default=EvalArgs.batch_size
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})
