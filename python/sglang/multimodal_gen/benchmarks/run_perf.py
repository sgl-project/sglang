import argparse
import os

from tqdm import tqdm

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import maybe_dump_performance
from sglang.multimodal_gen.runtime.server_args import ServerArgs

def run_benchmark(args):
    args.request_id = "mocked_fake_id_for_offline_generate"
    os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "True"
    envs.SGLANG_DIFFUSION_STAGE_LOGGING = True
    server_args = ServerArgs.from_cli_args(args)
    sampling_params_kwargs = SamplingParams.get_cli_args(args)

    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path, server_args=server_args
    )

    sampling_params_kwargs["save_output"] = False
    for _ in tqdm(range(args.warmup)):
        generator.generate(
            sampling_params_kwargs=sampling_params_kwargs
        )
    
    sampling_params_kwargs["save_output"] = True
    results = generator.generate(
        sampling_params_kwargs=sampling_params_kwargs
    )

    maybe_dump_performance(args, server_args, args.prompt, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        required=True,
        help="Model path compatible with Hugging Face Transformers."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A Logo With Bold Large Text: SGL Diffusion",
        required=False,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        required=False,
        help="Number of warmup iters before performance trial."
    )
    parser.add_argument(
        "--perf-dump-path",
        type=str,
        default="perf.json",
        required=False,
        help="Path to dump the performance metrics (JSON) for the run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    args = parser.parse_args()

    run_benchmark(args)