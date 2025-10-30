# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import argparse
import os
from typing import Any

from sglang.multimodal_gen import PipelineConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.architectures.preprocess.preprocess_pipeline_i2v import (
    PreprocessPipeline_I2V,
)
from sglang.multimodal_gen.runtime.architectures.preprocess.preprocess_pipeline_ode_trajectory import (
    PreprocessPipeline_ODE_Trajectory,
)
from sglang.multimodal_gen.runtime.architectures.preprocess.preprocess_pipeline_t2v import (
    PreprocessPipeline_T2V,
)
from sglang.multimodal_gen.runtime.architectures.preprocess.preprocess_pipeline_text import (
    PreprocessPipeline_Text,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_world_size,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def main(args) -> None:
    args.model_path = maybe_download_model(args.model_path)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    num_gpus = int(os.environ["WORLD_SIZE"])
    assert num_gpus == 1, "Only support 1 GPU"

    pipeline_config = PipelineConfig.from_pretrained(args.model_path)

    kwargs: dict[str, Any] = {}
    if args.preprocess_task == "text_only":
        kwargs = {
            "text_encoder_cpu_offload": False,
        }
    else:
        # Full config for video/image processing
        kwargs = {
            "vae_precision": "fp32",
            "vae_config": WanVAEConfig(load_encoder=True, load_decoder=True),
        }
    pipeline_config.update_config_from_dict(kwargs)

    server_args = ServerArgs(
        model_path=args.model_path,
        num_gpus=get_world_size(),
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pipeline_config=pipeline_config,
    )
    if args.preprocess_task == "t2v":
        PreprocessPipeline = PreprocessPipeline_T2V
    elif args.preprocess_task == "i2v":
        PreprocessPipeline = PreprocessPipeline_I2V
    elif args.preprocess_task == "text_only":
        PreprocessPipeline = PreprocessPipeline_Text
    elif args.preprocess_task == "ode_trajectory":
        assert args.flow_shift is not None, "flow_shift is required for ode_trajectory"
        server_args.pipeline_config.flow_shift = args.flow_shift
        PreprocessPipeline = PreprocessPipeline_ODE_Trajectory
    else:
        raise ValueError(
            f"Invalid preprocess task: {args.preprocess_task}. "
            f"Valid options: t2v, i2v, ode_trajectory, text_only"
        )

    logger.info(
        "Preprocess task: %s using %s",
        args.preprocess_task,
        PreprocessPipeline.__name__,
    )

    pipeline = PreprocessPipeline(args.model_path, server_args)
    pipeline.forward(batch=None, server_args=server_args, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--preprocess_video_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--samples_per_file", type=int, default=64)
    parser.add_argument(
        "--flush_frequency",
        type=int,
        default=256,
        help="how often to save to parquet files",
    )
    parser.add_argument(
        "--num_latent_t", type=int, default=28, help="Number of latent timesteps."
    )
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range", type=int, default=2.0)
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument(
        "--preprocess_task",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "text_only", "ode_trajectory"],
        help="Type of preprocessing task to run",
    )
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--do_temporal_sample", default=False, action="store_true")
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--training_cfg_rate", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()
    main(args)
