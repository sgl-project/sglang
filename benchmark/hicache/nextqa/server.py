"""
Launch the inference server for Llava-video model.
Example: python server.py --model-path lmms-lab/LLaVA-NeXT-Video-7B --tokenizer-path llava-hf/llava-1.5-7b-hf --port 3000 --chat-template vicuna_v1.1
"""

import argparse
import multiprocessing as mp

from sglang.srt.server import ServerArgs, launch_server

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument(
        "--max-frames",
        type=int,
        choices=[16, 32],
        default=16,
        help="The max number of frames to process in each video. If the input is less then max_frames, the model will pad the input to max_frames, and most of the time the output will be correct. However, if the input is more than max_frames, the model will output wrong answer",
    )
    ServerArgs.add_cli_args(parser)
    # parse cli arguments
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    # model specific arguments
    model_overide_args = {}
    model_overide_args["mm_spatial_pool_stride"] = 2
    model_overide_args["architectures"] = ["LlavaVidForCausalLM"]
    model_overide_args["num_frames"] = args.max_frames
    model_overide_args["model_type"] = "llavavid"
    if model_overide_args["num_frames"] == 32:
        model_overide_args["rope_scaling"] = {"factor": 2.0, "type": "linear"}
        model_overide_args["max_sequence_length"] = 4096 * 2
        model_overide_args["tokenizer_model_max_length"] = 4096 * 2
        model_overide_args["model_max_length"] = 4096 * 2

    print(f"num_frames: {model_overide_args['num_frames']}")

    if "34b" in args.model_path.lower():
        model_overide_args["image_token_index"] = 64002

    pipe_reader, pipe_writer = mp.Pipe(duplex=False)

    launch_server(server_args, pipe_writer, model_overide_args)

"""
Launch the inference server for Llava-video model.
Example: python server.py --model-path lmms-lab/LLaVA-NeXT-Video-7B --tokenizer-path llava-hf/llava-1.5-7b-hf --port 3000 --chat-template vicuna_v1.1
"""

import argparse
import multiprocessing as mp

from sglang.srt.server import ServerArgs, launch_server

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument(
        "--max-frames",
        type=int,
        choices=[16, 32],
        default=16,
        help="The max number of frames to process in each video. If the input is less then max_frames, the model will pad the input to max_frames, and most of the time the output will be correct. However, if the input is more than max_frames, the model will output wrong answer",
    )
    ServerArgs.add_cli_args(parser)
    # parse cli arguments
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    # model specific arguments
    model_overide_args = {}
    model_overide_args["mm_spatial_pool_stride"] = 2
    model_overide_args["architectures"] = ["LlavaVidForCausalLM"]
    model_overide_args["num_frames"] = args.max_frames
    model_overide_args["model_type"] = "llavavid"
    if model_overide_args["num_frames"] == 32:
        model_overide_args["rope_scaling"] = {"factor": 2.0, "type": "linear"}
        model_overide_args["max_sequence_length"] = 4096 * 2
        model_overide_args["tokenizer_model_max_length"] = 4096 * 2
        model_overide_args["model_max_length"] = 4096 * 2

    print(f"num_frames: {model_overide_args['num_frames']}")

    if "34b" in args.model_path.lower():
        model_overide_args["image_token_index"] = 64002

    pipe_reader, pipe_writer = mp.Pipe(duplex=False)

    launch_server(server_args, pipe_writer, model_overide_args)
