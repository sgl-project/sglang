
from sglang.srt.server import ServerArgs, launch_server

import multiprocessing as mp

import argparse

# model_overide_args["image_token_index"] = 64002



if __name__ == "__main__":

    model_overide_args = {}

    model_overide_args["mm_spatial_pool_stride"] = 2
    model_overide_args["architectures"] = ["LlavaVidForCausalLM"]
    model_overide_args["num_frames"] = 16
    model_overide_args["model_type"] = "llavavid"

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    pipe_reader, pipe_writer = mp.Pipe(duplex=False)


    launch_server(server_args, pipe_writer, model_overide_args)