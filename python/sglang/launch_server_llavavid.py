import argparse
import multiprocessing as mp

from sglang.srt.server import ServerArgs, launch_server

if __name__ == "__main__":

    model_overide_args = {}

    model_overide_args["mm_spatial_pool_stride"] = 2
    model_overide_args["architectures"] = ["LlavaVidForCausalLM"]
    model_overide_args["num_frames"] = 16
    model_overide_args["model_type"] = "llavavid"
    if model_overide_args["num_frames"] == 32:
        model_overide_args["rope_scaling"] = {"factor": 2.0, "type": "linear"}
        model_overide_args["max_sequence_length"] = 4096 * 2
        model_overide_args["tokenizer_model_max_length"] = 4096 * 2
        model_overide_args["model_max_length"] = 4096 * 2

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    if "34b" in args.model_path.lower():
        model_overide_args["image_token_index"] = 64002

    server_args = ServerArgs.from_cli_args(args)

    pipe_reader, pipe_writer = mp.Pipe(duplex=False)

    launch_server(server_args, pipe_writer, model_overide_args)
