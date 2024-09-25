"""Launch the inference server for Llava-video model."""

import json
import sys

from sglang.srt.server import launch_server, prepare_server_args

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    model_override_args = {}
    model_override_args["mm_spatial_pool_stride"] = 2
    model_override_args["architectures"] = ["LlavaVidForCausalLM"]
    model_override_args["num_frames"] = 16
    model_override_args["model_type"] = "llavavid"
    if model_override_args["num_frames"] == 32:
        model_override_args["rope_scaling"] = {"factor": 2.0, "type": "linear"}
        model_override_args["max_sequence_length"] = 4096 * 2
        model_override_args["tokenizer_model_max_length"] = 4096 * 2
        model_override_args["model_max_length"] = 4096 * 2
    if "34b" in server_args.model_path.lower():
        model_override_args["image_token_index"] = 64002
    server_args.json_model_override_args = json.dumps(model_override_args)

    launch_server(server_args)
