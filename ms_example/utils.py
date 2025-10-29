# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse


def get_args():
    """get mindspore examples args"""
    parser = argparse.ArgumentParser("sglang-mindspore offline infer")

    parser.add_argument(
        "--model_path,",
        metavar="--model_path",
        dest="model_path",
        required=False,
        default="/home/ckpt/Qwen3-8B",
        help="the model path",
        type=str,
    )
    parser.add_argument(
        "--device,",
        metavar="--device",
        dest="device",
        required=False,
        default="npu",
        help="the device to run",
        type=str,
    )
    parser.add_argument(
        "--model_impl,",
        metavar="--model_impl",
        dest="model_impl",
        required=False,
        default="mindspore",
        help="the model implementation class",
        type=str,
    )
    parser.add_argument(
        "--max_total_tokens,",
        metavar="--max_total_tokens",
        dest="max_total_tokens",
        required=False,
        default=20000,
        help="the max total tokens",
        type=int,
    )
    parser.add_argument(
        "--attention_backend,",
        metavar="--attention_backend",
        dest="attention_backend",
        required=False,
        default="ascend",
        help="the attention backend",
        type=str,
    )
    parser.add_argument(
        "--tp_size,",
        metavar="--tp_size",
        dest="tp_size",
        required=False,
        default=1,
        help="the tp parallel size",
        type=int,
    )
    parser.add_argument(
        "--dp_size,",
        metavar="--dp_size",
        dest="dp_size",
        required=False,
        default=1,
        help="the dp parallel size",
        type=int,
    )
    parser.add_argument(
        "--log_level,",
        metavar="--log_level",
        dest="log_level",
        required=False,
        default="INFO",
        help="the log level for sglang",
        type=str,
    )
    parser.add_argument(
        "--enable_greedy,",
        metavar="--enable_greedy",
        dest="enable_greedy",
        required=False,
        default=False,
        help="enable greedy mode",
        type=bool,
    )
    parser.add_argument(
        "--mem_fraction_static,",
        metavar="--mem_fraction_static",
        dest="mem_fraction_static",
        required=False,
        default=0.8,
        help="the memory fraction static",
        type=float,
    )

    args = parser.parse_args()

    return args
