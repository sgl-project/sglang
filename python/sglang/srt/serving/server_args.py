"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""The arguments of the server."""

import argparse
import dataclasses
import logging
from dataclasses import fields
from typing import Dict, List, Optional

from sglang.srt.serving.engine_args import EngineArgs

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerArgs:
    # The core engine args
    engine_args: EngineArgs  # = field(default_factory=EngineArgs)

    #
    #  The server specifc args
    #
    # Connection
    host: str = "127.0.0.1"
    port: int = 30000

    # OpenAI API
    chat_template: Optional[str] = None
    file_storage_pth: str = "SGLang_storage"

    # Authentication
    api_key: Optional[str] = None

    def __post_init__(self): ...

    def __getattr__(self, item):
        # Avoid recursion by checking if `engine_args` exists first
        if item == "engine_args" or "engine_args" not in self.__dict__:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            )

        # Forward attribute access to engine_args if not found in ServerArgs.
        # For attribute in server_args, it will be found in ServerArgs's __dict__
        # and no entry into this function.
        if hasattr(self.engine_args, item):
            return getattr(self.engine_args, item)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __setattr__(self, key, value):
        # If the attribute exists in ServerArgs, set it directly
        if key in {f.name for f in fields(ServerArgs)}:
            super().__setattr__(key, value)
        # If the attribute exists in EngineArgs, forward it to engine_args
        elif hasattr(self.engine_args, key):
            setattr(self.engine_args, key, value)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    @classmethod
    def from_kwargs(
        cls,
        *args,
        **kwargs: Dict[str, any],
    ) -> "ServerArgs":
        """Creates a ServerArgs instance by separating EngineArgs and ServerArgs parameters."""
        engine_args_fields = {field.name for field in fields(EngineArgs)}
        server_args_fields = {field.name for field in fields(cls)} - {"engine_args"}

        engine_args_dict = {k: v for k, v in kwargs.items() if k in engine_args_fields}
        server_args_dict = {k: v for k, v in kwargs.items() if k in server_args_fields}

        engine_args = EngineArgs(*args, **engine_args_dict)

        return cls(engine_args=engine_args, **server_args_dict)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=EngineArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )
        parser.add_argument(
            "--additional-ports",
            type=int,
            nargs="*",
            default=[],
            help="The additional ports specified for the server.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=EngineArgs.load_format,
            choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=EngineArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=EngineArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" is supported for CUDA 11.8+.',
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=EngineArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=EngineArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "squeezellm",
                "bitsandbytes",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=EngineArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=EngineArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=EngineArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-num-reqs",
            type=int,
            default=EngineArgs.max_num_reqs,
            help="The maximum number of requests to serve in the memory pool. If the model have a large context length, you may need to decrease this value to avoid out-of-memory errors.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=EngineArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=EngineArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            type=int,
            default=EngineArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=EngineArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=EngineArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=EngineArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=EngineArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=EngineArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=EngineArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=EngineArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log the inputs and outputs of all requests.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--file-storage-pth",
            type=str,
            default=ServerArgs.file_storage_pth,
            help="The path of the file storage in backend.",
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=EngineArgs.dp_size,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=EngineArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
            ],
        )

        # Multi-node distributed serving args
        parser.add_argument(
            "--nccl-init-addr",
            type=str,
            help="The nccl init address of multi-node server.",
        )
        parser.add_argument(
            "--nnodes", type=int, default=EngineArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument("--node-rank", type=int, help="The node rank.")

        # Optimization/debug options
        parser.add_argument(
            "--disable-flashinfer",
            action="store_true",
            help="Disable flashinfer attention kernels.",
        )
        parser.add_argument(
            "--disable-flashinfer-sampling",
            action="store_true",
            help="Disable flashinfer sampling kernels.",
        )
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--disable-regex-jump-forward",
            action="store_true",
            help="Disable regex jump-forward.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--disable-disk-cache",
            action="store_true",
            help="Disable disk cache to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            default=False,
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile, experimental feature.",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        )
        parser.add_argument(
            "--enable-mla",
            action="store_true",
            help="Enable Multi-head Latent Attention (MLA) for DeepSeek-V2",
        )
        parser.add_argument(
            "--attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermidiate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels",
        )
        parser.add_argument(
            "--efficient-weight-load",
            action="store_true",
            help="Turn on memory efficient weight loading with quantization (quantize per layer during loading).",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.dp_size = args.data_parallel_size

        # Init EngineArgs
        engine_args_fields = {field.name for field in dataclasses.fields(EngineArgs)}
        engine_args_dict = {
            key: getattr(args, key) for key in engine_args_fields if hasattr(args, key)
        }
        engine_args = EngineArgs(**engine_args_dict)

        # Init ServerArgs with the remaining fields...
        server_args_fields = {field.name for field in dataclasses.fields(cls)} - {
            "engine_args"
        }
        server_args_dict = {
            key: getattr(args, key) for key in server_args_fields if hasattr(args, key)
        }

        return cls(engine_args=engine_args, **server_args_dict)

    def url(self):
        return f"http://{self.host}:{self.port}"
