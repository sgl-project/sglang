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
"""CLI argument definitions for model and tokenizer."""

import argparse
import json


class ModelArgs:
    """CLI argument definitions for model and tokenizer, API, and model override."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        from sglang.srt.args.constants import LOAD_FORMAT_CHOICES
        from sglang.srt.utils.common import human_readable_int

        # Model and tokenizer
        parser.add_argument(
            "--model-path",
            "--model",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--tokenizer-worker-num",
            type=int,
            default=ServerArgs.tokenizer_worker_num,
            help="The worker num of the tokenizer manager.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=LOAD_FORMAT_CHOICES,
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling."
            '"gguf" will load the weights in the gguf format. '
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller.",
        )
        parser.add_argument(
            "--model-loader-extra-config",
            type=str,
            help="Extra config for model loader. "
            "This will be passed to the model loader corresponding to the chosen load_format.",
            default=ServerArgs.model_loader_extra_config,
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--context-length",
            type=human_readable_int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead)."
            + f"\n\n{human_readable_int.__doc__}",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--enable-multimodal",
            default=ServerArgs.enable_multimodal,
            action="store_true",
            help="Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--model-impl",
            type=str,
            default=ServerArgs.model_impl,
            help="Which implementation of the model to use.\n\n"
            '* "auto" will try to use the SGLang implementation if it exists '
            "and fall back to the Transformers implementation if no SGLang "
            "implementation is available.\n"
            '* "sglang" will use the SGLang model implementation.\n'
            '* "transformers" will use the Transformers model '
            '* "mindspore" will use the MindSpore model '
            "implementation.\n",
        )

        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--admin-api-key",
            type=str,
            default=ServerArgs.admin_api_key,
            help=(
                "Set admin API key for sensitive management endpoints (e.g. /clear_hicache_storage_backend). "
                "When set, admin endpoints require this key and do NOT accept --api-key."
            ),
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--weight-version",
            type=str,
            default=ServerArgs.weight_version,
            help="Version identifier for the model weights. Defaults to 'default' if not specified.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--hf-chat-template-name",
            type=str,
            default=ServerArgs.hf_chat_template_name,
            help="When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), "
            "specify which named template to use. If not set, the first available template is used.",
        )
        parser.add_argument(
            "--completion-template",
            type=str,
            default=ServerArgs.completion_template,
            help="The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
        )
        parser.add_argument(
            "--file-storage-path",
            type=str,
            default=ServerArgs.file_storage_path,
            help="The path of the file storage in backend.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=list(ReasoningParser.DetectorMap.keys()),
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models, supported parsers are: {list(ReasoningParser.DetectorMap.keys())}.",
        )
        tool_call_parser_choices = list(FunctionCallParser.ToolCallParserEnum.keys())
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=tool_call_parser_choices,
            default=ServerArgs.tool_call_parser,
            help=f"Specify the parser for handling tool-call interactions. Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--tool-server",
            type=str,
            default=None,
            help="Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
        )
        parser.add_argument(
            "--sampling-defaults",
            type=str,
            choices=["openai", "model"],
            default=ServerArgs.sampling_defaults,
            help="Where to get default sampling parameters. "
            "'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). "
            "'model' uses the model's generation_config.json to get the recommended "
            "sampling parameters if available. Default is 'model'.",
        )

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.json_model_override_args,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=json.loads,
            help="json-formatted sampling settings that will be returned in /get_model_info",
        )
