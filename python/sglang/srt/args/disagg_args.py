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
"""CLI argument definitions for PD disaggregation."""

import argparse


class DisaggArgs:
    """CLI argument definitions for PD disaggregation and encode prefill disaggregation."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import (
            DISAGG_TRANSFER_BACKEND_CHOICES,
            ENCODER_TRANSFER_BACKEND_CHOICES,
        )

        # PD disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default=ServerArgs.disaggregation_mode,
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=DISAGG_TRANSFER_BACKEND_CHOICES,
            help="The backend for disaggregation transfer. Default is mooncake.",
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )
        parser.add_argument(
            "--disaggregation-ib-device",
            type=str,
            default=ServerArgs.disaggregation_ib_device,
            help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
            "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
        )
        parser.add_argument(
            "--disaggregation-decode-enable-offload-kvcache",
            action="store_true",
            help="Enable async KV cache offloading on decode server (PD mode).",
        )
        parser.add_argument(
            "--num-reserved-decode-tokens",
            type=int,
            default=ServerArgs.num_reserved_decode_tokens,
            help="Number of decode tokens that will have memory reserved when adding new request to the running batch.",
        )
        parser.add_argument(
            "--disaggregation-decode-polling-interval",
            type=int,
            default=ServerArgs.disaggregation_decode_polling_interval,
            help="The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this.",
        )

        # Encode prefill disaggregation
        parser.add_argument(
            "--encoder-only",
            action="store_true",
            help="For MLLM with an encoder, launch an encoder-only server",
        )
        parser.add_argument(
            "--language-only",
            action="store_true",
            help="For VLM, load weights for the language model only.",
        )
        parser.add_argument(
            "--encoder-transfer-backend",
            type=str,
            default=ServerArgs.encoder_transfer_backend,
            choices=ENCODER_TRANSFER_BACKEND_CHOICES,
            help="The backend for encoder disaggregation transfer. Default is zmq_to_scheduler.",
        )
        parser.add_argument(
            "--encoder-urls",
            nargs="+",
            type=str,
            default=[],
            help="List of encoder server urls.",
        )
        parser.add_argument(
            "--enable-adaptive-dispatch-to-encoder",
            default=ServerArgs.enable_adaptive_dispatch_to_encoder,
            action="store_true",
            help="When enabled, adaptively dispatch: multi-image requests go to encoder in language_only epd mode, single-image requests are processed locally.",
        )
