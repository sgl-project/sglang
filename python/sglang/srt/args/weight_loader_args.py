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
"""CLI argument definitions for custom weight loader."""

import argparse


class WeightLoaderArgs:
    """CLI argument definitions for custom weight loader."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils.common import json_list_type

        # Custom weight loader
        parser.add_argument(
            "--custom-weight-loader",
            type=str,
            nargs="*",
            default=None,
            help="The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func",
        )
        parser.add_argument(
            "--weight-loader-disable-mmap",
            action="store_true",
            help="Disable mmap while loading weight using safetensors.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-seed-instance-ip",
            type=str,
            default=ServerArgs.remote_instance_weight_loader_seed_instance_ip,
            help="The ip of the seed instance for loading weights from remote instance.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-seed-instance-service-port",
            type=int,
            default=ServerArgs.remote_instance_weight_loader_seed_instance_service_port,
            help="The service port of the seed instance for loading weights from remote instance.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-send-weights-group-ports",
            type=json_list_type,
            default=ServerArgs.remote_instance_weight_loader_send_weights_group_ports,
            help="The communication group ports for loading weights from remote instance.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-backend",
            type=str,
            choices=["transfer_engine", "nccl", "modelexpress"],
            default=ServerArgs.remote_instance_weight_loader_backend,
            help="The backend for loading weights from remote instance. Can be 'transfer_engine', 'nccl', or 'modelexpress'. Default is 'nccl'.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-start-seed-via-transfer-engine",
            action="store_true",
            help="Start seed server via transfer engine backend for remote instance weight loader.",
        )
        parser.add_argument(
            "--engine-info-bootstrap-port",
            type=int,
            default=ServerArgs.engine_info_bootstrap_port,
            help="Port for the engine info bootstrap server. Default is 6789. "
            "Must be set explicitly when running multiple instances on the same node.",
        )
        parser.add_argument(
            "--modelexpress-config",
            type=str,
            default=ServerArgs.modelexpress_config,
            help='JSON config for ModelExpress P2P weight loading. Keys: "url" (required, gRPC host:port), "model_name" (optional, defaults to --model-path), "source" (optional bool, true for seed mode). Example: \'{"url": "localhost:8001", "model_name": "my-model", "source": true}\'',
        )
