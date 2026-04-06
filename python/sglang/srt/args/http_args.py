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
"""CLI argument definitions for HTTP server and SSL/TLS."""

import argparse


class HttpArgs:
    """CLI argument definitions for HTTP server and SSL/TLS."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs

        # HTTP server
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="The host of the HTTP server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="The port of the HTTP server.",
        )
        parser.add_argument(
            "--fastapi-root-path",
            type=str,
            default=ServerArgs.fastapi_root_path,
            help="App is behind a path based routing proxy.",
        )
        parser.add_argument(
            "--grpc-mode",
            action="store_true",
            help="If set, use gRPC server instead of HTTP server.",
        )
        parser.add_argument(
            "--skip-server-warmup",
            action="store_true",
            help="If set, skip warmup.",
        )
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )
        parser.add_argument(
            "--nccl-port",
            type=int,
            default=ServerArgs.nccl_port,
            help="The port for NCCL distributed environment setup. Defaults to a random port.",
        )
        parser.add_argument(
            "--checkpoint-engine-wait-weights-before-ready",
            action="store_true",
            help="If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods "
            "before serving inference requests.",
        )

        # SSL/TLS
        parser.add_argument(
            "--ssl-keyfile",
            type=str,
            default=ServerArgs.ssl_keyfile,
            help="The file path to the SSL key file.",
        )
        parser.add_argument(
            "--ssl-certfile",
            type=str,
            default=ServerArgs.ssl_certfile,
            help="The file path to the SSL certificate file.",
        )
        parser.add_argument(
            "--ssl-ca-certs",
            type=str,
            default=ServerArgs.ssl_ca_certs,
            help="The CA certificates file.",
        )
        parser.add_argument(
            "--ssl-keyfile-password",
            type=str,
            default=ServerArgs.ssl_keyfile_password,
            help="The password to decrypt the SSL keyfile.",
        )
        parser.add_argument(
            "--enable-ssl-refresh",
            action="store_true",
            default=ServerArgs.enable_ssl_refresh,
            help="Enable automatic SSL certificate hot-reloading when cert/key "
            "files change on disk. Requires --ssl-certfile and --ssl-keyfile.",
        )
