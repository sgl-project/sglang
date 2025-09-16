"""Start bootstrap/kv-store-related server"""

import os
from typing import Type

from sglang.srt.disaggregation.base import BaseKVBootstrapServer
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs


def start_disagg_service(
    server_args: ServerArgs,
):
    # Start kv boostrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class: Type[BaseKVBootstrapServer] = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server: BaseKVBootstrapServer = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        is_create_store = (
            server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND
        )
        if is_create_store:
            try:
                from mf_adapter import create_config_store

                ascend_url = os.getenv("ASCEND_MF_STORE_URL")
                create_config_store(ascend_url)
            except Exception as e:
                error_message = f"Failed create mf store, invalid ascend_url."
                error_message += f" With exception {e}"
                raise error_message

        return bootstrap_server
