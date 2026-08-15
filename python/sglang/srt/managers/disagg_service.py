"""Start bootstrap/kv-store-related server"""

import os

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
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        maybe_create_ascend_config_store(
            server_args=server_args, transfer_backend=transfer_backend
        )

        return bootstrap_server


def maybe_create_ascend_config_store(
    server_args: ServerArgs, transfer_backend: TransferBackend
) -> None:
    """Also called directly by the rust-server scheduler: there the KV
    bootstrap registry is served by the embedded rust server's api listener
    (one rust implementation covers every transfer backend — their
    bootstrap-server subclasses are all plain ``CommonKVBootstrapServer``,
    which the rust registry ports verbatim), leaving this store as the only
    ``start_disagg_service`` duty left to perform."""
    if not (server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND):
        return
    try:
        from memfabric_hybrid import create_config_store

        ascend_url = os.getenv("ASCEND_MF_STORE_URL")
        create_config_store(ascend_url)
    except Exception as e:
        raise RuntimeError(
            f"Failed create mf store, invalid ascend_url. With exception {e}"
        )
