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
        _maybe_create_ascend_config_store(
            server_args=server_args, transfer_backend=transfer_backend
        )

        return bootstrap_server


def start_rust_disagg_service(server_args: ServerArgs):
    """``start_disagg_service`` for the embedded-rust-server scheduler: the KV
    bootstrap registry is served by the rust extension on its own native
    thread (the aiohttp server above would run inside the scheduler process
    and contend for its GIL). One rust implementation serves every transfer
    backend — their bootstrap-server subclasses are all plain
    ``CommonKVBootstrapServer``, which the rust registry ports verbatim.

    Returns ``None`` on non-prefill roles. The returned handle must be kept
    referenced: dropping it stops the server.
    """
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    if disagg_mode != DisaggregationMode.PREFILL:
        return None

    # Lazy: the compiled extension only exists in rust-server builds.
    from sglang.srt.server._core import BootstrapServer

    bootstrap_server = BootstrapServer(
        host=server_args.host,
        port=server_args.disaggregation_bootstrap_port,
    )
    _maybe_create_ascend_config_store(
        server_args=server_args,
        transfer_backend=TransferBackend(server_args.disaggregation_transfer_backend),
    )
    return bootstrap_server


def _maybe_create_ascend_config_store(
    server_args: ServerArgs, transfer_backend: TransferBackend
) -> None:
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
