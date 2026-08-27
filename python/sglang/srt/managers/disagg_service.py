"""Start bootstrap/kv-store-related server"""

import os

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_serving,
)


def start_disagg_service():
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(get_disagg().disaggregation_mode)
    transfer_backend = TransferBackend(get_disagg().disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=get_serving().host,
            port=get_disagg().disaggregation_bootstrap_port,
        )
        maybe_create_ascend_config_store(transfer_backend=transfer_backend)

        return bootstrap_server


def maybe_create_ascend_config_store(transfer_backend: TransferBackend) -> None:
    """Also called directly by the rust-server scheduler: there the KV
    bootstrap registry is served by the embedded rust server's api listener
    (one rust implementation covers every transfer backend — their
    bootstrap-server subclasses are all plain ``CommonKVBootstrapServer``,
    which the rust registry ports verbatim), leaving this store as the only
    ``start_disagg_service`` duty left to perform."""
    if not (
        get_parallel().config.node_rank == 0
        and transfer_backend == TransferBackend.ASCEND
    ):
        return
    try:
        from memfabric_hybrid import create_config_store

        ascend_url = os.getenv("ASCEND_MF_STORE_URL")
        create_config_store(ascend_url)
    except Exception as e:
        raise RuntimeError(
            f"Failed create mf store, invalid ascend_url. With exception {e}"
        )
