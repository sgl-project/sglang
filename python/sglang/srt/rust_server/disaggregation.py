"""Bootstrap protocol selection and launch-process ownership for Rust serving."""

from contextlib import contextmanager

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.disaggregation import utils as disaggregation_utils
from sglang.srt.environ import envs


def _bootstrap_class(server_args):
    cfg = resolving_view(server_args)
    return disaggregation_utils.get_kv_class(
        disaggregation_utils.TransferBackend(cfg.disaggregation_transfer_backend),
        disaggregation_utils.KVClassType.BOOTSTRAP_SERVER,
    )


def http_bootstrap_port(server_args) -> int | None:
    cfg = resolving_view(server_args)
    if cfg.disaggregation_mode != "prefill":
        return None
    if _bootstrap_class(server_args).bootstrap_protocol != "http":
        return None
    return cfg.disaggregation_bootstrap_port


@contextmanager
def bootstrap_service(server_args):
    """Start non-HTTP bootstrap before any scheduler can register with it."""
    cfg = resolving_view(server_args)
    if (
        not envs.SGLANG_RUST_SERVER.get()
        or cfg.disaggregation_mode != "prefill"
        or cfg.node_rank != 0
    ):
        yield
        return
    bootstrap_class = _bootstrap_class(server_args)
    if bootstrap_class.bootstrap_protocol == "http":
        yield
        return
    if cfg.disaggregation_bootstrap_port == cfg.port:
        raise ValueError(
            f"The {bootstrap_class.bootstrap_protocol} bootstrap service needs a "
            "different --disaggregation-bootstrap-port from the HTTP --port"
        )
    server = bootstrap_class(cfg.host, cfg.disaggregation_bootstrap_port)
    try:
        yield server
    finally:
        server.close()
