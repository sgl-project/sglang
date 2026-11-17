"""Keep ordinary DFlash2's cached SWA prefix and its bidirectional block distinct."""

import logging

logger = logging.getLogger(__name__)


def requires_anchor_window(model):
    if type(model).__name__ != "DFlash2DraftModel":
        return False
    config = model.config
    cfg = config.dflash_config
    window = cfg.get("swa_window_size")
    windows = cfg.get("draft_layer_context_windows")
    if window is None and windows is None:
        return False
    if (
        window != 4096
        or windows not in (None, [4096] * config.num_hidden_layers)
        or cfg.get("block_size") != 8
        or config.block_size != 8
        or cfg.get("block_attention") != "bidirectional"
        or cfg.get("context_kv_boundary") != "strictly_before_anchor"
        or cfg.get("mtp_prefix", False)
        or config.layer_types != ["full_attention"] * config.num_hidden_layers
        or getattr(config, "sliding_window", None) not in (None, -1)
        or getattr(config, "use_sliding_window", False)
    ):
        raise ValueError(
            "Ordinary DFlash2 anchor window requires uniform SWA4096, "
            "A+7MASK, bidirectional block and full-attention backend layers"
        )
    return True


def install_anchor_window(worker):
    if not requires_anchor_window(worker.draft_model):
        return
    from sglang.srt.runtime_context import get_schedule
    from sglang.srt.speculative.mtp_anchor_fa3 import AnchorFA3Forward

    runner = worker.draft_model_runner
    if (
        worker.use_compact_draft_cache
        or worker.block_size != 8
        or not 1 <= get_schedule().max_running_requests <= 32
        or runner.runner_ids.is_translating
    ):
        raise ValueError(
            "DFlash2 anchor pages require static shared mapping and B1..32"
        )
    backend = runner.attn_backend
    if getattr(backend, "fa_impl_ver", None) != 3:
        raise ValueError("Ordinary DFlash2 anchor-relative SWA requires FA3")
    if hasattr(backend, "dflash2_anchor_window"):
        raise RuntimeError("DFlash2 anchor wrapper installed twice")
    backend.dflash2_anchor_window = AnchorFA3Forward(backend)
    backend.forward_extend = backend.dflash2_anchor_window


def validate_anchor_capture(worker):
    backend = worker.draft_model_runner.attn_backend
    adapter = getattr(backend, "dflash2_anchor_window", None)
    if adapter is None:
        return
    graph = worker.draft_model_runner.decode_cuda_graph_runner is not None
    if graph and adapter.capture_calls < 1:
        raise RuntimeError("Ordinary DFlash2 anchor window was not captured")
    logger.info(
        "DFLASH2_ANCHOR_WINDOW_CAPTURE window=4096 slots=8 calls=%d graph=%s",
        adapter.capture_calls,
        graph,
    )
