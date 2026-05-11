from __future__ import annotations


def get_draft_kv_pool(
    *,
    draft_worker,
    spec_algorithm,
    server_args,
    enable_overlap: bool,
):
    """Return (draft_token_to_kv_pool, draft_model_config) for the current
    draft worker, or (None, None) when no draft KV pool is available."""
    if draft_worker is None or spec_algorithm.is_ngram():
        return None, None

    if spec_algorithm.supports_spec_v2() and enable_overlap:
        if server_args.enable_multi_layer_eagle:
            draft_runner = draft_worker.draft_worker.draft_runner_list[0]
        else:
            draft_runner = draft_worker.draft_worker.draft_runner
        return draft_runner.token_to_kv_pool, draft_runner.model_config

    return (
        draft_worker.model_runner.token_to_kv_pool,
        draft_worker.model_config,
    )
