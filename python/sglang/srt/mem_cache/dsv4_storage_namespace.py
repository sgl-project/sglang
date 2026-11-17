"""Versioned compatibility domains for native DSV4 shared persistent pages.

The operator supplies immutable checkpoint and compute-recipe identities. Pool
formats and resolved runtime settings are derived, never inferred from names.
The explicit compute recipe also identifies any external speculative draft weights.
"""

import hashlib
import json
import re

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_spec

_SCHEMA = "sglang-dsv4-persistent-v2"
_TAG_SUFFIX = re.compile(r"(?:^|-)dsv4-v2-[0-9a-f]{64}$")


def validate_dsv4_storage_namespace(extra_config):
    identity = (extra_config or {}).get("dsv4_cache_namespace")
    fields = ("checkpoint_id", "compute_recipe")
    if not isinstance(identity, dict) or any(
        not isinstance(identity.get(name), str) or not identity[name].strip()
        for name in fields
    ):
        raise ValueError(
            "Native DeepSeek-V4 Mooncake storage requires dsv4_cache_namespace "
            "with non-empty immutable checkpoint_id and compute_recipe strings."
        )
    return {name: identity[name] for name in fields}


def validate_dsv4_mooncake_transport(extra_config):
    if (
        extra_config.get("protocol") != "tcp"
        or not isinstance(extra_config.get("master_server_address"), str)
        or not extra_config["master_server_address"].strip()
        or extra_config.get("standalone_storage", False)
    ):
        raise ValueError(
            "Native DSV4 Mooncake L3 currently requires explicit protocol=tcp, "
            "master_server_address and standalone_storage=False. RDMA and "
            "borrowed PD transfer engines require separate validation."
        )


def dsv4_storage_namespace_descriptor(pool, page_size, extra_config):
    identity = validate_dsv4_storage_namespace(extra_config)
    pools = []
    for info in pool.get_device_pool_infos():
        if info.pool_name == PoolName.SWA or info.indices_from_pool != PoolName.KV:
            continue
        region = info.buffer_info
        pools.append(
            dict(
                name=str(info.pool_name),
                layers=list(info.layer_ids),
                indices_from=str(info.indices_from_pool),
                hit_policy=PoolHitPolicy.ALL_PAGES.value,
                layout=str(region.layout),
                page_slots=int(region.page_slots),
                page_bytes=int(region.page_bytes),
                buffers=[
                    dict(
                        dtype=str(t.dtype),
                        shape=list(t.shape[1:]),
                        stride=list(t.stride()),
                    )
                    for t in region.buffers
                ],
            )
        )
    if not pools or len({p["name"] for p in pools}) != len(pools):
        raise ValueError(
            "Native DSV4 Mooncake namespace needs unique persistent pools."
        )
    execution = get_exec()
    parallel, kernel, model = get_parallel(), execution.kernel, get_model()
    features, moe = execution.features, execution.moe
    descriptor = dict(
        schema=_SCHEMA,
        identity=identity,
        logical_page_size=int(page_size),
        main_layout=pool.kv_layout.value,
        compression_ratios=list(pool.compression_ratios),
        kv_source_layers=list(pool.kv_source_layers),
        persistent_pools=sorted(pools, key=lambda item: item["name"]),
        topology=dict(
            tp=parallel.tp_size,
            attention_tp=parallel.attn_tp_size,
            attention_dp=parallel.attn_dp_size,
            ep=parallel.ep_size,
            pp=parallel.pp_size,
            cp=parallel.attn_cp_size,
            dcp=parallel.attn_dcp_size,
            dcp_enabled=parallel.dcp_enabled,
            persistent_mapping="compressed-mla-rank-replicated-pp-key-v1",
        ),
        compute=dict(
            attention=kernel.attention_backend,
            prefill_attention=kernel.prefill_attention_backend,
            decode_attention=kernel.decode_attention_backend,
            dsv4_attention=kernel.dsv4_attn_backend,
            dtype=str(model.dtype),
            kv_dtype=str(model.kv_cache_dtype),
            quantization=model.quantization,
            enable_decoder_swa_bounded_replay=features.enable_decoder_swa_bounded_replay,
            enable_encoder_swa_bounded_replay=features.enable_encoder_swa_bounded_replay,
            moe_runner_backend=moe.moe_runner_backend,
            moe_a2a_backend=moe.moe_a2a_backend,
            flashinfer_mxfp4_moe_precision=moe.flashinfer_mxfp4_moe_precision,
            enable_w4a4_mxfp4_megamoe=moe.enable_w4a4_mxfp4_megamoe,
        ),
    )
    spec = get_spec()
    if spec.speculative_algorithm is not None:
        descriptor["compute"].update(
            speculative_algorithm=str(spec.speculative_algorithm).upper(),
            speculative_num_draft_tokens=spec.speculative_num_draft_tokens,
        )
    return descriptor


def dsv4_mooncake_extra_config(pool, page_size, extra_config):
    """Return a canonical namespace; repeated preparation does not append tags."""
    validate_dsv4_mooncake_transport(extra_config)
    descriptor = dsv4_storage_namespace_descriptor(pool, page_size, extra_config)
    digest = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    extra = dict(extra_config)
    tag = extra.get("extra_backend_tag")
    # Already-resolved admin config can pass this entry again. Only remove this
    # schema's exact generated suffix; the operator's separate domain survives.
    tag = _TAG_SUFFIX.sub("", str(tag)) if tag is not None else ""
    extra["extra_backend_tag"] = f"{tag + '-' if tag else ''}dsv4-v2-{digest}"
    extra["dsv4_cache_namespace"] = dict(descriptor["identity"])
    extra["dsv4_cache_namespace_descriptor"] = descriptor
    extra["dsv4_cache_namespace_digest"] = digest
    return extra
