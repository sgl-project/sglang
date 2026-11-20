"""Bind the rank-local model after quantization and weight postprocessing."""

from functools import cache


@cache
def _init_shemm(rank, size):
    from blockrt.dist.utils import init_shemm

    init_shemm(rank=rank, world_size=size)


def build_runtime(model, factory, *, draft):
    from blockrt.models.glm_5_2.model_args import ModelArgsGLM5
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    _init_shemm(parallel.tp_rank, parallel.tp_size)
    aliases = {
        "ffn_hidden_size": "intermediate_size",
        "max_seq_len": "max_position_embeddings",
        "moe_ffn_hidden_size": "moe_intermediate_size",
        "n_layers": "num_hidden_layers",
        "num_experts": "n_routed_experts",
        "num_heads": "num_attention_heads",
        "num_kv_heads": "num_key_value_heads",
        "num_layers": "num_hidden_layers",
        "top_k": "num_experts_per_tok",
    }
    args = ModelArgsGLM5()
    for name in type(args).__annotations__:
        if name == "index_share_for_mtp_iteration":
            continue
        value = getattr(model.config, aliases.get(name, name), None)
        if value is not None:
            setattr(args, name, value)
    if not draft:
        args.num_nextn_predict_layers = 0
    weights = {
        name: tensor.detach()
        for name, tensor in model.state_dict(keep_vars=True).items()
    }
    if draft and getattr(model.model, "rot_weight", None) is not None:
        weights["rot.weight"] = model.model.rot_weight.detach()
    runtime = factory(args, weights, tp_size=parallel.tp_size, tp_rank=parallel.tp_rank)
    inner = runtime.model if draft else runtime.model.model
    if (
        inner.embed_tokens.weight.data_ptr()
        != model.model.embed_tokens.weight.data_ptr()
    ):
        raise RuntimeError("BlockRT duplicated the shared GLM embedding")
    return runtime
