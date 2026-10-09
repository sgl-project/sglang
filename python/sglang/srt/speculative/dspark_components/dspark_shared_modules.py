"""Make native target embedding/head shards available to every PP draft replica."""

from dataclasses import asdict

import torch
from sglang.srt.distributed import get_world_group
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

_DTYPES = {
    str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32)
}


def _embedding(target_model):
    if hasattr(target_model, "get_input_embeddings"):
        return target_model.get_input_embeddings()
    return target_model.model.get_input_embeddings()


def _describe(module, *, tp_size, tp_rank):
    if type(module) not in (VocabParallelEmbedding, ParallelLMHead):
        raise ValueError("PP DSpark requires native vocabulary embedding/head modules")
    if type(module.quant_method) is not UnquantizedEmbeddingMethod:
        raise ValueError("PP DSpark requires unquantized embedding/head weights")
    weight = module.weight
    if (
        str(weight.dtype) not in _DTYPES
        or weight.ndim != 2
        or not weight.is_contiguous()
        or tuple(weight.shape)
        != (module.num_embeddings_per_partition, module.embedding_dim)
        or module.tp_size != tp_size
        or getattr(module, "bias", None) is not None
    ):
        raise ValueError("PP DSpark embedding/head weight layout is unsupported")
    expected_shard = module._get_indices(
        module.num_embeddings_padded,
        module.org_vocab_size_padded,
        module.num_embeddings,
        module.org_vocab_size,
        tp_rank,
        tp_size,
    )
    if module.shard_indices != expected_shard:
        raise ValueError("PP DSpark source module belongs to a different TP rank")
    return {
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
        "org_num_embeddings": module.org_vocab_size,
        "padding_size": module.padding_size,
        "enable_tp": module.enable_tp,
        "use_attn_tp_group": module.use_attn_tp_group,
        "use_presharded_weights": module.use_presharded_weights,
        "dtype": str(weight.dtype),
        "shape": tuple(weight.shape),
        "shard": asdict(module.shard_indices),
    }


def _check_errors(records):
    errors = [
        f"rank {index}: {record['error']}"
        for index, record in enumerate(records)
        if record["error"] is not None
    ]
    if errors:
        raise RuntimeError("PP DSpark shared modules: " + "; ".join(errors))


def _make_replica(description, *, head, device):
    options = dict(description)
    dtype = _DTYPES[options.pop("dtype")]
    options.pop("shape")
    options.pop("shard")
    cls = ParallelLMHead if head else VocabParallelEmbedding
    with torch.device(device):
        return cls(**options, params_dtype=dtype)


def resolve_dspark_shared_modules(*, target_model, pp_group, tp_group, device):
    """Borrow owner modules and replicate their TP shards on other PP stages.

    Every DP=1 rank must enter this startup operation. The source modules stay
    attached to their target stages; only the draft receives the missing copies.
    Target weight replacement requires rebuilding these replicas.
    """
    if pp_group.world_size == 1:
        head = getattr(target_model, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )
        return _embedding(target_model), head

    world = get_world_group()
    stage = pp_group.rank_in_group
    tensor_rank = tp_group.rank_in_group
    local = {"stage": stage, "tp_rank": tensor_rank, "error": None}
    embedding = head = None
    try:
        if world.world_size != pp_group.world_size * tp_group.world_size:
            raise ValueError("PP DSpark shared modules require DP=1")
        if stage == 0:
            embedding = _embedding(target_model)
            local["embedding"] = _describe(
                embedding, tp_size=tp_group.world_size, tp_rank=tensor_rank
            )
            if embedding.weight.device != torch.device(device):
                raise ValueError("embedding weight is on the wrong device")
        if stage == pp_group.world_size - 1:
            head = target_model.lm_head
            local["head"] = _describe(
                head, tp_size=tp_group.world_size, tp_rank=tensor_rank
            )
            if head.weight.device != torch.device(device):
                raise ValueError("head weight is on the wrong device")
    except Exception as error:  # noqa: BLE001 - All ranks must learn startup failure.
        local["error"] = f"{type(error).__name__}: {error}"
    records = world.all_gather_object(local)
    _check_errors(records)
    owners = {(record["stage"], record["tp_rank"]): record for record in records}
    expected = {
        (pp_rank, tp_rank)
        for pp_rank in range(pp_group.world_size)
        for tp_rank in range(tp_group.world_size)
    }
    if len(owners) != len(records) or set(owners) != expected:
        raise RuntimeError("PP DSpark shared-module rank layout is inconsistent")
    for name, owner in (("embedding", 0), ("head", pp_group.world_size - 1)):
        descriptions = [
            {
                key: value
                for key, value in owners[(owner, rank)][name].items()
                if key != "shard"
            }
            for rank in range(tp_group.world_size)
        ]
        if any(description != descriptions[0] for description in descriptions[1:]):
            raise RuntimeError(f"PP DSpark {name} metadata differs across TP ranks")
    source_embedding = owners[(0, tensor_rank)]["embedding"]
    source_head = owners[(pp_group.world_size - 1, tensor_rank)]["head"]

    # Fence allocation/layout errors across the entire world before any device
    # collective; otherwise another TP lane could wait for a failed rank.
    error = None
    try:
        if embedding is None:
            embedding = _make_replica(source_embedding, head=False, device=device)
        if head is None:
            head = _make_replica(source_head, head=True, device=device)
        for module, source in ((embedding, source_embedding), (head, source_head)):
            if (
                _describe(module, tp_size=tp_group.world_size, tp_rank=tensor_rank)
                != source
            ):
                raise ValueError("replica differs from its owner's TP shard layout")
        if (
            embedding.embedding_dim != head.embedding_dim
            or embedding.org_vocab_size != head.org_vocab_size
        ):
            raise ValueError("target embedding/head dimensions disagree")
    except Exception as failure:  # noqa: BLE001 - Fence replica allocation failures.
        error = f"{type(failure).__name__}: {failure}"
    _check_errors(world.all_gather_object({"error": error}))
    pp_group.broadcast(embedding.weight.detach(), src=0)
    pp_group.broadcast(head.weight.detach(), src=pp_group.world_size - 1)
    return embedding, head
