"""Shared fixture plumbing for layer-level backend parity UTs.

Hand-written quantization references (oracle side) live in quant_ref_utils.
"""

import os

import torch


def init_single_process_dist(master_port: int = 29632, backend: str = "gloo"):
    """world=1 dist + model-parallel groups; srt layers require them even
    at tp=1."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend=backend
        )
    if not model_parallel_is_initialized():
        # kwargs only: a positional backend would land in the
        # attention_data_parallel_size slot and explode on int // str.
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend=backend,
        )


def make_tp1_column_parallel_linear(
    quant_config, n: int, k: int, prefix: str = "model.layers.0.mlp.up_proj", **kwargs
):
    from sglang.srt.layers.linear import ColumnParallelLinear

    return ColumnParallelLinear(
        input_size=k,
        output_size=n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
        tp_rank=0,
        tp_size=1,
        **kwargs,
    ).cuda()


def load_linear_weights(layer, shard_id=None, **named_weights):
    """Feed checkpoint-format tensors through the real weight_loader."""
    for name, loaded in named_weights.items():
        if shard_id is None:
            layer.weight_loader_v2(getattr(layer, name), loaded)
        else:
            layer.weight_loader_v2(getattr(layer, name), loaded, shard_id)


def assert_output_close(tc, out, ref, cos_threshold=0.99, rtol=None, atol=None):
    tc.assertEqual(tuple(out.shape), tuple(ref.shape))
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0
    ).item()
    tc.assertGreater(cos, cos_threshold)
    if rtol is not None:
        torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)
