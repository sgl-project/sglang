# SPDX-License-Identifier: Apache-2.0
"""Request-scoped prefix KV and graph replay regression tests; no checkpoint needed."""

from copy import deepcopy

import pytest
import torch
from diffusers.models.normalization import RMSNorm as ReferenceRMSNorm

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    DiffusionBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    QwenImage21Transformer2DModel,
    build_layout,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def model():
    config = QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(
            in_channels=4,
            out_channels=4,
            num_layers=3,
            num_attention_heads=4,
            attention_head_dim=32,
            context_in_dim=16,
            mlp_ratio=2,
            axes_dims_rope=(8, 12, 12),
        )
    )
    args = ServerArgs(
        model_path="Qwen/Qwen-Image-2.1",
        num_gpus=1,
        pipeline_config=QwenImage21PipelineConfig(dit_config=config),
        attention_backend="torch_sdpa",
    )
    set_global_server_args(args)
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.manual_seed(42)
    model = QwenImage21Transformer2DModel(config, {}).cuda().eval()
    # Parallel linear layers allocate empty weights for checkpoint loading.
    for name, param in model.named_parameters():
        torch.nn.init.normal_(param, std=0.02)
        if name.endswith(("norm_q.weight", "norm_k.weight")):
            torch.nn.init.ones_(param)
    return model


def inputs(seed, edit):
    torch.manual_seed(seed)
    slots = [False] * 3 + ([True, False, False] if edit else [])
    shapes = ([(1, 2, 4)] if edit else []) + [(1, 4, 4)]
    return dict(
        hidden_states=torch.randn(1, 16, 4, device="cuda"),
        encoder_hidden_states=torch.randn(1, len(slots), 16, device="cuda"),
        condition_latents=torch.randn(1, 8, 4, device="cuda") if edit else None,
        layouts=[build_layout(slots, shapes, (8, 12, 12), "cuda")],
        prefix_caches=[[{} for _ in range(3)]],
        timestep=torch.tensor([700.0], device="cuda"),
    )


def test_bf16_qk_norm_matches_reference(model):
    norm = deepcopy(model.transformer_blocks[0].attn.norm_q).bfloat16()
    reference = ReferenceRMSNorm(32, eps=1e-6).cuda().bfloat16()
    weight = torch.linspace(0.3, 1.7, 32, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 8, 4, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
        reference.weight.copy_(weight)
        torch.testing.assert_close(norm(x), reference(x), atol=0, rtol=0)


@pytest.mark.parametrize("edit", [False, True])
def test_cached_prefix_matches_full_recomputation(model, edit):
    kwargs = inputs(5, edit)
    with torch.no_grad(), set_forward_context(None, None):
        model(**kwargs)
        keys = [layer["key"].clone() for layer in kwargs["prefix_caches"][0]]
        kwargs["timestep"].fill_(300.0)
        actual = model(**kwargs)
        expected = model(**dict(kwargs, prefix_caches=None))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, cache in zip(keys, kwargs["prefix_caches"][0], strict=True):
        torch.testing.assert_close(key, cache["key"], atol=0, rtol=0)


@pytest.mark.parametrize("edit", [False, True])
def test_graph_replay_uses_new_request_prefix(model, edit):
    first, second = inputs(5, edit), inputs(9, edit)
    runner = DiffusionBreakableCudaGraphRunner(model, torch.device("cuda"))
    try:
        with torch.no_grad(), set_forward_context(None, None):
            model(**first)
            assert runner.capture(**first)
            model(**second)
            expected = model(**second)
            actual = runner(**second)
        assert len(runner.entries) == 1
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    finally:
        runner.reset()
