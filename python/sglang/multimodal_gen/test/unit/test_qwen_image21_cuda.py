# SPDX-License-Identifier: Apache-2.0
"""Request-scoped prefix KV and graph replay regression tests; no checkpoint needed."""

from copy import deepcopy

import pytest
import torch
from diffusers.models.normalization import RMSNorm as ReferenceRMSNorm
from safetensors.torch import save_file

from sglang.kernels.ops.diffusion import BitExactFusionGate
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
from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as model_module
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    QwenImage21Transformer2DModel,
    build_layout,
)
from sglang.multimodal_gen.runtime.pipelines.qwen_image21 import QwenImage21Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    set_global_server_args,
)
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


@pytest.fixture
def bf16_model(model):
    # parallel modules own process groups and cannot be deep-copied
    config = QwenImage21DitConfig(arch_config=model.config)
    result = QwenImage21Transformer2DModel(config, {}).cuda().bfloat16().eval()
    result.load_state_dict(model.state_dict())
    return result


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


@pytest.mark.parametrize("merge_mode", ["dynamic", "merge"])
@torch.no_grad()
def test_diffusers_lora_matches_weight_delta_and_restores_base(
    model, tmp_path, monkeypatch, merge_mode
):
    # Reuse loaded native components, then exercise the real adapter loader.
    monkeypatch.setattr(ComposedPipelineBase, "__init__", lambda self: None)
    pipeline = object.__new__(QwenImage21Pipeline)
    config = QwenImage21DitConfig(arch_config=model.config)
    pipeline.server_args = ServerArgs(
        model_path="Qwen/Qwen-Image-2.1",
        num_gpus=1,
        pipeline_config=QwenImage21PipelineConfig(dit_config=config),
        attention_backend="torch_sdpa",
    )
    set_global_server_args(pipeline.server_args)
    actual_model = QwenImage21Transformer2DModel(config, {}).cuda().eval()
    reference = QwenImage21Transformer2DModel(config, {}).cuda().eval()
    for loaded in (actual_model, reference):
        loaded.load_state_dict(model.state_dict())
    pipeline.modules = {"transformer": actual_model}
    pipeline.__init__()
    weights = {}
    for name in ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mlp.out"):
        layer = reference.get_submodule(name)
        a = torch.randn(2, layer.weight.shape[1], device="cuda") * 0.2
        b = torch.randn(layer.weight.shape[0], 2, device="cuda") * 0.2
        weights[f"transformer.{name}.lora_A.weight"] = a.cpu()
        weights[f"transformer.{name}.lora_B.weight"] = b.cpu()
        layer.weight.add_(b @ a)
    adapter = tmp_path / "adapter.safetensors"
    save_file(weights, str(adapter))
    kwargs = dict(inputs(5, False), prefix_caches=None)
    with set_forward_context(None, None):
        baseline = actual_model(**kwargs)
        expected = reference(**kwargs)
        pipeline.set_lora(
            "test", str(adapter), target="transformer", merge_mode=merge_mode
        )
        assert pipeline.is_lora_effective("transformer")
        actual = actual_model(**kwargs)
        assert not torch.equal(actual, baseline)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        pipeline.unmerge_lora_weights("transformer")
        torch.testing.assert_close(actual_model(**kwargs), baseline, atol=0, rtol=0)


@pytest.mark.parametrize("edit", [False, True])
def test_cached_prefix_matches_full_recomputation(model, edit):
    kwargs = inputs(5, edit)
    with torch.no_grad(), set_forward_context(None, None):
        model(**kwargs)
        prefix_length = kwargs["layouts"][0]["prefix_rope"].shape[0]
        for cache in kwargs["prefix_caches"][0]:
            for tensor in cache.values():
                assert tensor.shape[1] == prefix_length
                assert (
                    tensor.untyped_storage().nbytes()
                    == tensor.numel() * tensor.element_size()
                )
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


@pytest.mark.parametrize("edit", [False, True])
@torch.no_grad()
def test_bf16_fusions_match_eager_prefill_and_cached_steps(
    bf16_model, edit, monkeypatch
):
    actual_model = bf16_model
    kwargs = inputs(5, edit)
    for key in (
        "hidden_states",
        "encoder_hidden_states",
        "condition_latents",
        "timestep",
    ):
        if kwargs[key] is not None:
            kwargs[key] = kwargs[key].bfloat16()
    reference_kwargs = deepcopy(kwargs)
    expected = []
    disabled = BitExactFusionGate("reference")
    disabled.disable()
    with monkeypatch.context() as reference, set_forward_context(None, None):
        reference.setattr(model_module, "_SILU_MUL_FUSION", disabled)
        reference.setattr(
            model_module,
            "residual_gate_add",
            lambda residual, update, gate: residual + gate * update,
        )
        for timestep in (700, 300, 10):
            reference_kwargs["timestep"].fill_(timestep)
            expected.append(actual_model(**reference_kwargs))

    gate = BitExactFusionGate("test SiLU-mul")
    monkeypatch.setattr(model_module, "_SILU_MUL_FUSION", gate)
    with set_forward_context(None, None):
        for timestep, output in zip((700, 300, 10), expected, strict=True):
            kwargs["timestep"].fill_(timestep)
            torch.testing.assert_close(actual_model(**kwargs), output, atol=0, rtol=0)
    assert gate.verified and not gate.disabled
    for actual, reference in zip(
        kwargs["prefix_caches"][0], reference_kwargs["prefix_caches"][0], strict=True
    ):
        for key in ("key", "value"):
            torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)

    runner = DiffusionBreakableCudaGraphRunner(actual_model, torch.device("cuda"))
    try:
        with set_forward_context(None, None):
            assert runner.capture(**kwargs)
            kwargs["hidden_states"].add_(0.1)
            expected = actual_model(**kwargs)
            torch.testing.assert_close(runner(**kwargs), expected, atol=0, rtol=0)
    finally:
        runner.reset()


@torch.no_grad()
def test_silu_fusion_mismatch_restores_eager(bf16_model, monkeypatch):
    mlp = bf16_model.transformer_blocks[0].img_mlp
    x = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    gate = BitExactFusionGate("test mismatch")
    monkeypatch.setattr(model_module, "_SILU_MUL_FUSION", gate)
    monkeypatch.setattr(
        model_module, "fused_silu_mul_bitexact", lambda a, b: torch.zeros_like(a)
    )
    with set_forward_context(None, None):
        expected = mlp.out(
            torch.nn.functional.silu(mlp.gate_layer(x)[0]) * mlp.proj(x)[0]
        )[0]
        torch.testing.assert_close(mlp(x), expected, atol=0, rtol=0)
        assert gate.disabled and not gate.verified
        torch.testing.assert_close(mlp(x), expected, atol=0, rtol=0)
