# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage import (
    QwenImageDitConfig,
    QwenImageEditPlus_2511_DitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPlus_2511_PipelineConfig,
    QwenImageEditPlusPipelineConfig,
    QwenImageLayeredPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import denoising
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

_DEVICE = torch.device("cuda")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the tiny DiT runs fused CUDA kernels"
)
# 32x32 pixels pack into 2x2 latent tokens; the edit reference into 3x2
HEIGHT = WIDTH = 32
REF_SIZE = (32, 48)
CHANNELS = 8
TEXT_DIM = 12


def _tiny_dit(dit_config):
    for key, value in dict(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=2,
        joint_attention_dim=TEXT_DIM,
    ).items():
        setattr(dit_config.arch_config, key, value)
    model = QwenImageTransformer2DModel(dit_config, {})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    return model.to(device=_DEVICE, dtype=torch.bfloat16)


def _request(
    cond_len,
    uncond_len,
    edit,
    dtype,
    quality=None,
    skip_softmax=None,
):
    generator = torch.Generator().manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(_DEVICE, dtype)

    noise_tokens = (HEIGHT // 16) * (WIDTH // 16)
    batch = SimpleNamespace(
        height=HEIGHT,
        width=WIDTH,
        latents=randn(1, noise_tokens, CHANNELS),
        prompt_embeds=[randn(1, cond_len, TEXT_DIM)],
        negative_prompt_embeds=[randn(1, uncond_len, TEXT_DIM)],
        vae_image_sizes=[REF_SIZE],
        do_classifier_free_guidance=True,
        is_cfg_negative=False,
        true_cfg_scale=None,
        guidance_scale=4.0,
        cfg_normalization=0,
        guidance_rescale=0,
        sampling_params=SimpleNamespace(
            quality=quality, skip_softmax_params=skip_softmax
        ),
        extra={},
    )
    latent_input = batch.latents
    if edit:
        ref_tokens = (REF_SIZE[1] // 16) * (REF_SIZE[0] // 16)
        reference = randn(1, ref_tokens, CHANNELS)
        latent_input = torch.cat([batch.latents, reference], dim=1)
    return batch, latent_input


def _stage(compile_only_offload=None):
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = SimpleNamespace(
        dit_layerwise_offload=compile_only_offload is None,
        is_dit_layerwise_offload_selected=False,
    )
    stage._cache_dit_enabled = False
    stage._extra_func_kwarg_names_cache = {}
    stage._offloaded_dit_modules_for_compile = (
        [] if compile_only_offload is None else [compile_only_offload]
    )
    return stage


@torch.no_grad()
def _denoise_step(
    pipeline_config,
    model,
    cond_len,
    uncond_len,
    *,
    edit,
    streamed,
    stage=None,
    resident_layers=0,
    free_gib=1024.0,
    quality=None,
    skip_softmax=None,
    steps=1,
    reusable_bytes=0,
    autotune=False,
):
    dtype = next(model.parameters()).dtype
    batch, latent_input = _request(
        cond_len, uncond_len, edit, dtype, quality, skip_softmax
    )
    branch_kwargs = []
    for embeds, prepare in (
        (batch.prompt_embeds, pipeline_config.prepare_pos_cond_kwargs),
        (batch.negative_prompt_embeds, pipeline_config.prepare_neg_cond_kwargs),
    ):
        kwargs = prepare(batch, _DEVICE, model.rotary_emb, dtype)
        branch_kwargs.append(kwargs | dict(encoder_hidden_states=embeds))
    policy = CFGPolicy().build(batch, {}, *branch_kwargs)
    stage = stage or _stage()
    stage.server_args.enable_breakable_cuda_graph = False
    widths = []
    forward = model.forward

    def counting_forward(**kwargs):
        widths.append(kwargs["hidden_states"].shape[0])
        return forward(**kwargs)

    server_args = SimpleNamespace(
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        enable_attention_backend_autotune=autotune,
        pipeline_config=pipeline_config,
    )
    free_memory = (
        Mock(side_effect=free_gib)
        if isinstance(free_gib, list)
        else Mock(return_value=free_gib)
    )
    with (
        patch.object(current_platform, "get_available_gpu_memory", free_memory),
        patch.object(torch.cuda, "memory_reserved", return_value=reusable_bytes),
        patch.object(torch.cuda, "memory_allocated", return_value=0),
        patch.object(model, "forward", counting_forward),
        patch.object(
            model,
            "layerwise_offload_managers",
            [
                SimpleNamespace(
                    enabled=streamed,
                    resident_layers=resident_layers,
                    num_layers=len(model.transformer_blocks),
                )
            ],
        ),
    ):
        for _ in range(steps):
            out = stage._predict_noise_with_cfg(
                current_model=model,
                latent_model_input=latent_input,
                timestep=torch.tensor([500.0], device=_DEVICE, dtype=dtype),
                batch=batch,
                timestep_index=0,
                attn_metadata=None,
                target_dtype=dtype,
                current_guidance_scale=4.0,
                cfg_policy=policy,
                cfg_gate_state=None,
                server_args=server_args,
                guidance=None,
                latents=batch.latents,
            )
    return out, widths


CASES = [
    (QwenImagePipelineConfig, QwenImageDitConfig, False),
    (QwenImageEditPlusPipelineConfig, QwenImageDitConfig, True),
    (QwenImageEditPlus_2511_PipelineConfig, QwenImageEditPlus_2511_DitConfig, True),
]


@requires_cuda
@pytest.mark.parametrize("pipeline_cls,dit_cls,edit", CASES)
@pytest.mark.parametrize("cond_len,uncond_len", [(7, 3), (3, 7), (5, 5)])
def test_streamed_dit_runs_both_cfg_branches_in_one_forward(
    single_process_model_parallel, pipeline_cls, dit_cls, edit, cond_len, uncond_len
):
    model = _tiny_dit(dit_cls())
    serial, serial_widths = _denoise_step(
        pipeline_cls(), model, cond_len, uncond_len, edit=edit, streamed=False
    )
    batched, batched_widths = _denoise_step(
        pipeline_cls(), model, cond_len, uncond_len, edit=edit, streamed=True
    )
    assert serial_widths == [1, 1]
    assert batched_widths == [2]
    assert batched.shape == serial.shape
    # bf16 tolerance; a wrong mask or timestep row moves the output far more
    rel_l2 = (batched.float() - serial.float()).norm() / serial.float().norm()
    assert rel_l2 < 0.05, rel_l2


def test_layered_stays_serial():
    kwargs = dict(
        encoder_hidden_states=[torch.zeros(1, 3, TEXT_DIM)],
        encoder_hidden_states_mask=None,
        txt_seq_lens=[3],
        img_shapes=[[(1, 2, 2), (1, 2, 2)]],
        freqs_cis=(torch.zeros(8, 4), torch.zeros(3, 4)),
        additional_t_cond=torch.zeros(1, dtype=torch.long),
    )
    config = QwenImageLayeredPipelineConfig()
    assert config.batch_cfg_branch_kwargs(kwargs, dict(kwargs)) is None


@requires_cuda
def test_dit_offloaded_only_for_compile_warmup_stays_serial(
    single_process_model_parallel,
):
    model = _tiny_dit(QwenImageDitConfig())
    stage = _stage(compile_only_offload=model)
    _, widths = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=True, stage=stage
    )
    assert widths == [1, 1]

    stage.server_args.dit_layerwise_offload = True
    _, widths = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=True, stage=stage
    )
    assert widths == [2]


@requires_cuda
def test_dit_with_every_layer_resident_stays_serial(single_process_model_parallel):
    model = _tiny_dit(QwenImageDitConfig())
    num_layers = len(model.transformer_blocks)
    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        resident_layers=num_layers,
    )
    assert widths == [1, 1]

    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        resident_layers=num_layers - 1,
    )
    assert widths == [2]


@pytest.mark.parametrize(
    "pipeline_cls,latent_tokens,measured_mib",
    [
        # 1024x1024 default request on H200: the extra peak allocated memory of the
        # batched path, which one estimated row must cover (text tokens left out).
        (QwenImagePipelineConfig, 4096, 318),
        (QwenImageEditPlusPipelineConfig, 8192, 636),
        (QwenImageEditPlus_2511_PipelineConfig, 8192, 734),
    ],
)
def test_cfg_row_estimate_covers_measured_extra_memory(
    pipeline_cls, latent_tokens, measured_mib
):
    row = pipeline_cls().batch_cfg_row_bytes(latent_tokens, 0, torch.bfloat16)
    assert row >= measured_mib * 1024**2


@requires_cuda
def test_streamed_dit_without_headroom_runs_cfg_serially(
    single_process_model_parallel,
):
    model = _tiny_dit(QwenImageDitConfig())
    serial, _ = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=False
    )
    out, widths = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=True, free_gib=0.0
    )
    assert widths == [1, 1]
    torch.testing.assert_close(out, serial, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("edit", [False, True])
def test_cfg_row_estimate_gets_the_request_shape(single_process_model_parallel, edit):
    pipeline_cls = QwenImageEditPlusPipelineConfig if edit else QwenImagePipelineConfig
    model = _tiny_dit(QwenImageDitConfig())
    _, latent_input = _request(7, 3, edit, torch.bfloat16)
    calls = []

    def spy(config, *args):
        calls.append(args)
        return 0

    with patch.object(pipeline_cls, "batch_cfg_row_bytes", spy, create=True):
        _denoise_step(pipeline_cls(), model, 7, 3, edit=edit, streamed=True)
    # every latent token, the edit reference included, and the longer prompt
    assert calls == [(latent_input.shape[1], 7, torch.bfloat16)]


@requires_cuda
def test_batched_cfg_needs_two_rows_plus_margin(single_process_model_parallel):
    model = _tiny_dit(QwenImageDitConfig())
    row = 3 * 1024**2
    need = 2 * row + 1024**3  # two rows plus the 1 GiB margin
    with patch.object(
        QwenImagePipelineConfig, "batch_cfg_row_bytes", return_value=row, create=True
    ):
        for free, expected in ((need, [2]), (need - 1, [1, 1])):
            _, widths = _denoise_step(
                QwenImagePipelineConfig(),
                model,
                7,
                3,
                edit=False,
                streamed=True,
                free_gib=free / 1024**3,
            )
            assert widths == expected, (free, widths)


@requires_cuda
def test_tensor_parallel_runs_cfg_serially(single_process_model_parallel):
    model = _tiny_dit(QwenImageDitConfig())
    with patch.object(
        denoising, "get_tp_group", return_value=SimpleNamespace(world_size=2)
    ):
        _, widths = _denoise_step(
            QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=True
        )
    assert widths == [1, 1]


@requires_cuda
def test_exact_quality_runs_cfg_serially(single_process_model_parallel):
    """Batching changes the reduction order, which the bit-exact tier forbids."""
    model = _tiny_dit(QwenImageDitConfig())
    serial, _ = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=False
    )
    out, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        quality="exact",
    )
    assert widths == [1, 1]
    assert torch.equal(out, serial)


@requires_cuda
@pytest.mark.parametrize("cond_len,uncond_len,expected", [(7, 3, [1, 1]), (5, 5, [2])])
def test_skip_softmax_keeps_unequal_prompts_serial(
    single_process_model_parallel, cond_len, uncond_len, expected
):
    """USPAttention refuses Skip Softmax with the padding mask of unequal prompts."""
    model = _tiny_dit(QwenImageDitConfig())
    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        cond_len,
        uncond_len,
        edit=False,
        streamed=True,
        skip_softmax=object(),
    )
    assert widths == expected


@requires_cuda
@pytest.mark.parametrize(
    "backend,varlen_fa",
    [(AttentionBackendEnum.SPARGE_ATTN, True), (AttentionBackendEnum.FA, False)],
)
@pytest.mark.parametrize("cond_len,uncond_len,expected", [(7, 3, [1, 1]), (5, 5, [2])])
def test_mask_unsupported_attention_keeps_unequal_prompts_serial(
    single_process_model_parallel, backend, varlen_fa, cond_len, uncond_len, expected
):
    """A masked batch-2 forward would run dense SDPA instead of the selected kernel."""
    model = _tiny_dit(QwenImageDitConfig())
    attention = [m for m in model.modules() if isinstance(m, USPAttention)]
    assert attention
    # only the enum the decision reads changes; attn_impl stays dense, so the
    # serial forward still runs
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(attention_layer, "_VARLEN_FA_ENABLED", varlen_fa)
        )
        for module in attention:
            stack.enter_context(patch.object(module, "backend", backend))
        _, widths = _denoise_step(
            QwenImagePipelineConfig(),
            model,
            cond_len,
            uncond_len,
            edit=False,
            streamed=True,
        )
    assert widths == expected


@requires_cuda
def test_batched_cfg_rechecks_memory_every_step(single_process_model_parallel):
    """Free memory can shrink between steps, e.g. retained trajectory latents."""
    model = _tiny_dit(QwenImageDitConfig())
    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        free_gib=[1024.0, 0.0, 1024.0],
        steps=3,
    )
    assert widths == [2, 1, 1, 2]


@requires_cuda
@pytest.mark.parametrize("reusable_gib,expected", [(1024, [2]), (0, [1, 1])])
def test_reusable_allocator_cache_counts_as_free(
    single_process_model_parallel, reusable_gib, expected
):
    """The previous step's activations stay reserved; the driver counts them as used."""
    model = _tiny_dit(QwenImageDitConfig())
    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        free_gib=0.0,
        reusable_bytes=reusable_gib * 1024**3,
    )
    assert widths == expected


@requires_cuda
@pytest.mark.parametrize("cond_len,uncond_len,expected", [(7, 3, [1, 1]), (5, 5, [2])])
def test_attention_autotune_keeps_unequal_prompts_serial(
    single_process_model_parallel, cond_len, uncond_len, expected
):
    """Autotune can pick an impl the masked batch-2 path would bypass."""
    model = _tiny_dit(QwenImageDitConfig())
    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        cond_len,
        uncond_len,
        edit=False,
        streamed=True,
        autotune=True,
    )
    assert widths == expected


@pytest.mark.parametrize(
    "device,dtype,keeps_fa",
    [
        ("cuda", torch.bfloat16, True),
        ("cpu", torch.bfloat16, False),
        ("cuda", torch.float32, False),
    ],
)
def test_fa_counts_only_where_the_varlen_kernel_runs(device, dtype, keeps_fa):
    """Unequal prompts stay serial where the masked call cannot keep FA."""
    model = torch.nn.Module()
    model.attn = USPAttention.__new__(USPAttention)
    torch.nn.Module.__init__(model.attn)
    model.attn.backend = AttentionBackendEnum.FA
    batch = SimpleNamespace(sampling_params=SimpleNamespace(skip_softmax_params=None))
    server_args = SimpleNamespace(enable_attention_backend_autotune=False)
    with patch.object(attention_layer, "_VARLEN_FA_ENABLED", True):
        assert (
            _stage()._cfg_mask_keeps_attention(
                model, batch, server_args, device=torch.device(device), dtype=dtype
            )
            is keeps_fa
        )
