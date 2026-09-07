# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision weight and TP/Ulysses numerical contracts for H3 DiT."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.cube_sparse_attn import (
    CubeSparseAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.usp import _usp_input_all_to_all_packed_qkv
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _needs_device_weight_postprocess,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3Attention,
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
    _copy_grouped_qkv_tp_shard,
    _diffusers_h3_checkpoint,
    _modulate_gate,
    _qkv_scale_block_rows,
    _reorder_grouped_qkv_to_qkv,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_pruned_adaln_lora_projection_preserves_affine_term():
    model = SimpleNamespace(
        arch=SimpleNamespace(adaln_affine_input_dim=3, time_embed_dim=2),
        adaln_basis=torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]),
        adaln_mean=torch.tensor([1.0, 2.0, 3.0]),
        _adaln_precomputed=False,
    )
    prefix = "blocks.0.adaln_proj.linear."
    a = torch.tensor([[2.0, 3.0, 4.0]])
    b = torch.tensor([[5.0], [6.0]])
    actual = MiniMaxH3DiTModel.prepare_lora_adapter(
        model, {prefix + "lora_A": a, prefix + "lora_B": b}
    )
    torch.testing.assert_close(actual[prefix + "lora_A"], a @ model.adaln_basis.T)
    torch.testing.assert_close(
        actual[prefix + "lora_output_offset"], b @ (a @ model.adaln_mean)
    )


def test_native_weight_names_and_grouped_qkv_reorder():
    arch = MiniMaxH3DiTArchConfig()
    assert arch.reverse_param_names_mapping == {}
    mapping = get_param_names_mapping(arch.param_names_mapping)
    for key in (
        "video_patch_proj.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "blocks.49.attn.qkv_proj.weight",
        "final_layer.audio_out.weight",
    ):
        assert mapping(key) == (key, None, None)

    assert mapping(
        "base_model.model.transformer.transformer_blocks.7.attn.to_k.lora_A.default"
    ) == ("blocks.7.attn.qkv_proj.lora_A", 1, 3)
    assert mapping("token_refiner.refiner_blocks.1.ff.net.0.proj.lora_B") == (
        "token_refiner.blocks.1.mlp.fc1.lora_B",
        None,
        None,
    )
    assert mapping("transformer.transformer_blocks.3.adaln_proj.linear.lora_A") == (
        "blocks.3.adaln_proj.linear.lora_A",
        None,
        None,
    )
    assert mapping("transformer.audio_proj_out.lora_B") == (
        "final_layer.audio_out.lora_B",
        None,
        None,
    )
    assert mapping("blocks.3.attn.out_proj.lora_A") == (
        "blocks.3.attn.out_proj.lora_A",
        None,
        None,
    )
    assert mapping("transformer.blocks.0.attn.qkv_proj.weight") == (
        "transformer.blocks.0.attn.qkv_proj.weight",
        None,
        None,
    )

    assert mapping("transformer_blocks.7.attn.to_k.qweight") == (
        "blocks.7.attn.qkv_proj.qweight",
        1,
        3,
    )
    assert mapping("time_embedder.table") == ("adaln_t_table", None, None)
    assert mapping("transformer_blocks.7.adaln_proj.folded_bias") == (
        "blocks.7.adaln_proj.linear.bias",
        None,
        None,
    )

    diffusers_weights = [
        ("transformer_blocks.0.attn.to_q.qweight", torch.full((2, 3), 1)),
        ("transformer_blocks.0.attn.to_k.qweight", torch.full((2, 3), 2)),
        ("transformer_blocks.0.attn.to_v.qweight", torch.full((2, 3), 3)),
        (
            "transformer_blocks.0.ff.net.0.proj.weight",
            torch.arange(8).reshape(4, 2),
        ),
    ]
    converted = dict(_diffusers_h3_checkpoint(diffusers_weights))
    assert torch.equal(
        converted["blocks.0.attn.qkv_proj.qweight"],
        torch.cat([tensor for _, tensor in diffusers_weights[:3]], dim=1),
    )
    assert torch.equal(
        converted["blocks.0.mlp.fc1.weight"],
        torch.tensor([[4, 5], [6, 7], [0, 1], [2, 3]]),
    )

    pruned_config = MiniMaxH3DiTConfig()
    pruned_config.update_model_arch(
        {
            "_class_name": "MiniMaxH3PrunedTransformer3DModel",
            "adaln_rank": 8,
            "time_embed_dim": 2688,
            "time_table_size": 1025,
        }
    )
    assert pruned_config.arch_config.time_embed_dim == 8
    assert pruned_config.arch_config.adaln_curve_grid == 1025
    assert pruned_config.arch_config.adaln_affine_input_dim == 2688

    weight = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    actual = _reorder_grouped_qkv_to_qkv(
        weight,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=2,
    )
    expected = torch.tensor(
        [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11],
        dtype=torch.float32,
    ).reshape(12, 1)
    torch.testing.assert_close(actual, expected)

    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        grouped = torch.arange(48, dtype=torch.float32).reshape(24, 2).to(dtype)
        reordered = _reorder_grouped_qkv_to_qkv(
            grouped,
            num_query_groups=4,
            heads_per_group=1,
            head_dim=2,
        )
        for tp_size in (1, 2, 4):
            local_rows = 8 // tp_size
            for tp_rank in range(tp_size):
                start = tp_rank * local_rows
                param = torch.nn.Parameter(
                    torch.empty(3 * local_rows, 2, dtype=dtype),
                    requires_grad=False,
                )
                param.output_dim = 0
                assert _copy_grouped_qkv_tp_shard(
                    param,
                    grouped,
                    num_query_groups=4,
                    head_dim=2,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                )
                expected_shard = reordered.view(3, 8, 2)[
                    :, start : start + local_rows
                ].reshape(-1, 2)
                assert torch.equal(param, expected_shard)


def test_pruned_adaln_curve_interpolates_without_timestep_mlp():
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.time_embedder = None
    model.adaln_t_table = torch.nn.Parameter(
        torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]),
        requires_grad=False,
    )

    result = model._time_embedding(torch.tensor([0.0, 0.25, 1.0]))

    torch.testing.assert_close(
        result,
        torch.tensor([[0.0, 2.0], [1.0, 3.0], [4.0, 6.0]]),
        rtol=0,
        atol=0,
    )


class _KwargIdentity(torch.nn.Module):
    def forward(self, x, **_kwargs):
        return x


def test_cache_dit_preservation_only_makes_first_gate_out_of_place():
    block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
    torch.nn.Module.__init__(block)
    block.norm1 = torch.nn.Identity()
    block.norm2 = torch.nn.Identity()
    block.attn = _KwargIdentity()
    block.mlp = torch.nn.Identity()
    gate_modes = []

    def fake_gate(residual, _gate, _other, _indices, *, dtype, allow_inplace=True):
        gate_modes.append(allow_inplace)
        return residual.to(dtype)

    def run(preserve):
        block.preserve_input_for_cache_dit = preserve
        gate_modes.clear()
        block(
            torch.zeros(2, 4),
            adaln_input=torch.zeros(1, 4),
            combined_indices=torch.zeros(2, dtype=torch.long),
            rope_cache=None,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            max_seqlen=2,
            adaln_params=tuple(torch.zeros(1, 4) for _ in range(6)),
        )
        return list(gate_modes)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3._modulate_scale_shift",
            side_effect=lambda value, *_args, **_kwargs: value,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3._modulate_gate",
            side_effect=fake_gate,
        ),
    ):
        assert run(preserve=False) == [True, True]
        # Only the first gated residual can alias the block input Cache-DiT
        # holds by reference, so only it goes out-of-place. The second works on
        # a block-local buffer and keeps the fused in-place kernel.
        assert run(preserve=True) == [False, True]


def test_cache_dit_input_preservation_toggles_every_block():
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(5)])

    model.set_cache_dit_input_preservation(True)
    assert all(block.preserve_input_for_cache_dit for block in model.blocks)

    model.set_cache_dit_input_preservation(False)
    assert not any(block.preserve_input_for_cache_dit for block in model.blocks)


@pytest.mark.parametrize(
    ("global_backend", "expected_backend"),
    [
        (None, AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN),
        (AttentionBackendEnum.FA, AttentionBackendEnum.FA),
    ],
)
def test_lazy_attention_resolution_preserves_backend_precedence(
    global_backend, expected_backend
):
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.arch = SimpleNamespace(attention_head_dim=128)
    model._component_attention_backend_override = (
        AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
    )
    model._resolved_attention_backend = None
    backend = Mock()
    backend.get_enum.return_value = expected_backend

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
            "get_global_forced_attn_backend",
            return_value=global_backend,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_attn_backend",
            return_value=backend,
        ) as get_attn_backend,
    ):
        model._resolve_attention_backend_once()

    get_attn_backend.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=expected_backend,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )
    assert model._resolved_attention_backend is expected_backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cache_dit_out_of_place_gate_preserves_cuda_input():
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    gate = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    other = torch.randn_like(x)
    indices = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.long)
    expected = _modulate_gate(
        x.clone(),
        gate,
        other,
        indices,
        dtype=torch.bfloat16,
        allow_inplace=True,
    )

    output = _modulate_gate(
        x,
        gate,
        other,
        indices,
        dtype=torch.bfloat16,
        allow_inplace=False,
    )

    assert output.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_tp_and_ulysses_admission_uses_tp_local_shapes():
    arch = MiniMaxH3DiTArchConfig()
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    model._validate_tp_config(arch=arch, tp_size=2)
    model._validate_tp_config(arch=arch, tp_size=8)
    MiniMaxH3DiTModel._validate_sequence_parallel_config(
        arch=arch,
        tp_size=2,
        ulysses_size=4,
        ring_size=1,
    )

    with pytest.raises(ValueError):
        model._validate_tp_config(arch=arch, tp_size=3)
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._validate_sequence_parallel_config(
            arch=arch,
            tp_size=4,
            ulysses_size=4,
            ring_size=1,
        )
    # ring is implemented now: it splits rows, not heads, so it carries no
    # head-divisibility constraint of its own
    MiniMaxH3DiTModel._validate_sequence_parallel_config(
        arch=arch,
        tp_size=1,
        ulysses_size=1,
        ring_size=2,
    )
    MiniMaxH3DiTModel._validate_sequence_parallel_config(
        arch=arch,
        tp_size=1,
        ulysses_size=8,
        ring_size=2,
    )
    # what ring does constrain is the packed-sequence alignment, which has to
    # divide by the *combined* degree because ring adds an outer row split on
    # top of Ulysses's inner one
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._validate_sequence_parallel_config(
            arch=arch,
            tp_size=1,
            ulysses_size=8,
            ring_size=3,
        )
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._validate_sequence_parallel_config(
            arch=arch,
            tp_size=1,
            ulysses_size=1,
            ring_size=0,
        )


def test_cube_backend_advertises_packed_varlen_capability():
    assert CubeSparseAttentionBackend.supports_packed_varlen()


def test_attention_retains_transformer_scoped_backend():
    _ensure_single_process_parallel_runtime()
    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
            "get_component_forced_attn_backend",
            return_value=AttentionBackendEnum.CUBE_SPARSE_ATTN,
        ),
        torch.device("meta"),
    ):
        attention = MiniMaxH3Attention(
            MiniMaxH3DiTArchConfig(), None, prefix="blocks.0.attn"
        )

    assert (
        attention._selected_attention_backend is AttentionBackendEnum.CUBE_SPARSE_ATTN
    )


def test_model_lazy_resolver_keeps_transformer_scoped_backend():
    _ensure_single_process_parallel_runtime()
    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
            "get_component_forced_attn_backend",
            return_value=AttentionBackendEnum.CUBE_SPARSE_ATTN,
        ),
        torch.device("meta"),
    ):
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(), hf_config={}, quant_config=None
        )

    class FakeBackend:
        @staticmethod
        def get_enum():
            return AttentionBackendEnum.CUBE_SPARSE_ATTN

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_attn_backend",
            return_value=FakeBackend,
        ) as resolve,
        patch.object(MiniMaxH3Attention, "_set_attention_backend") as install,
    ):
        model._resolve_attention_backend_once()

    resolve.assert_called_once_with(
        model.arch.attention_head_dim,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.CUBE_SPARSE_ATTN,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )
    attention_count = sum(
        isinstance(module, MiniMaxH3Attention) for module in model.modules()
    )
    assert install.call_count == attention_count
    assert model._resolved_attention_backend is AttentionBackendEnum.CUBE_SPARSE_ATTN


def test_token_refiner_routes_cube_selection_to_exact_fa():
    class FakeImpl:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeBackend:
        def __init__(self, enum):
            self.enum = enum

        def get_enum(self):
            return self.enum

        def get_impl_cls(self):
            return FakeImpl

    cube = FakeBackend(AttentionBackendEnum.CUBE_SPARSE_ATTN)
    fa = FakeBackend(AttentionBackendEnum.FA)
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 128
    attention.num_heads = 8
    attention.softmax_scale = 128**-0.5
    attention.prefix = "test.attn"
    attention._cube_sparse_capable = False

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.get_attn_backend",
        return_value=fa,
    ) as resolve:
        attention._set_attention_backend(cube)

    assert attention._attention_backend_enum is AttentionBackendEnum.FA
    resolve.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.FA,
    )


def test_meta_model_captures_component_attention_override():
    _ensure_single_process_parallel_runtime()
    with (
        component_attn_backend_context_manager(
            AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
            component_name="transformer",
        ),
        torch.device("meta"),
    ):
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(),
            hf_config={},
            quant_config=None,
        )

    assert (
        model._component_attention_backend_override
        is AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
    )


def test_meta_model_enforces_mixed_precision_weight_contract():
    expected_fp32 = set(MINIMAX_H3_FP32_PARAM_NAMES) | set(MINIMAX_H3_FP32_BUFFER_NAMES)
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(),
            hf_config={},
            quant_config=None,
        )

    assert model._fsdp_mixed_dtype_params
    qkv_weight = model.blocks[0].attn.qkv_proj.weight
    assert callable(qkv_weight.rank_local_weight_transform)
    for name, tensor in model.state_dict().items():
        if name in expected_fp32:
            assert tensor.dtype == torch.float32, name
        elif tensor.is_floating_point():
            assert tensor.dtype == torch.bfloat16, name


def test_pruned_meta_model_preserves_curve_adaln_fp32_island():
    _ensure_single_process_parallel_runtime()
    config = MiniMaxH3DiTConfig(
        arch_config=MiniMaxH3DiTArchConfig(
            adaln_curve_grid=1025,
            time_embed_dim=8,
        )
    )
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=config,
            hf_config={},
            quant_config=None,
        )

    assert model.time_embedder is None
    assert model.adaln_t_table.dtype == torch.float32
    assert model.blocks[0].adaln_proj.linear.weight.dtype == torch.float32
    assert model.final_layer.adaln_proj.linear.weight.dtype == torch.float32


def test_online_fp8_keeps_fp32_boundaries_and_ignored_layers_unquantized():
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(),
            hf_config={},
            quant_config=Fp8Config(ignored_layers=["blocks.0.attn.out_proj"]),
        )

    assert isinstance(model.blocks[0].attn.qkv_proj.quant_method, Fp8LinearMethod)
    assert isinstance(
        model.blocks[0].attn.out_proj.quant_method, UnquantizedLinearMethod
    )
    for layer in (
        model.video_patch_proj,
        model.audio_patch_proj,
        model.time_embedder.proj_in,
        model.time_embedder.proj_out,
        model.final_layer.video_out,
        model.final_layer.audio_out,
    ):
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)


def _block_fp8_quant_config(block: int = 128, **kwargs) -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[block, block],
        **kwargs,
    )


def _meta_h3(quant_config) -> MiniMaxH3DiTModel:
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        return MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(), hf_config={}, quant_config=quant_config
        )


def test_offline_block_fp8_checkpoint_layout_and_cpu_load():
    quant_config = _block_fp8_quant_config(
        ignored_layers=[
            "video_patch_proj",
            "audio_patch_proj",
            "time_embedder.proj_in",
            "time_embedder.proj_out",
            "final_layer.video_out",
            "final_layer.audio_out",
        ],
    )
    model = _meta_h3(quant_config)

    params = dict(model.named_parameters())
    fp8_weights = {name for name, p in params.items() if p.dtype == torch.float8_e4m3fn}
    scales = {name for name in params if name.endswith("weight_scale_inv")}
    assert len(fp8_weights) == 260, len(fp8_weights)
    assert {f"{n[: -len('weight')]}weight_scale_inv" for n in fp8_weights} == scales

    for scale_name in scales:
        weight = params[f"{scale_name[: -len('weight_scale_inv')]}weight"]
        n, k = weight.shape
        assert params[scale_name].dtype == torch.float32, scale_name
        assert tuple(params[scale_name].shape) == (
            -(-n // 128),
            -(-k // 128),
        ), scale_name

    for layer in (
        model.video_patch_proj,
        model.audio_patch_proj,
        model.time_embedder.proj_in,
        model.time_embedder.proj_out,
        model.final_layer.video_out,
        model.final_layer.audio_out,
    ):
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)
        assert layer.weight.dtype != torch.float8_e4m3fn

    # False keeps the DiT off the GPU during load, which is the point of
    # loading a pre-quantized checkpoint.
    assert _needs_device_weight_postprocess(quant_config) is False
    assert _needs_device_weight_postprocess(Fp8Config()) is True


def test_block_fp8_qkv_scale_follows_its_weight_through_the_grouped_reorder():
    """A block scale is row-indexed in blocks, so it moves with the qkv rows.

    The checkpoint is quantized in the grouped per-head [q, k, v] layout, and
    the DiT permutes those rows into [q_all, k_all, v_all] as it loads. Leaving
    the scale behind leaves every 128x128 tile scaling rows it no longer covers:
    the model loads without error and renders noise.
    """
    heads, head_dim, block = 8, 128, 128
    rows, cols = heads * 3 * head_dim, 256
    torch.manual_seed(0)
    weight = torch.randn(rows, cols)

    reordered = _reorder_grouped_qkv_to_qkv(
        weight, num_query_groups=heads, heads_per_group=1, head_dim=head_dim
    )
    # A quantizer sees the grouped layout, so its scales are computed there.
    tiles = weight.view(rows // block, block, cols // block, block)
    scale = tiles.abs().amax(dim=(1, 3)) / 448.0
    reordered_scale = _reorder_grouped_qkv_to_qkv(
        scale, num_query_groups=heads, heads_per_group=1, head_dim=head_dim // block
    )

    def tile_max(w: torch.Tensor) -> torch.Tensor:
        return w.view(rows // block, block, cols // block, block).abs().amax(dim=(1, 3))

    assert torch.equal(tile_max(reordered) / 448.0, reordered_scale)
    assert not torch.equal(tile_max(reordered) / 448.0, scale)


def test_qkv_block_scale_param_is_reordered_in_blocks():
    model = _meta_h3(_block_fp8_quant_config())
    arch = model.arch
    qkv = model.blocks[0].attn.qkv_proj
    block = 128
    scale_rows = 3 * arch.num_attention_heads * arch.attention_head_dim // block
    assert qkv.weight_scale_inv.shape[0] == scale_rows

    # The installed transform is what rank-local FSDP applies and what the
    # wrapped weight_loader runs before handing off, so it is the contract.
    loaded = torch.arange(scale_rows * 4, dtype=torch.float32).reshape(scale_rows, 4)
    expected = _reorder_grouped_qkv_to_qkv(
        loaded,
        num_query_groups=arch.num_attention_heads,
        heads_per_group=1,
        head_dim=arch.attention_head_dim // block,
    )
    got = qkv.weight_scale_inv.rank_local_weight_transform(loaded)
    assert torch.equal(got, expected)
    assert not torch.equal(got, loaded)

    # The weight itself keeps the per-row permutation.
    unblocked = torch.zeros(scale_rows * block, 4)
    assert qkv.weight.rank_local_weight_transform(unblocked).shape == unblocked.shape


def test_unquantized_and_per_tensor_qkv_keep_their_loaders():
    for quant_config in (None, Fp8Config(is_checkpoint_fp8_serialized=True)):
        qkv = _meta_h3(quant_config).blocks[0].attn.qkv_proj
        assert not hasattr(qkv, "weight_scale_inv")
        assert _qkv_scale_block_rows(qkv, 128) == 1


def test_block_that_straddles_heads_is_rejected():
    # head_dim is 128, so a 256-row block covers two heads' rows at once and no
    # row permutation can repair it.
    with pytest.raises(ValueError, match="divides the head dim"):
        _meta_h3(_block_fp8_quant_config(block=256))


def test_sdpa_varlen_fallback_matches_naive_packed_reference():
    torch.manual_seed(0)
    heads, dim = 2, 8
    bounds = (0, 5, 6, 13)
    cu = torch.tensor(bounds, dtype=torch.int32)
    q = torch.randn(bounds[-1], heads, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = dim**-0.5

    attention = SDPAImpl(
        num_heads=heads,
        head_size=dim,
        causal=False,
        softmax_scale=scale,
    )
    out = attention.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        cu_seqlens_host=bounds,
        max_seqlen=7,
    )
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        seg_q = q[start:stop].transpose(0, 1)
        seg_k = k[start:stop].transpose(0, 1)
        seg_v = v[start:stop].transpose(0, 1)
        expected = (
            torch.softmax(seg_q @ seg_k.transpose(-1, -2) * scale, dim=-1) @ seg_v
        ).transpose(0, 1)
        torch.testing.assert_close(out[start:stop], expected, atol=1e-6, rtol=1e-6)


@patch(
    "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
    return_value=2,
)
def test_packed_qkv_exchange_preserves_rank_and_head_order(_):
    seq, heads, dim = 3, 4, 2
    q_ranks = [
        torch.arange(seq * heads * dim).reshape(seq, heads, dim) + rank * 1000
        for rank in range(2)
    ]
    k_ranks = [q + 100 for q in q_ranks]
    v_ranks = [q + 200 for q in q_ranks]
    target_rank = 1

    def packet(q, k, v, destination):
        head_slice = slice(destination * 2, (destination + 1) * 2)
        return torch.cat((q[:, head_slice], k[:, head_slice], v[:, head_slice]), dim=-1)

    def fake_all_to_all(actual, role=None):
        expected_input = torch.stack(
            [
                packet(q_ranks[0], k_ranks[0], v_ranks[0], destination)
                for destination in range(2)
            ]
        )
        torch.testing.assert_close(actual, expected_input)
        return torch.stack(
            [
                packet(q, k, v, target_rank)
                for q, k, v in zip(q_ranks, k_ranks, v_ranks, strict=True)
            ]
        )

    with patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_all_to_all,
    ):
        actual = _usp_input_all_to_all_packed_qkv(q_ranks[0], k_ranks[0], v_ranks[0])

    for index, tensors in enumerate((q_ranks, k_ranks, v_ranks)):
        expected = torch.cat(
            [tensor[:, target_rank * 2 : (target_rank + 1) * 2] for tensor in tensors]
        )
        torch.testing.assert_close(actual[index], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ulysses_qkv_pack_is_bit_exact():
    from sglang.kernels.ops.diffusion import pack_qkv_destination_major

    torch.manual_seed(23)
    rows, world_size, heads, head_size = 65, 8, 56, 128
    qkv = torch.randn(
        rows,
        3 * heads * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = (
        tensor.reshape(rows, heads, head_size)
        for tensor in qkv.split(heads * head_size, dim=-1)
    )
    local_heads = heads // world_size
    expected = torch.empty(
        world_size,
        rows,
        local_heads,
        3 * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.view(rows, world_size, local_heads, head_size).permute(
            1, 0, 2, 3
        )
        expected[..., index * head_size : (index + 1) * head_size].copy_(shards)

    actual = pack_qkv_destination_major(q.contiguous(), k.contiguous(), v, world_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
