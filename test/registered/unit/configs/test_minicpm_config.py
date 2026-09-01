import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.hybrid_arch import (
    hybrid_lightning_config,
    mambaish_config,
)
from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    get_linear_attn_spec_by_arch,
)
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.lightning_backend import (
    LightningAttentionBackend,
)
from sglang.srt.models import minicpm as minicpm_module
from sglang.srt.models.minicpm import (
    MiniCPMAttention,
    MiniCPMDecoderLayer,
    MiniCPMLightningMixer,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_minicpm_lightning_config_defaults_are_complete():
    """A checkpoint missing optional SALA fields must still define every model input."""
    config = MiniCPMHybridConfig()

    assert config.scale_emb == 12
    assert config.scale_depth == 1.4
    assert config.dim_model_base == 256
    assert config.lightning_use_rope is True
    assert config.use_output_gate is False
    assert config.attention_bias is False
    assert config.use_output_norm is False
    assert config.qk_norm is True


def test_minicpm_empty_mixer_types_default_to_full_attention():
    config = MiniCPMHybridConfig(num_hidden_layers=3, mixer_types=[])

    assert config.mixer_types == ["minicpm4", "minicpm4", "minicpm4"]
    assert config.full_attention_layer_ids == [0, 1, 2]


def test_minicpm_sparse_config_uses_nested_fields_only():
    sparse_config = {
        "block_size": 64,
        "dense_len": 8192,
        "init_blocks": 1,
        "kernel_size": 32,
        "kernel_stride": 16,
        "topk": 64,
        "window_size": 2048,
    }
    config = MiniCPMHybridConfig(sparse_config=sparse_config)

    assert config.has_minicpm_sparse_attention
    assert config.sparse_config == sparse_config
    assert not hasattr(config, "sparse_dense_len")


def test_minicpm_short_mixer_pattern_repeats_to_layer_count():
    config = MiniCPMHybridConfig(
        num_hidden_layers=5,
        mixer_types=["minicpm4", "lightning-attn"],
        lightning_nkv=32,
    )

    assert config.mixer_types == [
        "minicpm4",
        "lightning-attn",
        "minicpm4",
        "lightning-attn",
        "minicpm4",
    ]
    assert config.full_attention_layer_ids == [0, 2, 4]
    assert config.lightning_layer_ids == [1, 3]


def test_minicpm_mixer_aliases_are_canonicalized():
    config = MiniCPMHybridConfig(
        num_hidden_layers=4,
        mixer_types=["attention", "lightning_attn"],
        lightning_nkv=32,
    )

    assert config.mixer_types == [
        "minicpm4",
        "lightning-attn",
        "minicpm4",
        "lightning-attn",
    ]


def test_minicpm_rejects_more_mixer_types_than_layers():
    with pytest.raises(ValueError, match="Invalid number of mixer types: 3"):
        MiniCPMHybridConfig(
            num_hidden_layers=2,
            mixer_types=["minicpm4", "lightning", "minicpm4"],
        )


def test_minicpm_lightning_dimensions_fall_back_to_base_attention():
    config = MiniCPMHybridConfig(
        hidden_size=96,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=None,
        lightning_nh=None,
        lightning_nkv=None,
        lightning_head_dim=None,
    )

    assert config.head_dim == 16
    assert config.lightning_nh == 6
    assert config.lightning_nkv == 3
    assert config.lightning_head_dim == 16


def test_minicpm_rejects_lightning_gqa():
    with pytest.raises(ValueError, match="seg_la backend does not support GQA"):
        MiniCPMHybridConfig(
            num_attention_heads=6,
            num_key_value_heads=3,
            mixer_types=["lightning-attn"],
        )


def test_minicpm_lightning_idle_batch_returns_empty_output():
    """An idle DP rank must return empty output instead of reducing empty tensors."""
    mixer = MiniCPMLightningMixer.__new__(MiniCPMLightningMixer)
    torch.nn.Module.__init__(mixer)
    mixer.hidden_size = 8
    forward_batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: True))

    output = mixer.forward(
        positions=torch.empty(0, dtype=torch.int64),
        hidden_states=torch.empty(0, 4),
        forward_batch=forward_batch,
    )

    assert output.shape == (0, 8)


def test_minicpm_lightning_attention_bias_applies_to_every_projection():
    """Enabling attention bias must cover every Lightning projection."""
    with get_parallel().override(tp_size=1, tp_rank=0):
        mixer = MiniCPMLightningMixer(
            hidden_size=8,
            num_heads=2,
            num_kv_heads=2,
            head_dim=4,
            use_rope=False,
            use_output_gate=True,
            attention_bias=True,
            qk_norm=False,
        )

    assert mixer.qkv_proj.bias is not None
    assert mixer.o_proj.bias is not None
    assert mixer.z_proj.bias is not None


def test_minicpm_lightning_rejects_unknown_scale():
    with (
        get_parallel().override(tp_size=1, tp_rank=0),
        pytest.raises(ValueError, match="Unsupported lightning scale"),
    ):
        MiniCPMLightningMixer(
            hidden_size=8,
            num_heads=2,
            num_kv_heads=2,
            head_dim=4,
            use_rope=False,
            qk_norm=False,
            scale="unknown",
        )


def test_minicpm_full_attention_bias_applies_to_every_projection():
    """Enabling attention bias must cover every full-attention projection."""
    with get_parallel().override(tp_size=1, tp_rank=0):
        mixer = MiniCPMAttention(
            hidden_size=8,
            num_heads=2,
            num_kv_heads=2,
            attn_use_rope=False,
            use_output_gate=True,
            attention_bias=True,
        )

    assert mixer.qkv_proj.bias is not None
    assert mixer.o_proj.bias is not None
    assert mixer.o_gate.bias is not None


def test_minicpm_full_attention_uses_configured_head_dim(monkeypatch):
    monkeypatch.setattr(minicpm_module, "SiluAndMul", torch.nn.Identity)
    config = MiniCPMHybridConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=6,
        intermediate_size=32,
        attn_use_rope=False,
    )

    with get_parallel().override(tp_size=1, tp_rank=0):
        layer = MiniCPMDecoderLayer(config)

    assert layer.self_attn.head_dim == 6
    assert layer.self_attn.q_size == 12
    assert layer.self_attn.kv_size == 12


def test_minicpm_lightning_reuses_shared_backend_and_cache_shape():
    config = MiniCPMHybridConfig(
        num_hidden_layers=2,
        mixer_types=["lightning", "minicpm4"],
        lightning_nh=4,
        lightning_nkv=4,
        lightning_head_dim=64,
    )

    model_config = SimpleNamespace(
        hf_config=config,
        linear_attn_registry_result=get_linear_attn_config(config),
    )
    assert hybrid_lightning_config(model_config) is config
    assert mambaish_config(model_config) is config

    with get_parallel().override(attn_tp_size=1):
        cache = config.mamba2_cache_params
    assert isinstance(cache, Mamba2CacheParams)
    assert cache.layers == [0]
    assert cache.shape.conv == [(0, 0)]
    assert cache.shape.temporal == (4, 64, 64)
    assert config.num_linear_key_value_heads == 4

    with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
        slopes = LightningAttentionBackend._build_slope_tensor(
            4, 2, device="cpu", layerwise_decay=False
        )
    assert len(slopes) == 2
    assert slopes[0].equal(slopes[1])


def test_non_lightning_minicpm_is_not_classified_as_linear_attention():
    config = MiniCPMHybridConfig(
        num_hidden_layers=1,
        mixer_types=["minicpm4"],
        sparse_config={},
    )
    model_config = SimpleNamespace(
        hf_config=config,
        linear_attn_registry_result=get_linear_attn_config(config),
    )

    assert hybrid_lightning_config(model_config) is None
    assert mambaish_config(model_config) is None
    for architecture in ("MiniCPMForCausalLM", "MiniCPMSALAForCausalLM"):
        assert get_linear_attn_spec_by_arch(architecture) is None


def test_lightning_backend_reads_structural_linear_config(monkeypatch):
    def fake_base_init(self, model_runner):
        self.topk = 1

    monkeypatch.setattr(MambaAttnBackendBase, "__init__", fake_base_init)
    config = SimpleNamespace(
        num_attention_heads=8,
        num_linear_key_value_heads=4,
        num_hidden_layers=2,
        lightning_layerwise_decay=False,
    )
    model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(conv=[torch.empty(0)])
            )
        ),
        sliding_window_size=None,
        model_config=SimpleNamespace(
            hf_config=config,
            is_encoder_decoder=False,
            context_len=128,
            block=256,
        ),
        device="cpu",
        kv_cache_dtype=torch.float32,
        kv_cache_dtype_str="float32",
    )

    with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
        backend = LightningAttentionBackend(model_runner)

    assert [slope.shape for slope in backend.tp_slope] == [(4, 1, 1), (4, 1, 1)]
    assert backend.tp_slope[0].equal(backend.tp_slope[1])


def test_lightning_backend_uses_layer_scale(monkeypatch):
    """Each layer's attention scale must reach the linear-attention computation."""
    captured = {}

    def fake_seg_la_fwd(**kwargs):
        captured.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        "sglang.srt.layers.attention.linear.lightning_backend.seg_la_fwd",
        fake_seg_la_fwd,
    )
    backend = LightningAttentionBackend.__new__(LightningAttentionBackend)
    backend.tp_slope = [torch.ones(1, 1, 1)]
    layer = SimpleNamespace(layer_id=0, scaling=0.25)
    metadata = SimpleNamespace(
        batch_size=1,
        query_start_loc=torch.tensor([0, 1]),
        has_initial_states=torch.tensor([False]),
    )
    q = torch.ones(1, 1, 1)

    backend._linear_attention_entry(
        q=q,
        k=q,
        v=q,
        kv_cache=torch.zeros(1, 1, 1, 1),
        state_indices_tensor=torch.tensor([0]),
        metadata=metadata,
        layer=layer,
    )

    assert captured["softmax_scale"] == 0.25


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
