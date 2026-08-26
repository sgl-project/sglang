from contextlib import ExitStack
from unittest.mock import patch

from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    FastHunyuanConfig,
    HunyuanConfig,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
    MMSingleStreamBlock,
    _hunyuan_default_attention_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_HUNYUAN = "sglang.multimodal_gen.runtime.models.dits.hunyuanvideo"


@patch(f"{_HUNYUAN}.get_ring_parallel_world_size", return_value=1)
@patch(f"{_HUNYUAN}.current_platform.is_sm120", return_value=True)
def test_fasthunyuan_prefers_cudnn_sdpa_on_sm120(_mock_is_sm120, _mock_ring):
    config = FastHunyuanConfig()

    assert config.dit_config.prefer_cudnn_sdpa_on_sm120
    assert (
        _hunyuan_default_attention_backend(config.dit_config)
        is AttentionBackendEnum.TORCH_CUDNN_SDPA
    )


@patch(f"{_HUNYUAN}.get_ring_parallel_world_size", return_value=1)
@patch(f"{_HUNYUAN}.current_platform.is_sm120", return_value=False)
def test_fasthunyuan_keeps_default_outside_sm120(_mock_is_sm120, _mock_ring):
    config = FastHunyuanConfig()

    assert _hunyuan_default_attention_backend(config.dit_config) is None


@patch(f"{_HUNYUAN}.get_ring_parallel_world_size", return_value=1)
@patch(f"{_HUNYUAN}.current_platform.is_sm120", return_value=True)
def test_fasthunyuan_keeps_compile_backend_on_sm120(_mock_is_sm120, _mock_ring):
    config = FastHunyuanConfig()

    assert (
        _hunyuan_default_attention_backend(
            config.dit_config,
            enable_torch_compile=True,
        )
        is None
    )


@patch(f"{_HUNYUAN}.get_ring_parallel_world_size", return_value=2)
@patch(f"{_HUNYUAN}.current_platform.is_sm120", return_value=True)
def test_fasthunyuan_keeps_ring_backend_on_sm120(_mock_is_sm120, _mock_ring):
    config = FastHunyuanConfig()

    assert _hunyuan_default_attention_backend(config.dit_config) is None


@patch(f"{_HUNYUAN}.get_ring_parallel_world_size", return_value=1)
@patch(f"{_HUNYUAN}.current_platform.is_sm120", return_value=True)
def test_regular_hunyuan_keeps_platform_default_on_sm120(_mock_is_sm120, _mock_ring):
    config = HunyuanConfig()

    assert not config.dit_config.prefer_cudnn_sdpa_on_sm120
    assert _hunyuan_default_attention_backend(config.dit_config) is None


def test_default_backend_is_forwarded_to_usp():
    module_names = (
        "MergedColumnParallelLinear",
        "MixedRowParallelLinear",
        "RMSNorm",
        "LayerNormScaleShift",
        "MulAdd",
        "ModulateProjection",
    )
    with ExitStack() as stack:
        for name in module_names:
            stack.enter_context(patch(f"{_HUNYUAN}.{name}", return_value=nn.Identity()))
        stack.enter_context(patch(f"{_HUNYUAN}.get_tp_world_size", return_value=1))
        usp = stack.enter_context(
            patch(f"{_HUNYUAN}.USPAttention", return_value=nn.Identity())
        )
        MMSingleStreamBlock(
            hidden_size=128,
            num_attention_heads=1,
            default_attention_backend=AttentionBackendEnum.TORCH_CUDNN_SDPA,
        )

    assert (
        usp.call_args.kwargs["default_attention_backend"]
        is AttentionBackendEnum.TORCH_CUDNN_SDPA
    )
