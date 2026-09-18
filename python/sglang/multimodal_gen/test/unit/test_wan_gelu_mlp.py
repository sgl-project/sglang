import pytest
import torch

from sglang.kernels.ops.diffusion import (
    fused_gelu_active,
    mount_fused_linear_gelu,
    mount_nvfp4_bias_gelu,
    nvfp4_bias_gelu_active,
    unmount_fused_linear_gelu,
    unmount_nvfp4_bias_gelu,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import _WanGELUMLP
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires a Blackwell CUDA GPU",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_wan_gelu_mlp_quality_path_and_lossless_restore():
    _ensure_single_process_parallel_runtime()
    torch.manual_seed(0)
    mlp = _WanGELUMLP(64, 256, prefix="", quant_config=None).to(
        device="cuda", dtype=torch.bfloat16
    )
    for parameter in mlp.parameters():
        parameter.normal_(mean=0.0, std=0.02)

    x = torch.randn(2, 129, 64, device="cuda", dtype=torch.bfloat16)
    reference = mlp(x)
    assert not fused_gelu_active(mlp)

    assert mount_fused_linear_gelu(mlp)
    torch.testing.assert_close(mlp(x), reference, atol=2e-2, rtol=2e-2)

    unmount_fused_linear_gelu(mlp)
    assert not fused_gelu_active(mlp)
    assert torch.equal(mlp(x), reference)


@requires_blackwell
def test_wan_nvfp4_mlp_defers_bias_for_gelu_fusion():
    _ensure_single_process_parallel_runtime()
    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
    )

    mlp = _WanGELUMLP(64, 256, prefix="", quant_config=quant_config)

    assert mlp.fuse_bias_gelu_tanh
    assert not mlp.fc_in.skip_bias_add
    assert not mlp.fc_out.skip_bias_add
    assert not nvfp4_bias_gelu_active(mlp)

    assert mount_nvfp4_bias_gelu(mlp)
    assert nvfp4_bias_gelu_active(mlp)
    assert mlp.fc_in.skip_bias_add
    unmount_nvfp4_bias_gelu(mlp)
    assert not nvfp4_bias_gelu_active(mlp)
    assert not mlp.fc_in.skip_bias_add


@requires_blackwell
def test_wan_nvfp4_mlp_does_not_mark_excluded_linear():
    _ensure_single_process_parallel_runtime()
    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=["fc_in"],
    )

    mlp = _WanGELUMLP(64, 256, prefix="", quant_config=quant_config)

    assert not mlp.fuse_bias_gelu_tanh
    assert not mlp.fc_in.skip_bias_add
    assert not mount_nvfp4_bias_gelu(mlp)
