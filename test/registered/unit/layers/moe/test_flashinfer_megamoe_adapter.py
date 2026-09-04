import pytest
import torch

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.moe.flashinfer_mega_moe import _view_byte_dtype
from sglang.srt.server_args import ServerArgs


def _resolve_dummy(**kwargs):
    args = ServerArgs(
        model_path="dummy",
        tp_size=4,
        dp_size=4,
        enable_dp_attention=True,
        ep_size=4,
        moe_runner_backend="flashinfer_megamoe",
        **kwargs,
    )
    args.resolve_once()
    return resolved_view(args)


def test_flashinfer_megamoe_resolution_contract():
    cfg = _resolve_dummy()
    assert cfg.moe_a2a_backend == "megamoe"
    assert cfg.disable_shared_experts_fusion
    assert cfg.flashinfer_megamoe_max_num_tokens == 8192


@pytest.mark.parametrize("moe_a2a_backend", ["deepep", "mori"])
def test_flashinfer_megamoe_rejects_external_dispatch(moe_a2a_backend):
    with pytest.raises(ValueError, match="owns dispatch and combine"):
        _resolve_dummy(moe_a2a_backend=moe_a2a_backend)


def test_flashinfer_megamoe_rejects_eplb():
    with pytest.raises(ValueError, match="disable EPLB"):
        _resolve_dummy(enable_eplb=True)


def test_flashinfer_megamoe_rejects_tbo():
    with pytest.raises(ValueError, match="does not support.*two-batch-overlap"):
        _resolve_dummy(enable_two_batch_overlap=True)


def test_mxfp4_weight_view_is_zero_copy():
    storage = torch.arange(16, dtype=torch.uint8)
    viewed = _view_byte_dtype(storage, torch.float4_e2m1fn_x2)
    assert viewed.dtype == torch.float4_e2m1fn_x2
    assert viewed.data_ptr() == storage.data_ptr()
