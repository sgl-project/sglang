import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import petit_utils
from sglang.srt.layers.quantization.petit import PetitMxfp4MoEScheme

_NUM_EXPERTS = 32
_HIDDEN_SIZE = 128
_W13_SIZE_N = 256
_W13_SIZE_K = 128
_W2_SIZE_N = 128
_W2_SIZE_K = 256


def _moe_impl_kwargs(**overrides):
    kwargs = dict(
        hidden_states=torch.ones((1, _HIDDEN_SIZE), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_weights=torch.ones((1, 2), dtype=torch.float32),
        topk_ids=torch.zeros((1, 2), dtype=torch.int32),
        w13_weight=torch.empty((_NUM_EXPERTS, _W13_SIZE_N, 64), dtype=torch.uint8),
        w2_weight=torch.empty((_NUM_EXPERTS, _W2_SIZE_N, 128), dtype=torch.uint8),
        w13_weight_scale=torch.empty((_NUM_EXPERTS, _W13_SIZE_N, 4), dtype=torch.uint8),
        w2_weight_scale=torch.empty((_NUM_EXPERTS, _W2_SIZE_N, 8), dtype=torch.uint8),
        w13_size_n=_W13_SIZE_N,
        w13_size_k=_W13_SIZE_K,
        w2_size_n=_W2_SIZE_N,
        w2_size_k=_W2_SIZE_K,
        num_local_experts=_NUM_EXPERTS,
        activation="silu",
    )
    kwargs.update(overrides)
    return kwargs


def _fake_sorting_result(topk_ids):
    sorted_token_ids = torch.arange(topk_ids.numel(), dtype=torch.int32)
    sorted_weights = torch.ones_like(sorted_token_ids, dtype=torch.float32)
    sorted_expert_ids = torch.zeros((1,), dtype=torch.int32)
    num_valid_ids = torch.tensor(
        [topk_ids.numel(), topk_ids.shape[0]], dtype=torch.int32
    )
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, None


def _fake_quantize(hidden_states, num_local_tokens=None):
    return hidden_states.to(torch.uint8), torch.empty((1, hidden_states.shape[0]))


def _fake_fused_moe(**kwargs):
    input_q = kwargs["input_q"]
    w2_q = kwargs["w2_q"]
    return input_q.new_zeros((input_q.shape[0], w2_q.shape[1]), dtype=torch.bfloat16)


def _patch_petit_moe(monkeypatch, *, quantize, sorting, fused_moe=_fake_fused_moe):
    monkeypatch.setattr(petit_utils, "_quantize_fp8_block128", quantize)
    monkeypatch.setattr(petit_utils, "_aiter_moe_sorting", sorting)
    monkeypatch.setattr(
        petit_utils,
        "_raw_fused_moe_fp8_blockscale_g1u1_mxfp4",
        fused_moe,
    )


def test_petit_preserves_global_ids_and_masks_invalid_ids_for_aiter_sorting():
    expert_mask = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
    topk_ids = torch.tensor([[0, -1, 3, 9]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    (
        prepared_ids,
        prepared_weights,
        prepared_mask,
        num_sorting_experts,
    ) = petit_utils._prepare_petit_aiter_sorting_inputs(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorting_expert_mask=expert_mask,
    )

    assert num_sorting_experts == 5
    torch.testing.assert_close(
        prepared_ids, torch.tensor([[0, 4, 3, 4]], dtype=torch.int32)
    )
    torch.testing.assert_close(prepared_weights, topk_weights)
    torch.testing.assert_close(
        prepared_mask, torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32)
    )


def test_petit_moe_calls_aiter_sorting_with_global_mask_and_num_local_tokens(
    monkeypatch,
):
    captured = {}

    def fake_quantize(hidden_states, num_local_tokens=None):
        captured["quant_num_local_tokens"] = num_local_tokens
        return _fake_quantize(hidden_states, num_local_tokens=num_local_tokens)

    def fake_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        **kwargs,
    ):
        captured["topk_ids"] = topk_ids.clone()
        captured["topk_weights"] = topk_weights.clone()
        captured["num_experts"] = num_experts
        captured["model_dim"] = model_dim
        captured["moebuf_dtype"] = moebuf_dtype
        captured["block_size"] = block_size
        captured["expert_mask"] = kwargs["expert_mask"].clone()
        captured["sorting_num_local_tokens"] = kwargs["num_local_tokens"]
        captured["dispatch_policy_present"] = "dispatch_policy" in kwargs
        return _fake_sorting_result(topk_ids)

    _patch_petit_moe(monkeypatch, quantize=fake_quantize, sorting=fake_sorting)

    topk_ids = torch.tensor([[64, -1, 127, 300]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    expert_mask = torch.zeros((256,), dtype=torch.int32)
    expert_mask[64:128] = 1

    output = petit_utils._apply_petit_mxfp4_moe_impl(
        **_moe_impl_kwargs(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_mask=expert_mask,
            num_local_tokens=torch.tensor([5], dtype=torch.int64),
        )
    )

    assert output.shape == (1, _HIDDEN_SIZE)
    torch.testing.assert_close(
        captured["topk_ids"], torch.tensor([[64, 256, 127, 256]], dtype=torch.int32)
    )
    torch.testing.assert_close(captured["topk_weights"], topk_weights)
    torch.testing.assert_close(
        captured["expert_mask"],
        torch.cat([expert_mask, torch.zeros((1,), dtype=torch.int32)]),
    )
    torch.testing.assert_close(
        captured["quant_num_local_tokens"], torch.tensor([1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        captured["sorting_num_local_tokens"], torch.tensor([1], dtype=torch.int32)
    )
    assert captured["num_experts"] == 257
    assert captured["model_dim"] == _HIDDEN_SIZE
    assert captured["moebuf_dtype"] is torch.bfloat16
    assert captured["block_size"] == 32
    assert not captured["dispatch_policy_present"]


def test_petit_scheme_forwards_aiter_topk_contract_from_standard_dispatch(
    monkeypatch,
):
    captured = {}

    class FakeConfig:
        apply_router_weight_on_input = False
        activation = "silu"

    class FakeLayer:
        expert_mask_gpu = torch.zeros((257,), dtype=torch.int32)
        expert_mask_gpu[64:128] = 1
        expert_mask_gpu[256] = 1

    class FakeDispatchOutput:
        hidden_states = torch.ones((1, _HIDDEN_SIZE), dtype=torch.bfloat16)
        hidden_states_scale = None

        topk_ids = torch.tensor([[64, 63, 127, 256, -1, 300]], dtype=torch.int32)
        topk_ids.num_token_non_padded = torch.tensor([1], dtype=torch.int32)
        topk_output = StandardTopKOutput(
            torch.ones((1, 6), dtype=torch.float32),
            topk_ids,
            torch.empty((1, 257), dtype=torch.float32),
        )

    def fake_apply_layer(**kwargs):
        captured["topk_ids"] = kwargs["topk_ids"]
        captured["expert_mask"] = kwargs["expert_mask"]
        captured["num_local_tokens"] = kwargs["num_local_tokens"]
        return torch.zeros((1, _HIDDEN_SIZE), dtype=torch.bfloat16)

    monkeypatch.setattr(
        "sglang.srt.layers.quantization.petit.apply_petit_mxfp4_moe_layer",
        fake_apply_layer,
    )

    scheme = PetitMxfp4MoEScheme.__new__(PetitMxfp4MoEScheme)
    scheme.moe_runner_config = FakeConfig()
    scheme.apply_weights(FakeLayer(), FakeDispatchOutput())

    torch.testing.assert_close(
        captured["topk_ids"],
        torch.tensor([[64, 63, 127, 256, -1, 300]], dtype=torch.int32),
    )
    torch.testing.assert_close(captured["expert_mask"], FakeLayer.expert_mask_gpu)
    torch.testing.assert_close(
        captured["num_local_tokens"], torch.tensor([1], dtype=torch.int32)
    )
