import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _make_eagle(topk=4, spec_steps=3, device="cuda"):
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    return EagleVerifyInput.create_idle_input(
        topk=topk,
        spec_steps=spec_steps,
        num_verify_tokens=topk * (spec_steps + 1),
        device=device,
    )


def _make_dflash(draft_token_num=8, device="cuda"):
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    return DFlashVerifyInput(
        draft_token=torch.empty(0, dtype=torch.long, device=device),
        positions=torch.empty(0, dtype=torch.int64, device=device),
        draft_token_num=draft_token_num,
        custom_mask=torch.full((0,), True, dtype=torch.bool, device=device),
    )


def _call(spec_input, batch_size, device="cuda"):
    req_pool_indices = torch.arange(batch_size, device=device)
    paged_kernel_lens = torch.full((batch_size,), 10, dtype=torch.int32, device=device)
    paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
    req_to_token = torch.zeros((batch_size, 20), dtype=torch.int32, device=device)
    _, _, _, mask = spec_input.generate_attn_arg_prefill(
        req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, req_to_token
    )
    expected = (
        paged_kernel_lens_sum * spec_input.draft_token_num
        + (spec_input.draft_token_num**2) * batch_size
    )
    return expected, mask


def test_eagle_mask_exact_size():
    s = _make_eagle()
    for bs in [4, 8, 2]:
        expected, mask = _call(s, bs)
        assert mask.numel() == expected


def test_eagle_mask_no_unbounded_growth():
    s = _make_eagle()
    for bs in [2, 16, 2, 16, 2, 32, 2]:
        expected, mask = _call(s, bs)
        assert mask.numel() == expected
    _, mask_at_peak = _call(s, 32)
    assert s.custom_mask.numel() <= mask_at_peak.numel()


def test_dflash_mask_exact_size():
    s = _make_dflash()
    for bs in [4, 8, 2]:
        expected, mask = _call(s, bs)
        assert mask.numel() == expected


def test_dflash_mask_no_unbounded_growth():
    s = _make_dflash()
    for bs in [2, 16, 2, 16, 2, 32, 2]:
        expected, mask = _call(s, bs)
        assert mask.numel() == expected
    _, mask_at_peak = _call(s, 32)
    assert s.custom_mask.numel() <= mask_at_peak.numel()
