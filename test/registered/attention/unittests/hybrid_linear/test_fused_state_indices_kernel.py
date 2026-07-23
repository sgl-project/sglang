"""Bit-exact reference test for ``_fused_state_indices_kernel``.

``MambaAttnBackendBase._replay_metadata`` refreshes ``state_indices_list``
either through the fused single-launch kernel (CUDA + static hybrid pool with
identity v2p + replayssm off) or through the reference aten chain:

    req_pool_indices[bs - num_padding:] = 0        # zero padded rows (side effect!)
    mamba_indices = mapping[req_pool_indices]      # get_mamba_indices gather
    # identity v2p translate (static pool)
    mamba_indices[bs - num_padding:] = -1          # padding sentinel
    state_indices_list[bs - 1][:bs].copy_(mamba_indices)

The two paths must agree bit-for-bit, INCLUDING the side effect of zeroing the
padded rows of the static ``req_pool_indices`` replay buffer — captured kernels
gather with that buffer, so a non-zeroed padded row is a delayed illegal memory
access, not a visible diff. The test drives both paths on guard-padded buffers
across a bs x num_padding matrix (non-power-of-two sizes exercise the BS_UPPER
masking) and checks:

1. the produced state indices are identical over the whole ``[0, total_bs)``
   range (padding sentinel rows included);
2. the ``req_pool_indices`` buffer ends up identical (padded rows zeroed);
3. neither buffer is written beyond ``total_bs`` (guard tails stay intact).
"""

import unittest

import torch
import triton

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    _fused_state_indices_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

# Guard tail appended to every buffer; must stay untouched by both paths.
_GUARD = 8
_GUARD_SENTINEL = -7777
# Poison for the out buffer so unwritten cells inside [0, total_bs) are caught.
_OUT_POISON = -12345

_REQ_POOL_SIZE = 160
_MAMBA_POOL_SIZE = 4096


def _reference_chain(
    req_pool_indices: torch.Tensor,
    mapping: torch.Tensor,
    out_buf: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> None:
    """Replicates the _replay_metadata reference ops, in order, in place."""
    req_pool_indices[valid_bs:total_bs] = 0
    mamba_indices = mapping[req_pool_indices[:total_bs]]
    # static pool: _translate_mamba_indices is the identity
    mamba_indices[valid_bs:] = -1
    out_buf[: len(mamba_indices)].copy_(mamba_indices)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestFusedStateIndicesKernel(CustomTestCase):
    def _run_case(self, total_bs: int, num_padding: int, seed: int) -> None:
        device = torch.device("cuda")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        valid_bs = total_bs - num_padding

        # Production dtypes: req_pool_indices int64 (static replay buffer),
        # req_index_to_mamba_index_mapping int32, state_indices_list int32.
        req_pool = torch.randint(
            0, _REQ_POOL_SIZE, (total_bs + _GUARD,), generator=gen, dtype=torch.int64
        )
        req_pool[total_bs:] = _GUARD_SENTINEL
        mapping = torch.randint(
            0, _MAMBA_POOL_SIZE, (_REQ_POOL_SIZE,), generator=gen, dtype=torch.int32
        )
        out = torch.full((total_bs + _GUARD,), _OUT_POISON, dtype=torch.int32)

        req_pool_ref = req_pool.to(device)
        req_pool_fused = req_pool.to(device)
        mapping_dev = mapping.to(device)
        out_ref = out.to(device)
        out_fused = out.to(device)

        _reference_chain(
            req_pool_indices=req_pool_ref,
            mapping=mapping_dev,
            out_buf=out_ref,
            valid_bs=valid_bs,
            total_bs=total_bs,
        )
        _fused_state_indices_kernel[(1,)](
            req_pool_fused,
            mapping_dev,
            out_fused,
            valid_bs,
            total_bs,
            BS_UPPER=triton.next_power_of_2(total_bs),
        )
        torch.cuda.synchronize()

        case = f"{total_bs=} {num_padding=} {seed=}"
        # 1. state indices bit-identical over [0, total_bs), sentinels included
        self.assertTrue(
            torch.equal(out_ref[:total_bs], out_fused[:total_bs]),
            f"state indices mismatch ({case}):\n"
            f"  ref   {out_ref[:total_bs].tolist()}\n"
            f"  fused {out_fused[:total_bs].tolist()}",
        )
        # 2. req_pool_indices side effect bit-identical (padded rows zeroed)
        self.assertTrue(
            torch.equal(req_pool_ref[:total_bs], req_pool_fused[:total_bs]),
            f"req_pool_indices mismatch ({case}):\n"
            f"  ref   {req_pool_ref[:total_bs].tolist()}\n"
            f"  fused {req_pool_fused[:total_bs].tolist()}",
        )
        # Explicit re-statement of the contract, independent of the reference:
        self.assertTrue(
            (req_pool_fused[valid_bs:total_bs] == 0).all(),
            f"padded req_pool_indices rows not zeroed ({case})",
        )
        self.assertTrue(
            (out_fused[valid_bs:total_bs] == -1).all(),
            f"padding sentinel rows not -1 ({case})",
        )
        self.assertFalse(
            (out_fused[:total_bs] == _OUT_POISON).any(),
            f"unwritten cells inside [0, total_bs) ({case})",
        )
        # 3. no out-of-range writes past total_bs (BS_UPPER > total_bs masking)
        for name, buf in (("req_pool", req_pool_fused), ("out", out_fused)):
            expected = _GUARD_SENTINEL if name == "req_pool" else _OUT_POISON
            self.assertTrue(
                (buf[total_bs:] == expected).all(),
                f"{name} guard tail clobbered ({case}): {buf[total_bs:].tolist()}",
            )

    def test_matrix(self):
        # Non-power-of-two sizes (7, 33) exercise the BS_UPPER in_range mask;
        # num_padding sweeps none / one / half / all-but-one padded rows.
        for total_bs in (1, 2, 7, 32, 33):
            paddings = sorted(
                {0, 1, total_bs // 2, total_bs - 1} & set(range(total_bs))
            )
            for num_padding in paddings:
                for seed in (0, 1, 2):
                    with self.subTest(
                        total_bs=total_bs, num_padding=num_padding, seed=seed
                    ):
                        self._run_case(
                            total_bs=total_bs, num_padding=num_padding, seed=seed
                        )

    def test_shared_mamba_slots(self):
        # Multiple requests mapping to the same mamba slot (mapping is not
        # injective in general) must gather identically on both paths.
        mapping_const = torch.full((_REQ_POOL_SIZE,), 3, dtype=torch.int32)
        device = torch.device("cuda")
        total_bs, num_padding = 7, 2
        valid_bs = total_bs - num_padding
        req_pool = torch.arange(total_bs + _GUARD, dtype=torch.int64)
        out = torch.full((total_bs + _GUARD,), _OUT_POISON, dtype=torch.int32)

        req_ref, req_fused = req_pool.to(device), req_pool.to(device)
        out_ref, out_fused = out.to(device), out.to(device)
        mapping_dev = mapping_const.to(device)

        _reference_chain(
            req_pool_indices=req_ref,
            mapping=mapping_dev,
            out_buf=out_ref,
            valid_bs=valid_bs,
            total_bs=total_bs,
        )
        _fused_state_indices_kernel[(1,)](
            req_fused,
            mapping_dev,
            out_fused,
            valid_bs,
            total_bs,
            BS_UPPER=triton.next_power_of_2(total_bs),
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out_ref, out_fused))
        self.assertTrue(torch.equal(req_ref, req_fused))


if __name__ == "__main__":
    unittest.main()
