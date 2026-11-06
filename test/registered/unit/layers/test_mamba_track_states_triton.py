from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.mamba import mamba_state_scatter_triton
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    track_mamba_states_all_layers,
    track_mamba_states_if_needed,
)
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 3
CONV_SHAPE = (3 * 256, 3)
SSM_SHAPE = (4, 64, 64)


def _make_pools(
    pool_size,
    conv_dtype,
    ssm_dtype,
    gen,
    num_layers=NUM_LAYERS,
    conv_shape=CONV_SHAPE,
    ssm_shape=SSM_SHAPE,
):
    conv = torch.randn(
        (num_layers, pool_size) + conv_shape, generator=gen, device="cuda"
    ).to(conv_dtype)
    ssm = torch.randn(
        (num_layers, pool_size) + ssm_shape, generator=gen, device="cuda"
    ).to(ssm_dtype)
    return conv, ssm


def _make_batch(bs, real_bs, pool_size, gen, freed=False):
    """Rows >= real_bs are graph padding: mask False and index 0, like the
    decode graph's zero-padded track slots."""
    perm = torch.randperm(pool_size - 1, generator=gen, device="cuda") + 1
    cache_indices = torch.zeros(bs, dtype=torch.int64, device="cuda")
    track_indices = torch.zeros(bs, dtype=torch.int64, device="cuda")
    cache_indices[:real_bs] = perm[:real_bs]
    track_indices[:real_bs] = perm[real_bs : 2 * real_bs]
    mask = torch.zeros(bs, dtype=torch.bool, device="cuda")
    mask[:real_bs] = torch.rand(real_bs, generator=gen, device="cuda") < 0.5
    if freed and real_bs > 1:
        mask[0] = True
        track_indices[0] = -1
        mask[1] = True
        cache_indices[1] = -1
    return cache_indices, mask, track_indices


def _reference(conv, ssm, cache_indices, mask, track_indices):
    conv, ssm = conv.clone(), ssm.clone()
    for i in range(mask.numel()):
        src, dst = int(cache_indices[i]), int(track_indices[i])
        if bool(mask[i]) and src >= 0 and dst >= 0:
            conv[:, dst] = conv[:, src]
            ssm[:, dst] = ssm[:, src]
    return conv, ssm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestTrackMambaStates(CustomTestCase):
    def _check(self, bs, real_bs, conv_dtype, ssm_dtype, freed=False, **pool_shapes):
        gen = torch.Generator(device="cuda").manual_seed(1000 * bs + real_bs)
        pool_size = 2 * bs + 2
        conv, ssm = _make_pools(pool_size, conv_dtype, ssm_dtype, gen, **pool_shapes)
        cache_indices, mask, track_indices = _make_batch(
            bs, real_bs, pool_size, gen, freed
        )
        ref_conv, ref_ssm = _reference(conv, ssm, cache_indices, mask, track_indices)

        per_conv, per_ssm = conv.clone(), ssm.clone()
        for layer in range(conv.shape[0]):
            track_mamba_states_if_needed(
                per_conv[layer],
                per_ssm[layer],
                cache_indices,
                mask,
                track_indices,
                bs,
                check_freed_slots=freed,
            )
        all_conv, all_ssm = conv.clone(), ssm.clone()
        track_mamba_states_all_layers(
            all_conv,
            all_ssm,
            cache_indices,
            mask,
            track_indices,
            bs,
            check_freed_slots=freed,
        )
        for got_conv, got_ssm in ((per_conv, per_ssm), (all_conv, all_ssm)):
            self.assertTrue(torch.equal(got_conv, ref_conv))
            self.assertTrue(torch.equal(got_ssm, ref_ssm))

    def test_bf16_padded_batches(self):
        for bs in range(1, 41):
            for real_bs in sorted({1, bs // 2 or 1, bs}):
                self._check(bs, real_bs, torch.bfloat16, torch.bfloat16)

    def test_mixed_dtypes(self):
        for bs in (1, 7, 17, 40):
            self._check(bs, bs, torch.bfloat16, torch.float32)

    def test_freed_slots_skipped(self):
        for bs in (2, 9, 33):
            self._check(bs, bs, torch.bfloat16, torch.bfloat16, freed=True)

    def test_persistent_loop_and_partial_chunks(self):
        # With 4 programs each one loops over many (row, layer, chunk) items,
        # and neither state row is a whole number of 32 KB chunks.
        shapes = dict(num_layers=4, conv_shape=(3, 6000), ssm_shape=(2, 129, 130))
        with mock.patch.object(mamba_state_scatter_triton, "_num_sms", return_value=1):
            for ssm_dtype in (torch.bfloat16, torch.float32):
                self._check(9, 6, torch.bfloat16, ssm_dtype, **shapes)
                self._check(9, 9, torch.bfloat16, ssm_dtype, freed=True, **shapes)
        self._check(9, 9, torch.bfloat16, torch.bfloat16, **shapes)

    def test_all_rows_tracked(self):
        gen = torch.Generator(device="cuda").manual_seed(7)
        bs = 40
        conv, ssm = _make_pools(2 * bs + 1, torch.bfloat16, torch.bfloat16, gen)
        cache_indices = torch.arange(bs, device="cuda")
        track_indices = torch.arange(bs, 2 * bs, device="cuda")
        mask = torch.ones(bs, dtype=torch.bool, device="cuda")
        ref_conv, ref_ssm = _reference(conv, ssm, cache_indices, mask, track_indices)
        track_mamba_states_all_layers(conv, ssm, cache_indices, mask, track_indices, bs)
        self.assertTrue(torch.equal(conv, ref_conv))
        self.assertTrue(torch.equal(ssm, ref_ssm))

    def test_cuda_graph_replay_follows_mask(self):
        gen = torch.Generator(device="cuda").manual_seed(3)
        bs = 24
        pool_size = 2 * bs + 2
        conv, ssm = _make_pools(pool_size, torch.bfloat16, torch.bfloat16, gen)
        cache_indices = torch.zeros(bs, dtype=torch.int64, device="cuda")
        track_indices = torch.zeros(bs, dtype=torch.int64, device="cuda")
        mask = torch.zeros(bs, dtype=torch.bool, device="cuda")

        def run():
            track_mamba_states_all_layers(
                conv, ssm, cache_indices, mask, track_indices, bs
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

        for real_bs in (0, 1, 5, bs):
            new_ci, new_mask, new_ti = _make_batch(bs, real_bs, pool_size, gen)
            cache_indices.copy_(new_ci)
            mask.copy_(new_mask)
            track_indices.copy_(new_ti)
            ref_conv, ref_ssm = _reference(
                conv, ssm, cache_indices, mask, track_indices
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(conv, ref_conv))
            self.assertTrue(torch.equal(ssm, ref_ssm))


if __name__ == "__main__":
    unittest.main()
