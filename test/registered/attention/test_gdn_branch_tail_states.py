"""The optional branch copy must capture real conv and recurrent states."""

import unittest
from types import SimpleNamespace

import torch
from test_gdn_prefill_strided_pool import (
    CONV,
    DIM,
    HEAD_DIM,
    LAYERS,
    V_HEADS,
    _backend,
    _batch,
    _layer,
)

from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGDNBranchTailStates(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(_mamba_cache_chunk_size=64)
        override.install()
        self.addCleanup(override.restore)

    def _run(self, conv, ssm, inputs, lens, prefixes, slots, branch=None):
        caches = [
            SimpleNamespace(conv=[conv[layer]], temporal=ssm[layer]) for layer in LAYERS
        ]
        device_slots = torch.tensor(slots, device="cuda", dtype=torch.int32)
        backend = _backend(
            caches,
            device_slots,
            LinearAttnKernelBackend.TRITON,
            IdSpace(key=("state", None)),
        )
        track = {len(lens) - 1: lens[-1]}
        tracks = torch.tensor([4, 5][: len(lens)], device="cuda", dtype=torch.int64)
        batch = _batch(lens, prefixes, track, tracks)
        batch.gdn_branch_snapshot = branch
        backend.init_forward_metadata(batch)
        outputs = []
        for layer in LAYERS:
            outputs.append(
                backend.forward_extend(
                    _layer(layer), batch, *(tensor.clone() for tensor in inputs)
                ).clone()
            )
        return outputs, int(tracks[-1])

    def test_branch_and_tail_match_independent_prefix_forwards(self):
        for cached_prefix in (0, 1024):
            with self.subTest(cached_prefix=cached_prefix):
                gen = torch.Generator(device="cuda").manual_seed(41)
                conv = (
                    torch.randn(
                        len(LAYERS),
                        8,
                        DIM,
                        CONV - 1,
                        generator=gen,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.1
                )
                ssm = (
                    torch.randn(
                        len(LAYERS),
                        8,
                        V_HEADS,
                        HEAD_DIM,
                        HEAD_DIM,
                        generator=gen,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.01
                )
                inputs = tuple(
                    torch.randn(
                        649, width, generator=gen, device="cuda", dtype=torch.bfloat16
                    )
                    * 0.1
                    for width in (DIM, V_HEADS, V_HEADS)
                )
                plain_conv, plain_ssm = conv.clone(), ssm.clone()
                plain_outputs, tail_slot = self._run(
                    plain_conv,
                    plain_ssm,
                    inputs,
                    [137, 512],
                    [0, cached_prefix],
                    [2, 3],
                )
                actual_conv, actual_ssm = conv.clone(), ssm.clone()
                outputs, actual_tail = self._run(
                    actual_conv,
                    actual_ssm,
                    inputs,
                    [137, 512],
                    [0, cached_prefix],
                    [2, 3],
                    (1, cached_prefix + 128, torch.tensor([6], device="cuda")),
                )
                self.assertEqual(actual_tail, tail_slot)
                for actual, plain in zip(outputs, plain_outputs):
                    torch.testing.assert_close(actual, plain, rtol=0, atol=0)
                # Exporting an extra snapshot must leave live states identical.
                torch.testing.assert_close(
                    actual_conv[:, [2, 3]], plain_conv[:, [2, 3]], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual_ssm[:, [2, 3]], plain_ssm[:, [2, 3]], rtol=0, atol=0
                )
                # Native aligned tail is still captured, in a separate slot.
                torch.testing.assert_close(
                    actual_conv[:, actual_tail],
                    plain_conv[:, tail_slot],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_ssm[:, actual_tail], plain_ssm[:, tail_slot], rtol=0, atol=0
                )
                ref_conv, ref_ssm = conv.clone(), ssm.clone()
                prefix_inputs = tuple(tensor[137 : 137 + 128] for tensor in inputs)
                self._run(ref_conv, ref_ssm, prefix_inputs, [128], [cached_prefix], [3])
                torch.testing.assert_close(
                    actual_conv[:, 6], ref_conv[:, 3], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual_ssm[:, 6], ref_ssm[:, 3], rtol=0, atol=0
                )
                # Slots outside the requests and explicit destination remain intact.
                torch.testing.assert_close(
                    actual_conv[:, [0, 1, 7]], conv[:, [0, 1, 7]], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual_ssm[:, [0, 1, 7]], ssm[:, [0, 1, 7]], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
