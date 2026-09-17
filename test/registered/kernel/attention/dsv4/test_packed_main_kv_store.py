import unittest

import torch

from sglang.kernels.ops.attention.dsv4.packed_main_kv import (
    flash_c1_decode_pack_main_kv_fp4,
    flash_c2_prefill_pack_main_kv_fp4,
    pack_dsv41_main_kv_fp4,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    quantize_dsv41_packed_main_kv,
)
from sglang.srt.mem_cache.dsv41_main_kv_layout import (
    PackedMainKVView,
    make_dsv41_packed_main_kv_spec,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _is_sm90():
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability()[0] == 9
    )


def _identity_freqs(rows):
    return torch.ones((rows, 32), dtype=torch.complex64, device="cuda")


def _rope_tail(values, freqs):
    head, tail = values[..., :-64], values[..., -64:]
    tail = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    rotated = torch.view_as_real(tail * freqs).flatten(-2).to(values.dtype)
    return torch.cat((head, rotated), dim=-1)


def _assert_rows_equal(test, storage, spec, slots, values):
    expected_values = torch.zeros(
        (storage.shape[0], spec.page_slots, 512),
        dtype=torch.bfloat16,
        device="cuda",
    )
    for row, slot in enumerate(slots.tolist()):
        if slot > 0:
            expected_values[slot // spec.page_slots, slot % spec.page_slots] = values[
                row
            ]
    expected = PackedMainKVView(quantize_dsv41_packed_main_kv(expected_values), spec)
    actual = PackedMainKVView(storage, spec)
    for row, slot in enumerate(slots.tolist()):
        if slot <= 0:
            continue
        page, offset = divmod(slot, spec.page_slots)
        test.assertTrue(
            torch.equal(actual.payload[page, offset], expected.payload[page, offset])
        )
        test.assertTrue(
            torch.equal(actual.scales[page, offset], expected.scales[page, offset])
        )
        test.assertTrue(
            torch.equal(actual.rope[page, offset], expected.rope[page, offset])
        )


@unittest.skipUnless(_is_sm90(), "requires an SM90 CUDA GPU")
class TestPackedMainKVStore(CustomTestCase):
    def test_all_nibble_pairs(self):
        spec = make_dsv41_packed_main_kv_spec(128)
        num_rows = 10
        storage = torch.zeros((1, spec.page_bytes), dtype=torch.uint8, device="cuda")
        view = PackedMainKVView(storage, spec)
        values = torch.zeros((num_rows, 512), dtype=torch.bfloat16, device="cuda")
        magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

        def value_for_code(code):
            magnitude = magnitudes[code & 7]
            if code == 8:
                return -0.125
            return -magnitude if code & 8 else magnitude

        for pair in range(256):
            block = pair
            row, block_in_row = divmod(block, 28)
            base = block_in_row * 16
            values[row, base] = value_for_code(pair & 0xF)
            values[row, base + 1] = value_for_code(pair >> 4)
            values[row, base + 15] = 6.0

        slots = torch.arange(1, num_rows + 1, dtype=torch.int32, device="cuda")
        pack_dsv41_main_kv_fp4(
            latent=values,
            freqs_cis=_identity_freqs(num_rows),
            slots=slots,
            view=view,
        )
        for pair in range(256):
            row, block_in_row = divmod(pair, 28)
            self.assertEqual(int(view.payload[0, row + 1, block_in_row * 8]), pair)

    def test_c1_and_c2_entry_points_match_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(11)
        for page_slots, entry in (
            (256, flash_c1_decode_pack_main_kv_fp4),
            (128, flash_c2_prefill_pack_main_kv_fp4),
        ):
            with self.subTest(page_slots=page_slots):
                spec = make_dsv41_packed_main_kv_spec(page_slots)
                storage = torch.zeros(
                    (2, spec.page_bytes), dtype=torch.uint8, device="cuda"
                )
                view = PackedMainKVView(storage, spec)
                values = torch.randn(
                    (4, 512),
                    generator=generator,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                angles = torch.randn(
                    (4, 32), generator=generator, dtype=torch.float32, device="cuda"
                )
                freqs = torch.polar(torch.ones_like(angles), angles)
                slots = torch.tensor(
                    [1, page_slots - 1, page_slots, 0],
                    dtype=torch.int64,
                    device="cuda",
                )
                entry(
                    latent=values,
                    freqs_cis=freqs,
                    slots=slots,
                    view=view,
                )
                _assert_rows_equal(
                    self, storage, spec, slots, _rope_tail(values, freqs)
                )
                self.assertEqual(int(view.payload[0, 0].count_nonzero()), 0)
                self.assertEqual(int(view.scales[0, 0].count_nonzero()), 0)
                self.assertEqual(int(view.rope[0, 0].count_nonzero()), 0)

    def test_e2m1_codes_midpoints_and_reserved_bytes(self):
        spec = make_dsv41_packed_main_kv_spec(128)
        storage = torch.full(
            (1, spec.page_bytes), 0xFF, dtype=torch.uint8, device="cuda"
        )
        view = PackedMainKVView(storage, spec)
        values = torch.zeros((1, 512), dtype=torch.bfloat16, device="cuda")
        values[0, :16] = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.bfloat16,
            device="cuda",
        )
        values[0, 16:24] = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0],
            dtype=torch.bfloat16,
            device="cuda",
        )
        values[0, 31] = 6.0
        slots = torch.tensor([1], dtype=torch.int32, device="cuda")

        pack_dsv41_main_kv_fp4(
            latent=values,
            freqs_cis=_identity_freqs(1),
            slots=slots,
            view=view,
        )
        _assert_rows_equal(self, storage, spec, slots, values)
        self.assertEqual(int(view.scales[0, 1, 28:].count_nonzero()), 0)

    def test_debug_flag_reports_non_finite_input(self):
        spec = make_dsv41_packed_main_kv_spec(128)
        storage = torch.zeros((1, spec.page_bytes), dtype=torch.uint8, device="cuda")
        values = torch.zeros((1, 512), dtype=torch.bfloat16, device="cuda")
        values[0, 7] = float("inf")
        error_flag = torch.zeros((1,), dtype=torch.int32, device="cuda")

        pack_dsv41_main_kv_fp4(
            latent=values,
            freqs_cis=_identity_freqs(1),
            slots=torch.tensor([1], dtype=torch.int64, device="cuda"),
            view=PackedMainKVView(storage, spec),
            error_flag=error_flag,
        )
        self.assertEqual(error_flag.item(), 1)

    def test_graph_replay_refreshes_inputs(self):
        spec = make_dsv41_packed_main_kv_spec(128)
        storage = torch.zeros((1, spec.page_bytes), dtype=torch.uint8, device="cuda")
        view = PackedMainKVView(storage, spec)
        latent = torch.zeros((1, 512), dtype=torch.bfloat16, device="cuda")
        freqs = _identity_freqs(1)
        slots = torch.tensor([1], dtype=torch.int32, device="cuda")

        pack_dsv41_main_kv_fp4(latent=latent, freqs_cis=freqs, slots=slots, view=view)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack_dsv41_main_kv_fp4(
                latent=latent, freqs_cis=freqs, slots=slots, view=view
            )

        latent.fill_(6.0)
        graph.replay()
        torch.cuda.synchronize()
        first = view.payload[0, 1].clone()

        latent.fill_(-6.0)
        graph.replay()
        torch.cuda.synchronize()
        second = view.payload[0, 1].clone()
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
