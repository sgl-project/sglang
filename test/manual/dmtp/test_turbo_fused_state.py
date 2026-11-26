"""Turbo request-cache parity and invalidation; no sampling/math changes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.speculative.dmtp_qsa_state import PrefetchedT1State
from sglang.srt.speculative.native_mtp_bridge import NativeMTPBridge


class TurboFusedStateTest(unittest.TestCase):
    def bits(self, value):
        dense = torch.empty(value.shape, dtype=value.dtype, device=value.device)
        return dense.copy_(value).view(torch.uint8)

    def fixture(self, fused, device="cpu", bs=3):
        bridge = NativeMTPBridge.__new__(NativeMTPBridge)
        temp = torch.arange(bs, device=device, dtype=torch.float32) / 10 + 0.7
        mask = torch.arange(bs, device=device) % 2 == 0
        tail = SimpleNamespace(
            temperatures=temp,
            greedy_mask=mask,
            signature=lambda: (),
            initial_signature=(),
        )
        c = SimpleNamespace(two_graphs=True, mtp_tail=tail)
        c.validate_sampling = lambda batch: setattr(
            c, "selector_sampling", batch.sampling_info
        )
        bridge.controller, bridge.fused_t1_state = c, fused
        size, width = 40, 2048
        bridge.hidden_by_req = torch.empty(size, width, device=device)
        bridge.anchor_by_req = torch.full((size,), -1, dtype=torch.int64, device=device)
        bridge.token_by_req = torch.full_like(bridge.anchor_by_req, -1)
        bridge.proposal_valid = torch.zeros(size, dtype=torch.bool, device=device)
        slots = torch.arange(1, bs + 1, device=device)
        values = (
            slots[:, None] + 100,
            torch.arange(bs * 16, device=device).view(bs, 1, 16),
            torch.rand(bs, 1, 16, device=device),
            torch.randn(bs, 8, width, device=device, dtype=torch.bfloat16),
        )
        if fused:
            bridge.proposal_state = PrefetchedT1State(size)
        else:
            bridge.proposal_buffers = tuple(
                v.new_empty((size, *v.shape[1:])) for v in values
            )
            bridge.proposal_temperatures = torch.empty(size, device=device)
            bridge.proposal_greedy = torch.empty(size, dtype=torch.bool, device=device)
        lengths, anchors = slots + 2048, slots + 30000
        hidden = torch.randn(bs, width, device=device)
        bridge.seed(slots, hidden, lengths, anchors)
        return bridge, slots, values, lengths, anchors, hidden

    def read(self, bridge, slots, lengths, anchors, order=None):
        order = (
            torch.arange(len(slots) - 1, -1, -1, device=slots.device)
            if order is None
            else order
        )
        tail = bridge.controller.mtp_tail
        batch = SimpleNamespace(
            req_pool_indices=slots[order],
            seq_lens=lengths[order],
            sampling_info=SimpleNamespace(temperatures=tail.temperatures[order, None]),
        )
        block = torch.full(
            (len(slots), 16), -7, device=slots.device, dtype=torch.int64
        )[:, ::2]
        block[:, 0] = anchors[order]
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.resolve_greedy_mask",
            return_value=tail.greedy_mask[order],
        ):
            bridge.t1(batch, block)
        c = bridge.controller
        return (block[:, 1:2], *c.t1_sparse, c.prepared_embeds), batch, block

    def exercise(self, device):
        for bs in range(1, 33):
            new, slots, values, lengths, anchors, hidden = self.fixture(
                True, device, bs
            )
            old, *_ = self.fixture(False, device, bs)
            old.seed(slots, hidden, lengths, anchors)
            # Check exact byte transport, not approximate arithmetic, including
            # signed zero/NaN payloads in state fields.
            values[2].view(torch.int32).flatten()[:2] = torch.tensor(
                [0x7FC01234, -2147483648], device=device, dtype=torch.int32
            )
            reference = tuple(v.clone() for v in values)
            old.seed_proposal(slots, values)
            new.seed_proposal(slots, values, lengths, anchors)
            for v in values:
                v.zero_()
            actual, batch, block = self.read(new, slots, lengths, anchors)
            baseline, _, _ = self.read(old, slots, lengths, anchors)
            for a, b, r in zip(actual, baseline, reference):
                self.assertTrue(torch.equal(self.bits(a), self.bits(b)))
                self.assertTrue(torch.equal(self.bits(a), self.bits(r.flip(0))))
            self.assertIsNone(new.controller.hidden)
            self.assertTrue(torch.all(block[:, 2:] == -7))
            # Existing output batches must not alias a subsequent graph/state write.
            new.seed_proposal(slots, values, lengths, anchors)
            for a, r in zip(actual[1:], reference[1:]):
                self.assertTrue(torch.equal(self.bits(a), self.bits(r.flip(0))))
            new.seed(slots[:1], hidden[:1], lengths[:1], anchors[:1])
            self.assertFalse(new.proposal_valid[slots[0]].item())
            tail = new.controller.mtp_tail
            blocks = torch.zeros(bs, 8, dtype=torch.int64, device=device)
            blocks[:, 0] = anchors
            blocks[:, 1] = -99
            args = (
                slots,
                slots.clone(),
                lengths,
                blocks,
                tail.temperatures,
                tail.greedy_mask,
            )
            _, valid = new.proposal_state._checked_values(*args, new.proposal_valid)
            self.assertFalse(valid[0].item())
            self.assertTrue(valid[1:].all().item())
            self.assertEqual(blocks[0, 1].item(), -99)
            # Prefill's no-explicit-length call must restore a recycled slot.
            new.seed_proposal(slots[:1], tuple(v[:1] for v in values))
            _, valid = new.proposal_state._checked_values(*args, new.proposal_valid)
            self.assertTrue(valid.all().item())

    def test_cpu_byte_exact_b1_b32_reorder_and_reuse(self):
        self.exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel requires a GPU")
    def test_cuda_byte_exact_b1_b32_reorder_and_reuse(self):
        self.exercise("cuda")

    def test_cpu_controller_rejects_stale_state_and_weight_change(self):
        for field in (
            "valid",
            "length",
            "anchor",
            "temp",
            "nan_temp",
            "greedy",
            "weight",
        ):
            with self.subTest(field=field):
                b, slots, values, lengths, anchors, _ = self.fixture(True)
                b.seed_proposal(slots, values)
                if field == "valid":
                    b.proposal_valid[slots[0]] = False
                elif field == "length":
                    lengths[0] += 1
                elif field == "anchor":
                    anchors[0] += 1
                elif field == "temp":
                    b.controller.mtp_tail.temperatures[0] += 1
                elif field == "nan_temp":
                    b.controller.mtp_tail.temperatures[0] = float("nan")
                elif field == "greedy":
                    b.controller.mtp_tail.greedy_mask[0].logical_not_()
                else:
                    b.controller.mtp_tail.signature = lambda: ("changed",)
                with self.assertRaises(RuntimeError):
                    self.read(b, slots, lengths, anchors)

    def test_valid_storage_layout_rejected(self):
        b, slots, values, lengths, anchors, _ = self.fixture(True)
        b.seed_proposal(slots, values)
        blocks = torch.zeros(3, 8, dtype=torch.int64)
        blocks[:, 0] = anchors
        t = b.controller.mtp_tail
        for bad in (
            torch.ones(40),
            torch.ones(39, dtype=torch.bool),
            torch.ones(80, dtype=torch.bool)[::2],
        ):
            with self.assertRaises(ValueError):
                b.proposal_state.load_checked(
                    slots, slots, lengths, blocks, t.temperatures, t.greedy_mask, bad
                )


if __name__ == "__main__":
    unittest.main()
