import unittest

import torch
from args_util import to_v41_args
from dsv41_attention import (
    Attention,
    get_window_topk_idxs,
    sparse_attn,
)
from dsv41_shared import SharedAttentionRuntime
from ref_loader import RefTestCase, randomize_, report, requires_ref, small_args
from test_indexer import TWO_LEVEL


@requires_ref
class TestAttention(RefTestCase):
    def test_sparse_attn(self):
        b, m, h, d, n, topk = 2, 5, 4, 128, 40, 12
        q = torch.randn(b, m, h, d)
        kv = torch.randn(b, n, d)
        sink = torch.randn(h, dtype=torch.float32)
        idxs = torch.randint(0, n, (b, m, topk), dtype=torch.int32)
        idxs[:, :, -3:] = -1
        idxs[0, 1] = -1
        expected = self.kernel.sparse_attn(q, kv, sink, idxs, d**-0.5)
        actual = sparse_attn(q, kv, sink, idxs, d**-0.5)
        report("sparse_attn", actual, expected)
        torch.testing.assert_close(actual, expected, rtol=2**-6, atol=2**-8)

    def test_window_topk_idxs(self):
        for seqlen, start_pos in ((5, 0), (16, 0), (21, 0), (1, 7), (1, 16), (1, 37)):
            expected = self.model.get_window_topk_idxs(16, 2, seqlen, start_pos)
            actual = get_window_topk_idxs(16, 2, seqlen, start_pos, expected.device)
            self.assertEqual(
                actual.tolist(),
                expected.tolist(),
                f"seqlen={seqlen} start_pos={start_pos}",
            )

    def test_layer_stack(self):
        """Window-only layer, two ratio-2 layers (owner + reader), a ratio-1 layer."""
        model = self.model
        args = small_args(model, **TWO_LEVEL)
        v41 = to_v41_args(args)
        pairs = []
        for layer_id in range(args.n_layers):
            ref = model.Attention(layer_id, args)
            randomize_(ref, seed=layer_id)
            ours = Attention(v41, layer_id)
            ours.load_state_dict(ref.state_dict())
            pairs.append((ref, ours))
        shared = SharedAttentionRuntime()
        bsz, prefill_len, total = args.max_batch_size, 21, 27
        x = torch.randn(bsz, total, args.dim)
        steps = [(0, prefill_len)] + [
            (pos, pos + 1) for pos in range(prefill_len, total)
        ]
        worst = 0.0
        for start, end in steps:
            for layer_id, (ref, ours) in enumerate(pairs):
                expected = ref(x[:, start:end], start)
                actual = ours(x[:, start:end], start, shared)
                diff = (
                    actual.float() - expected.float()
                ).abs().amax() / expected.float().abs().amax()
                worst = max(worst, diff.item())
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2**-5,
                    atol=2**-5 * expected.float().abs().amax().item(),
                    msg=f"layer {layer_id} start_pos={start}",
                )
        print(f"attention stack worst max_rel={worst:.3e}")


if __name__ == "__main__":
    unittest.main()
