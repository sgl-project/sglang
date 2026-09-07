import unittest

import torch
from args_util import to_v41_args
from dsv41_block import Block
from dsv41_hc import make_identity_pre_mix
from dsv41_shared import SharedAttentionRuntime
from ref_loader import RefTestCase, randomize_, requires_ref, small_args
from test_indexer import TWO_LEVEL


@requires_ref
class TestBlock(RefTestCase):
    def test_layer_stack(self):
        model = self.model
        args = small_args(model, **TWO_LEVEL)
        v41 = to_v41_args(args)
        pairs = []
        for layer_id in range(args.n_layers):
            ref = model.Block(layer_id, args)
            randomize_(ref, seed=layer_id)
            ours = Block(v41, layer_id)
            ours.load_state_dict(ref.state_dict())
            pairs.append((ref, ours))
        shared = SharedAttentionRuntime()
        routed = {}
        for layer_id, (ref, ours) in enumerate(pairs):
            for side, module in (("ref", ref), ("ours", ours)):
                module.ffn.gate.register_forward_hook(
                    lambda m, i, o, key=(side, layer_id): routed.__setitem__(
                        key, o[1].sort(-1).values
                    )
                )
        bsz, prefill_len, total = args.max_batch_size, 21, 27
        x_all = torch.randn(bsz, total, args.hc_mult, args.dim)
        steps = [(0, prefill_len)] + [
            (pos, pos + 1) for pos in range(prefill_len, total)
        ]
        worst, flips = 0.0, 0
        for start, end in steps:
            # Teacher forcing: both sides consume the reference output of the previous
            # layer, so each comparison isolates one layer's own error. A token whose
            # expert routing flipped between the two sides is excluded: bf16 noise at a
            # top-k boundary swaps experts, which is a discontinuity, not a mismatch.
            x = x_all[:, start:end]
            pre = make_identity_pre_mix(x, args.hc_mult)
            for layer_id, (ref, ours) in enumerate(pairs):
                x_ref, pre_ref = ref(x, start, pre, None)
                x_ours, pre_ours = ours(x, start, pre, None, shared)
                same_route = (
                    routed[("ref", layer_id)] == routed[("ours", layer_id)]
                ).all(-1)
                flips += (~same_route).sum().item()
                keep = same_route.view(bsz, end - start)
                # fp8 activation quantization turns bf16-level input noise into a few
                # percent on single elements, so the whole-tensor error carries the check.
                err = (x_ours.float() - x_ref.float())[keep]
                max_rel = err.abs().amax().item() / x_ref.float().abs().amax().item()
                rel_l2 = (err.norm() / x_ref.float()[keep].norm()).item()
                worst = max(worst, max_rel)
                if start == 0:
                    print(
                        f"layer {layer_id} prefill: max_rel={max_rel:.3e} rel_l2={rel_l2:.3e}"
                    )
                self.assertLess(max_rel, 2**-4, f"layer {layer_id} start_pos={start}")
                self.assertLess(rel_l2, 2**-6, f"layer {layer_id} start_pos={start}")
                torch.testing.assert_close(pre_ours, pre_ref, rtol=2**-6, atol=2**-6)
                x, pre = x_ref, pre_ref
        print(f"block stack worst max_rel={worst:.3e}, routing flips={flips}")


if __name__ == "__main__":
    unittest.main()
