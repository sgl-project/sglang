import unittest

import torch
from dsv41_compressor import Compressor
from ref_loader import RefTestCase, randomize_, requires_ref, small_args


@requires_ref
class TestCompressor(RefTestCase):
    def _run(self, layer_id):
        model = self.model
        args = small_args(model)
        ref = model.Compressor(args, layer_id)
        randomize_(ref)
        ours = Compressor(
            dim=args.dim,
            head_dim=args.head_dim,
            compress_ratio=args.compress_ratios[layer_id],
            norm_eps=args.norm_eps,
            max_batch_size=args.max_batch_size,
        )
        ours.load_state_dict(ref.state_dict())

        prefill_len = 7
        x = torch.randn(args.max_batch_size, prefill_len + 6, args.dim)
        steps = [(0, x[:, :prefill_len])] + [
            (pos, x[:, pos : pos + 1]) for pos in range(prefill_len, x.size(1))
        ]
        for start_pos, chunk in steps:
            expected = ref(chunk, start_pos)
            actual = ours(chunk, start_pos)
            self.assertEqual(actual is None, expected is None, f"start_pos={start_pos}")
            if expected is not None:
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2**-7,
                    atol=1e-6,
                    msg=f"start_pos={start_pos}",
                )

    def test_ratio_2(self):
        self._run(1)

    def test_ratio_1(self):
        self._run(2)


if __name__ == "__main__":
    unittest.main()
