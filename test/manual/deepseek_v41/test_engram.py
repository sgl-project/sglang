import os
import unittest

import dsv41_engram as ours
import torch
from args_util import to_v41_args
from ref_loader import (
    RefTestCase,
    assert_equal,
    randomize_,
    report,
    requires_ref,
    small_args,
)

TOKENIZER_DIR = os.getenv("DSV41_TOKENIZER_DIR")

ENGRAM = dict(
    engram_layer_ids=(1, 2),
    engram_vocab_size=100,
    engram_max_ngram_size=3,
    engram_n_heads=2,
    engram_head_dim=64,
    engram_pad_id=2,
    engram_compressed_vocab_size=99092,
    vocab_size=129280,
)


@requires_ref
@unittest.skipUnless(
    TOKENIZER_DIR, "set DSV41_TOKENIZER_DIR to a directory holding tokenizer.json"
)
class TestEngram(RefTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def _layouts(self):
        model = self.model
        probe = small_args(model, engram_num_embeddings=(1, 1), **ENGRAM)
        sizes = tuple(
            sum(p for per in layer for p in per)
            for layer in self.engram.EngramLayout.from_args(probe).primes
        )
        args = small_args(model, engram_num_embeddings=sizes, **ENGRAM)
        ref_layout = self.engram.EngramLayout.from_args(args)
        our_layout = ours.EngramLayout.from_args(to_v41_args(args))
        self.assertEqual(our_layout.primes, ref_layout.primes)
        self.assertEqual(our_layout.num_embeddings, ref_layout.num_embeddings)
        return args, ref_layout, our_layout

    def test_hash_state(self):
        args, ref_layout, our_layout = self._layouts()
        ref = self.engram.NgramHashState(args, ref_layout, self.tokenizer)
        mine = ours.NgramHashState(to_v41_args(args), our_layout, self.tokenizer)
        assert_equal(mine.token_map, ref.token_map, msg="token_map")
        assert_equal(mine.multipliers, ref.multipliers, msg="multipliers")
        bsz, prefill_len, total = args.max_batch_size, 9, 13
        ids = torch.randint(0, args.vocab_size, (bsz, total))
        mask = torch.ones(bsz, total, dtype=torch.bool)
        mask[0, 3:5] = False
        steps = [(0, prefill_len)] + [
            (pos, pos + 1) for pos in range(prefill_len, total)
        ]
        for start, end in steps:
            expected = ref(ids[:, start:end], start, mask[:, start:end])
            actual = mine(ids[:, start:end], start, mask[:, start:end])
            assert_equal(actual, expected, msg=f"start_pos={start}")
            self.assertTrue((actual[:, :, 0] < args.engram_num_embeddings[0]).all())

    def test_engram_layer(self):
        args, ref_layout, our_layout = self._layouts()
        ref = self.model.Engram(args, 2, ref_layout)
        randomize_(ref)
        mine = ours.Engram(to_v41_args(args), 2, our_layout)
        mine.load_state_dict(ref.state_dict())
        bsz, seqlen = args.max_batch_size, 6
        n_hash_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        hash_ids = torch.randint(
            0, args.engram_num_embeddings[1], (bsz, seqlen, n_hash_cols)
        )
        x = torch.randn(bsz, seqlen, args.hc_mult, args.dim)
        mask = torch.ones(bsz, seqlen, dtype=torch.bool)
        mask[1, 2] = False
        for token_mask in (None, mask):
            expected = ref(x, hash_ids, token_mask)
            actual = mine(x, hash_ids, token_mask)
            report("engram", actual, expected)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2**-6,
                atol=2**-6 * expected.float().abs().amax().item(),
            )


if __name__ == "__main__":
    unittest.main()
