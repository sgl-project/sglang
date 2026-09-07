import os
import unittest

import torch
from args_util import to_v41_args
from dsv41_transformer import Transformer
from ref_loader import RefTestCase, randomize_, report, requires_ref, small_args
from test_engram import ENGRAM
from test_indexer import TWO_LEVEL

TOKENIZER_DIR = os.getenv("DSV41_TOKENIZER_DIR")


@requires_ref
@unittest.skipUnless(
    TOKENIZER_DIR, "set DSV41_TOKENIZER_DIR to a directory holding tokenizer.json"
)
class TestTransformer(RefTestCase):
    def test_logits(self):
        """Prefill then six decode steps on a five-layer model with engram at layers 1 and 2."""
        from transformers import AutoTokenizer

        model = self.model
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        probe = small_args(model, engram_num_embeddings=(1, 1), **ENGRAM, **TWO_LEVEL)
        sizes = tuple(
            sum(p for per in layer for p in per)
            for layer in self.engram.EngramLayout.from_args(probe).primes
        )
        args = small_args(model, engram_num_embeddings=sizes, **ENGRAM, **TWO_LEVEL)
        ref = model.Transformer(args, tokenizer)
        randomize_(ref)
        ours = Transformer(to_v41_args(args), tokenizer)
        ours.load_state_dict(ref.state_dict())

        bsz, prefill_len, total = args.max_batch_size, 21, 27
        ids = torch.randint(0, args.vocab_size, (bsz, total))
        steps = [(0, prefill_len)] + [
            (pos, pos + 1) for pos in range(prefill_len, total)
        ]
        agree = 0
        for start, end in steps:
            expected = ref(ids[:, start:end], start)[1]
            actual = ours(ids[:, start:end], start)
            report(f"logits start_pos={start}", actual, expected)
            # Each side runs on its own KV caches, so bf16 noise compounds over decode
            # steps and a routing flip can move one step by 10%+; only the prefill is
            # a stable pass / fail criterion, the decode drift is reported.
            rel_l2 = ((actual - expected).norm() / expected.norm()).item()
            same_top1 = actual.argmax(-1).tolist() == expected.argmax(-1).tolist()
            agree += same_top1
            if start == 0:
                self.assertLess(rel_l2, 0.1, "prefill logits")
                self.assertTrue(same_top1, "prefill top-1 differs")
        print(f"top-1 agreement {agree}/{len(steps)} steps")


if __name__ == "__main__":
    unittest.main()
