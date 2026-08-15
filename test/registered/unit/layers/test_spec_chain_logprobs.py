"""Per-request logprob params must expand in step with the chain row layout.

compute_spec_v2_logprobs expands top_logprobs_nums / token_ids_logprobs by
repeating each request's entry ``max_accept`` times, and the result processor
reads the resulting lists back at flat index ``i * chain_len + j``. Chain verify
(DFLASH / DSPARK) sets ``max_accept = chain_len``, so any drift between that
expansion and the row layout silently pairs one request's top-k / probe ids
with another request's distribution.

The server-level kits do not close this gap: SpecLogprobKit's decode-vs-prefill
comparison sends one request at a time (uniform params), and its ragged
token_ids_logprob case only asserts the server does not crash. Heterogeneous
params in a single batch are checked here.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import compute_spec_v2_chain_logprobs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BS = 3
CHAIN_LEN = 4
VOCAB = 17
# Heterogeneous per-request params; uniform ones hide the misalignment.
TOP_LOGPROBS_NUMS = [2, 0, 3]
TOKEN_IDS_LOGPROBS = [[1, 5], None, [0, 2, 9]]


class TestSpecChainLogprobs(CustomTestCase):
    def test_top_and_token_ids_lists_are_row_major(self):
        torch.manual_seed(0)
        logits = torch.randn(BS * CHAIN_LEN, VOCAB)
        out_tokens = torch.randint(0, VOCAB, (BS, CHAIN_LEN), dtype=torch.int64)
        batch = SimpleNamespace(
            seq_lens=torch.arange(BS),
            sampling_info=SimpleNamespace(is_all_greedy=True, temperatures=None),
            top_logprobs_nums=TOP_LOGPROBS_NUMS,
            token_ids_logprobs=TOKEN_IDS_LOGPROBS,
        )
        logits_output = SimpleNamespace(next_token_logits=logits)

        compute_spec_v2_chain_logprobs(batch, logits_output, out_tokens)

        reference = torch.nn.functional.log_softmax(logits, dim=-1)
        self.assertEqual(len(logits_output.next_token_top_logprobs_val), BS * CHAIN_LEN)
        self.assertEqual(
            len(logits_output.next_token_token_ids_logprobs_val), BS * CHAIN_LEN
        )
        for i in range(BS):
            for j in range(CHAIN_LEN):
                flat = i * CHAIN_LEN + j
                top_idx = logits_output.next_token_top_logprobs_idx[flat].tolist()
                expected_top = (
                    reference[flat].topk(TOP_LOGPROBS_NUMS[i]).indices.tolist()
                )
                self.assertEqual(top_idx, expected_top, msg=f"req={i} pos={j}")

                probe_ids = TOKEN_IDS_LOGPROBS[i]
                got_ids = logits_output.next_token_token_ids_logprobs_idx[flat]
                got_vals = logits_output.next_token_token_ids_logprobs_val[flat]
                if probe_ids is None:
                    self.assertEqual(got_ids, [], msg=f"req={i} pos={j}")
                    continue
                self.assertEqual(got_ids, probe_ids, msg=f"req={i} pos={j}")
                for k, tid in enumerate(probe_ids):
                    self.assertAlmostEqual(
                        got_vals[k].item(),
                        reference[flat, tid].item(),
                        places=5,
                        msg=f"req={i} pos={j} tid={tid}",
                    )


if __name__ == "__main__":
    unittest.main()
