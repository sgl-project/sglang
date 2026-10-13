"""PD decode admission seeds the first decode step's input into FutureMap, as
the device commit of the prefill node's step would have: the input token and,
with speculation, the draft state the prefill node sent and the committed
lengths. The first decode step's worker gathers them like any other."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import DisaggDecode  # noqa: E402
from sglang.srt.managers.overlap_utils import FutureMap  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
)
from sglang.srt.speculative import eagle_disaggregation  # noqa: E402
from sglang.srt.speculative.eagle_info import EagleDraftInput  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOL = 8
HIDDEN = 4


def _req(slot, prompt_len, output_ids, drafted, hidden):
    return SimpleNamespace(
        rid=f"r{slot}",
        kv=SimpleNamespace(req_pool_idx=slot),
        origin_input_ids=list(range(prompt_len)),
        output_ids=list(output_ids),
        return_logprob=False,
        grammar=None,
        is_prefill_only=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        # What the prefill node sent: its draft step's top-1 and hidden state.
        output_topk_p=[0.5],
        output_topk_index=[drafted],
        hidden_states_tensor=torch.full((HIDDEN,), float(hidden)),
        output_dsa_topk_indices=None,
    )


def _scheduler(algo):
    pool = SimpleNamespace(
        req_to_token=torch.zeros((POOL, 16), dtype=torch.int32), device="cpu"
    )
    return SimpleNamespace(
        device="cpu",
        spec_algorithm=algo,
        future_map=FutureMap(torch.device("cpu"), algo, pool),
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(vocab_size=32, hf_config=SimpleNamespace()),
        enable_overlap=True,
    )


class TestPdDecodeSeed(CustomTestCase):
    def setUp(self):
        self.reqs = [
            _req(slot=3, prompt_len=5, output_ids=[11], drafted=21, hidden=1),
            _req(slot=6, prompt_len=9, output_ids=[12, 13], drafted=22, hidden=2),
        ]
        self.slots = torch.tensor([3, 6], dtype=torch.int64)

    def test_non_spec_seeds_the_input_token(self):
        s = _scheduler(SpeculativeAlgorithm.NONE)
        DisaggDecode._seed_first_decode_input(s, self.reqs)
        self.assertEqual(s.future_map.output_tokens_buf[self.slots].tolist(), [11, 13])

    def test_spec_seeds_the_draft_state_the_first_verify_gathers(self):
        s = _scheduler(SpeculativeAlgorithm.EAGLE)
        spec = SimpleNamespace(
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=1,
            enable_multi_layer_eagle=False,
            speculative_use_rejection_sampling=False,
        )
        with (
            patch.object(eagle_disaggregation, "get_spec", return_value=spec),
            patch(
                "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
                return_value=True,
            ),
        ):
            DisaggDecode._seed_first_decode_input(s, self.reqs)

        # The first decode step keys the rows' slots; its worker gathers them.
        seq_lens, seq_lens_cpu, seq_lens_sum = s.future_map.gather_seq_lens(
            self.slots, self.slots
        )
        # Committed: the prompt and every output token but the last, whose KV
        # the first decode step writes.
        self.assertEqual(seq_lens.tolist(), [5, 10])
        self.assertEqual(seq_lens_cpu.tolist(), [5, 10])
        self.assertEqual(seq_lens_sum, 15)

        keyed = EagleDraftInput()
        keyed.future_indices = self.slots
        gathered = s.future_map.gather_draft_input(keyed, self.slots)
        self.assertEqual(gathered.bonus_tokens.tolist(), [11, 13])
        self.assertEqual(gathered.topk_index.tolist(), [[21], [22]])
        self.assertEqual(gathered.topk_p.tolist(), [[0.5], [0.5]])
        self.assertEqual(gathered.hidden_states[:, 0].tolist(), [1.0, 2.0])
        self.assertTrue(s.future_map.cuda_graph_compatible(self.slots))


if __name__ == "__main__":
    unittest.main()
