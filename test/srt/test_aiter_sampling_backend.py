import unittest
from typing import Optional

import torch

from sglang.srt.layers import sampler as sampler_mod
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import is_hip


@unittest.skipUnless(is_hip(), "AITER sampling requires ROCm")
class TestAiterSamplingOpsStandalone(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def _make_sampling_info(
        self,
        batch_size: int,
        vocab_size: int,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
    ) -> SamplingBatchInfo:
        need_top_k = top_k is not None and top_k > 0
        need_top_p = top_p is not None and top_p < 1.0
        need_min_p = min_p is not None and min_p > 0.0

        temperatures = torch.ones(batch_size, 1, device=self.device, dtype=torch.float)
        top_ps = torch.full((batch_size,), float(top_p or 1.0), device=self.device)
        top_ks = torch.full(
            (batch_size,), int(top_k or 0), device=self.device, dtype=torch.int32
        )
        min_ps = torch.full((batch_size,), float(min_p or 0.0), device=self.device)

        return SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=False,
            need_top_p_sampling=need_top_p,
            need_top_k_sampling=need_top_k,
            need_min_p_sampling=need_min_p,
            vocab_size=vocab_size,
            grammars=None,
            vocab_mask=None,
            apply_mask_func=None,
            penalizer_orchestrator=None,
            acc_linear_penalties=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            sampling_seed=None,
            device="cuda",
            logit_bias=None,
        )

    def test_greedy_local(self):
        logits = torch.tensor(
            [[0.1, 0.2, 0.9, 0.3, -1.0], [0.7, 0.2, -0.1, 0.6, 0.5]],
            device=self.device,
            dtype=torch.float,
        )
        _ops = sampler_mod._load_aiter_ops()
        # greedy_sample is provided via the aiter module APIs
        out_buf = torch.empty(logits.size(0), dtype=torch.int32, device=self.device)
        sampler_mod._aiter_module.greedy_sample(out_buf, logits)
        out = out_buf.to(torch.long)
        self.assertTrue(torch.equal(out.cpu(), torch.tensor([2, 0], dtype=torch.long)))

    def test_top_p_sampling_membership(self):
        probs = torch.tensor(
            [[0.55, 0.25, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05]],
            device=self.device,
            dtype=torch.float,
        )
        sampling_info = self._make_sampling_info(batch_size=2, vocab_size=5, top_p=0.5)
        _ops = sampler_mod._load_aiter_ops()
        for _ in range(10):
            ids = _ops.top_p_sampling_from_probs(
                probs,
                None,
                sampling_info.top_ps,
                1.0,
                True,
            ).to(torch.long)
            self.assertTrue(torch.all(ids == 0))

    def test_top_k_sampling_membership(self):
        probs = torch.tensor(
            [[0.5, 0.3, 0.1, 0.05, 0.05], [0.45, 0.4, 0.05, 0.05, 0.05]],
            device=self.device,
            dtype=torch.float,
        )
        sampling_info = self._make_sampling_info(batch_size=2, vocab_size=5, top_k=2)
        _ops = sampler_mod._load_aiter_ops()
        for _ in range(20):
            ids = _ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                sampling_info.top_ks,
                int(sampling_info.top_ks[0].item()),
                None,
                1.0,
                False,
            ).to(torch.long)
            self.assertTrue(torch.all(torch.isin(ids.cpu(), torch.tensor([0, 1]))))

    def test_top_k_top_p_combo(self):
        # With top_k=1 and top_p=0.9, selection should be strictly the top-1 index
        probs = torch.tensor(
            [[0.6, 0.2, 0.1, 0.05, 0.05], [0.7, 0.1, 0.1, 0.05, 0.05]],
            device=self.device,
            dtype=torch.float,
        )
        sampling_info = self._make_sampling_info(
            batch_size=2, vocab_size=5, top_k=1, top_p=0.9
        )
        _ops = sampler_mod._load_aiter_ops()
        for _ in range(5):
            ids = _ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                sampling_info.top_ks,
                int(sampling_info.top_ks[0].item()),
                sampling_info.top_ps,
                float(sampling_info.top_ps[0].item()),
                True,
            ).to(torch.long)
            self.assertTrue(torch.all(ids == 0))

    def test_min_p_fallback_executes(self):
        # Ensure min_p path runs without error and returns valid token ids
        probs = torch.tensor(
            [[0.4, 0.3, 0.2, 0.05, 0.05], [0.35, 0.3, 0.2, 0.1, 0.05]],
            device=self.device,
            dtype=torch.float,
        )
        sampling_info = self._make_sampling_info(batch_size=2, vocab_size=5, min_p=0.2)
        _ops = sampler_mod._load_aiter_ops()
        ids = _ops.top_k_top_p_sampling_from_probs(
            probs,
            None,
            None,
            0,
            None,
            1.0,
            False,
        ).to(torch.long)
        self.assertEqual(ids.shape, (2,))
        self.assertTrue(ids.dtype == torch.long)
        self.assertTrue(torch.all((ids >= 0) & (ids < 5)))


if __name__ == "__main__":
    unittest.main()
