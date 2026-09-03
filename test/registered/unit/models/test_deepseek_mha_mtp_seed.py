"""Hermetic unit tests for the DSA MHA MTP-seed publish path.

`_forward_dsa_indexer_for_mha` publishes the GLM-5.2 MTP IndexShare seed from
the MHA prefill path. `Indexer.forward_cuda` returns None whenever the
attention backend exposes no indexer metadata -- every non-DSA backend does,
e.g. an explicit `--attention-backend flashinfer` on a DSA model -- so the seed
publish has to degrade to the -1 "no seed" sentinel instead of killing the
scheduler.

Pure Python (no GPU, no model weights): only the attention-backend lookup is
faked, `Indexer.forward_cuda` and `_forward_dsa_indexer_for_mha` are real.
Runs on any PR-CI lane.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    _forward_dsa_indexer_for_mha,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

INDEX_TOPK = 4


def _fake_forward_batch(seed_buf, select=None):
    return SimpleNamespace(
        spec_info=SimpleNamespace(
            dsa_seed_topk_capture=seed_buf,
            dsa_seed_topk_select=select,
        )
    )


def _call(indexer, forward_batch):
    _forward_dsa_indexer_for_mha(
        indexer,
        hidden_states=torch.zeros((2, 8)),
        q_lora=torch.zeros((2, 8)),
        positions=torch.zeros((2,), dtype=torch.int64),
        forward_batch=forward_batch,
        layer_id=0,
    )


class TestDsaMhaMtpSeed(CustomTestCase):
    def test_seed_is_invalidated_without_indexer_metadata(self):
        # A backend without indexer metadata (any non-DSA backend) makes the
        # real Indexer.forward_cuda return None. The seed buffer must come back
        # all -1, not keep the previous batch's values and not raise.
        seed_buf = torch.full((2, INDEX_TOPK), 7, dtype=torch.int32)
        # forward_cuda returns before it touches `self`, so None binds fine.
        indexer = partial(Indexer.forward_cuda, None)
        with mock.patch.object(
            dsa_indexer, "get_attn_backend", return_value=AttentionBackend()
        ):
            _call(indexer, _fake_forward_batch(seed_buf))
        self.assertTrue(torch.all(seed_buf == -1))

    def test_seed_is_published_when_indexer_returns_indices(self):
        topk_indices = torch.arange(3 * INDEX_TOPK, dtype=torch.int32).view(
            3, INDEX_TOPK
        )
        seed_buf = torch.full((2, INDEX_TOPK), -1, dtype=torch.int32)
        select = torch.tensor([0, 2])
        _call(
            lambda **kwargs: topk_indices,
            _fake_forward_batch(seed_buf, select=select),
        )
        self.assertTrue(torch.equal(seed_buf, topk_indices[select]))


if __name__ == "__main__":
    unittest.main()
