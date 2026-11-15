import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSA_PAGE_SIZE,
    DSA_SPARSE_ATOL,
    DSA_SPARSE_INDEX_TOPK,
    DSA_SPARSE_RTOL,
    DSAAttentionCase,
    build_dsa_sparse_attention_fixture,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    EagleDraftRunnerSettings,
    _configure_runner_for_eagle_draft,
    _make_dsa_draft_inputs,
    _make_dsa_eagle_draft_forward_batch,
    _prepare_dsa_draft_replay_state,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSADraftChainLength(CustomTestCase):
    """A chain draft's decode step i attends the prefix and the i + 1 drafted keys.

    Request 0 crosses the index top-k budget while drafting, request 1 is past it.
    """

    CASE = DSAAttentionCase(
        name="dsa_draft_chain_length",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(DSA_SPARSE_INDEX_TOPK - 2, 3 * DSA_SPARSE_INDEX_TOPK),
    )
    STEPS = 4

    def setUp(self):
        bs = self.CASE.batch_size
        self.settings = EagleDraftRunnerSettings(
            topk=1,
            speculative_num_steps=self.STEPS,
            speculative_num_draft_tokens=self.STEPS + 1,
            capture_batch_size=bs,
            hidden_size=64,
            vocab_size=64,
            max_context_len=4 * DSA_SPARSE_INDEX_TOPK,
            dtype=torch.bfloat16,
            device="cuda",
            atol=DSA_SPARSE_ATOL,
            rtol=DSA_SPARSE_RTOL,
        )
        self.fixture = build_dsa_sparse_attention_fixture(
            self,
            self.CASE,
            max_context_len=self.settings.max_context_len,
            disable_cuda_graph=False,
            runner_batch_size=bs,
        )
        runner = self.fixture.runner
        _configure_runner_for_eagle_draft(runner, self.CASE, self.settings)
        self.draft_backend = DraftBackendFactory(
            runner, self.settings.topk, self.settings.speculative_num_steps
        ).create_decode_backend()
        _prepare_dsa_draft_replay_state(self.fixture, self.CASE, None, self.settings)
        self.batch = _make_dsa_eagle_draft_forward_batch(
            self.CASE, _make_dsa_draft_inputs(self.CASE, self.settings), self.settings
        )

    def assert_attends(self, metadata, extra):
        bs = self.CASE.batch_size
        lens = torch.tensor(self.CASE.prefix_lens, dtype=torch.int32) + extra
        self.assertEqual(metadata.cache_seqlens_int32[:bs].tolist(), lens.tolist())
        self.assertEqual(metadata.dsa_seqlens_expanded[:bs].tolist(), lens.tolist())
        self.assertEqual(
            metadata.cu_seqlens_k[1 : bs + 1].tolist(), lens.cumsum(0).tolist()
        )
        self.assertEqual(
            metadata.dsa_cache_seqlens_int32[:bs].tolist(),
            lens.clamp(max=DSA_SPARSE_INDEX_TOPK).tolist(),
        )
        req_to_token = self.fixture.runner.req_to_token_pool.req_to_token
        page = metadata.page_size
        for b, n in enumerate(lens.tolist()):
            torch.testing.assert_close(
                metadata.page_table_1[b, :n], req_to_token[b, :n].to(torch.int32)
            )
            # the page-64 table the indexer's paged scorer reads
            torch.testing.assert_close(
                metadata.real_page_table[b, : (n + page - 1) // page],
                (req_to_token[b, 0:n:page] // page).to(torch.int32),
            )

    def test_eager_steps_attend_their_chain(self):
        self.draft_backend.init_forward_metadata(self.batch)
        for i, backend in enumerate(self.draft_backend.attn_backends):
            with self.subTest(step=i):
                self.assert_attends(backend.forward_metadata, i + 1)

    def test_cuda_graph_steps_attend_their_chain(self):
        bs = self.CASE.batch_size
        self.draft_backend.init_cuda_graph_state(bs, bs)
        self.draft_backend.init_forward_metadata_out_graph(self.batch, in_capture=True)
        self.draft_backend.init_forward_metadata_out_graph(self.batch)
        for i, backend in enumerate(self.draft_backend.attn_backends):
            with self.subTest(step=i):
                self.assert_attends(backend.decode_cuda_graph_metadata[bs], i + 1)

    def test_target_decode_attends_the_prefix(self):
        self.fixture.backend.init_forward_metadata(self.batch)
        self.assert_attends(self.fixture.backend.forward_metadata, 0)


if __name__ == "__main__":
    unittest.main()
