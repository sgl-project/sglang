"""Unit tests for Frozen-KV MTP delayed target-pool binding."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
    FrozenKVMTPDraftWorker,
    FrozenKVMTPWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")


class _DraftModel:
    backbone_hidden_size = 16

    def __init__(self):
        self.embed_and_head = None
        self.bound_context = None
        self.build_context_args = None

    def set_embed_and_head(self, embed, head):
        self.embed_and_head = (embed, head)

    def build_frozen_kv_mtp_context(self, *, target_model, target_token_to_kv_pool):
        self.build_context_args = (target_model, target_token_to_kv_pool)
        return object()

    def bind_frozen_kv_context(self, ctx):
        self.bound_context = ctx


def _server_args():
    return SimpleNamespace(
        speculative_eagle_topk=1,
        speculative_num_steps=1,
        speculative_num_draft_tokens=1,
        speculative_algorithm="FROZEN_KV_MTP",
        speculative_adaptive=False,
        enable_dp_attention=False,
        device="cpu",
        page_size=1,
        context_length=128,
    )


def _target_worker():
    target_model = MagicMock()
    target_model.get_embed_and_head.return_value = ("embed", "head")
    target_runner = SimpleNamespace(
        model=target_model,
        model_config=SimpleNamespace(context_len=128),
        memory_pool_config=None,
        token_to_kv_pool=object(),
    )
    worker = MagicMock()
    worker.model_runner = target_runner
    worker.get_memory_pool.side_effect = AssertionError(
        "target pool should not be read during construction"
    )
    return worker


def _patch_tp_model_worker():
    init_calls = []
    alloc_calls = []

    def fake_init(
        self,
        *,
        server_args,
        gpu_id,
        tp_rank,
        pp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        is_draft_worker=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        memory_pool_config=None,
        **_,
    ):
        init_calls.append(
            {
                "is_draft_worker": is_draft_worker,
                "req_to_token_pool": req_to_token_pool,
                "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
                "memory_pool_config": memory_pool_config,
            }
        )
        self._model_runner = SimpleNamespace(
            model=_DraftModel(),
            tp_group=object(),
            device="cpu",
            attn_backend=object(),
        )

    def fake_alloc(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        alloc_calls.append(
            {
                "memory_pool_config": memory_pool_config,
                "req_to_token_pool": req_to_token_pool,
                "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            }
        )

    return (
        patch(
            "sglang.srt.speculative.frozen_kv_mtp_worker_v2.TpModelWorker.__init__",
            fake_init,
        ),
        patch(
            "sglang.srt.speculative.frozen_kv_mtp_worker_v2.TpModelWorker.alloc_memory_pool",
            fake_alloc,
        ),
        init_calls,
        alloc_calls,
    )


def _make_draft_worker(target):
    return FrozenKVMTPDraftWorker(
        _server_args(),
        gpu_id=0,
        tp_rank=0,
        dp_rank=None,
        moe_ep_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        nccl_port=12345,
        target_worker=target,
    )


class TestFrozenKVMTPPoolInit(CustomTestCase):
    def test_draft_worker_constructor_does_not_read_target_pool(self):
        init_patch, alloc_patch, init_calls, _ = _patch_tp_model_worker()
        target = _target_worker()

        with init_patch, alloc_patch:
            worker = _make_draft_worker(target)

        target.get_memory_pool.assert_not_called()
        self.assertIsNone(worker.req_to_token_pool)
        self.assertIsNone(worker.token_to_kv_pool_allocator)
        self.assertIsNone(worker.draft_pool_config)
        self.assertIsNone(worker.kv_context)
        self.assertEqual(len(init_calls), 1)
        self.assertIsNone(init_calls[0]["req_to_token_pool"])
        self.assertIsNone(init_calls[0]["token_to_kv_pool_allocator"])
        self.assertIsNone(init_calls[0]["memory_pool_config"])
        self.assertEqual(
            worker.draft_model_runner.model.embed_and_head, ("embed", "head")
        )
        self.assertIsNone(worker.draft_model_runner.model.bound_context)

    def test_draft_worker_alloc_binds_target_pool_after_target_init(self):
        init_patch, alloc_patch, _, alloc_calls = _patch_tp_model_worker()
        target = _target_worker()
        req_pool = object()
        allocator = object()
        target_cfg = MemoryPoolConfig(max_total_num_tokens=1024, max_running_requests=7)

        with init_patch, alloc_patch:
            worker = _make_draft_worker(target)
            worker.alloc_memory_pool(
                memory_pool_config=target_cfg,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=allocator,
            )

        self.assertIs(worker.req_to_token_pool, req_pool)
        self.assertIs(worker.token_to_kv_pool_allocator, allocator)
        self.assertEqual(worker.draft_pool_config.max_total_num_tokens, 64)
        self.assertEqual(worker.draft_pool_config.max_running_requests, 7)
        self.assertEqual(len(alloc_calls), 1)
        self.assertIs(alloc_calls[0]["memory_pool_config"], worker.draft_pool_config)
        self.assertIs(alloc_calls[0]["req_to_token_pool"], req_pool)
        self.assertIs(alloc_calls[0]["token_to_kv_pool_allocator"], allocator)
        draft_model = worker.draft_model_runner.model
        self.assertIs(draft_model.bound_context, worker.kv_context)
        self.assertEqual(
            draft_model.build_context_args,
            (target.model_runner.model, target.model_runner.token_to_kv_pool),
        )

    def test_draft_worker_alloc_requires_target_pool_config(self):
        init_patch, alloc_patch, _, alloc_calls = _patch_tp_model_worker()
        target = _target_worker()

        with init_patch, alloc_patch:
            worker = _make_draft_worker(target)
            with self.assertRaisesRegex(ValueError, "target memory pool config"):
                worker.alloc_memory_pool(memory_pool_config=None)

        self.assertEqual(alloc_calls, [])

    def test_worker_v2_constructor_does_not_read_target_pool(self):
        target = _target_worker()

        with (
            patch(
                "sglang.srt.speculative.frozen_kv_mtp_worker_v2.FrozenKVMTPDraftWorker",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.speculative.frozen_kv_mtp_worker_v2._get_plan_stream",
                return_value=(None, None),
            ),
        ):
            FrozenKVMTPWorkerV2(
                _server_args(),
                gpu_id=0,
                tp_rank=0,
                dp_rank=None,
                moe_ep_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                nccl_port=12345,
                target_worker=target,
            )

        target.get_memory_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
