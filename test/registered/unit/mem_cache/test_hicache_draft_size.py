"""Independent draft host budgets preserve the target-derived index space."""

import argparse
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.arg_groups.hicache_hook import handle_hicache_ratio_default
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftSizeArguments(CustomTestCase):
    def test_cli_default_and_fractional_gigabytes(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        self.assertEqual(
            parser.parse_args(["--model-path", "dummy"]).hicache_draft_size, 0
        )
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-hierarchical-cache",
                "--speculative-algorithm",
                "NEXTN",
                "--hicache-draft-size",
                "1.6",
            ]
        )
        args = ServerArgs.from_cli_args(parsed)
        self.assertEqual(args.hicache_draft_size, 1.6)
        handle_hicache_ratio_default(args)
        self.assertIsNone(resolution_result(args, "hicache_host_memory_fraction", 0.8))

    def test_existing_positional_arguments_keep_their_meaning(self):
        names = [field.name for field in msgspec.structs.fields(ServerArgs)]
        position = names.index("hicache_size") + 1
        defaults = ServerArgs(model_path="dummy")
        values = [getattr(defaults, name) for name in names[: position + 1]]
        values[-1] = "write_back"
        args = ServerArgs(*values)
        self.assertEqual(args.hicache_write_policy, "write_back")
        self.assertEqual(args.hicache_draft_size, 0)

    def test_zero_preserves_automatic_sizing(self):
        args = ServerArgs(model_path="dummy")
        handle_hicache_ratio_default(args)
        self.assertEqual(
            resolution_result(args, "hicache_host_memory_fraction", 0.8), 0.8
        )
        self.assertEqual(args.hicache_draft_size, 0)

    def test_invalid_sizes_and_missing_draft_are_rejected(self):
        for size in (-1, float("nan"), float("inf"), -float("inf")):
            with (
                self.subTest(size=size),
                self.assertRaisesRegex(ValueError, "finite and nonnegative"),
            ):
                handle_hicache_ratio_default(
                    ServerArgs(model_path="dummy", hicache_draft_size=size)
                )
        with self.assertRaisesRegex(ValueError, "enable-hierarchical-cache"):
            handle_hicache_ratio_default(
                ServerArgs(model_path="dummy", hicache_draft_size=1.6)
            )
        for algorithm in (None, "NGRAM"):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(ValueError, "draft model"),
            ):
                handle_hicache_ratio_default(
                    ServerArgs(
                        model_path="dummy",
                        enable_hierarchical_cache=True,
                        speculative_algorithm=algorithm,
                        hicache_draft_size=1.6,
                    )
                )


class TestIndependentDraftHostPools(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_context().override_server_args(
                enable_hierarchical_cache=True,
                speculative_algorithm="EAGLE",
                hicache_mem_layout="layer_first",
            )
        )
        self.enterContext(
            patch.object(assembler, "_get_allocator_type", return_value="default")
        )
        # Keep the host-memory probe independent of the CI container's reserve.
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
                return_value=1024**2,
            )
        )
        # Real pools and tensors with ordinary CPU allocation instead of pinning.
        self.enterContext(
            patch.object(
                assembler,
                "get_mha_host_pool_cls",
                return_value=partial(
                    MHATokenToKVPoolHost,
                    pin_memory=False,
                ),
            )
        )
        self.enterContext(
            patch.object(
                assembler,
                "MLATokenToKVPoolHost",
                partial(
                    MLATokenToKVPoolHost,
                    pin_memory=False,
                ),
            )
        )

    def device_pool(self, kind, size):
        common = dict(
            size=size,
            page_size=2,
            dtype=torch.float16,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        if kind == "mha":
            return MHATokenToKVPool(head_num=2, head_dim=4, **common)
        return MLATokenToKVPool(kv_lora_rank=4, qk_rope_head_dim=2, **common)

    def build(self, pool, target_tokens, size_gb):
        controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(logical_size=target_tokens),
            page_size=2,
            transfer_layer_id_max=1,
        )
        with get_memory().override(hicache_draft_size=size_gb):
            specs, entries = assembler.build_full_draft_pools(
                draft_kv_pool=pool,
                tree_cache=SimpleNamespace(cache_controller=controller),
            )
        self.assertEqual(len(specs), 1)
        self.assertEqual(controller.mem_pool_host.logical_size, target_tokens)
        return entries[0].host_pool

    def test_fixed_budget_is_independent_of_device_and_target_capacity(self):
        # 1000 bytes: MHA uses 32 B/token, MLA 12 B/token, both page-aligned.
        for kind, expected in (("mha", 32), ("mla", 84)):
            for device_size in (8, 24):
                for target_tokens in (16, 20):
                    with self.subTest(
                        kind=kind, device=device_size, target=target_tokens
                    ):
                        host = self.build(
                            self.device_pool(kind, device_size), target_tokens, 1e-6
                        )
                        self.assertEqual(host.logical_size, expected)
                        self.assertIsNotNone(host.kv_buffer)
                        self.assertEqual(host.kv_buffer.device.type, "cpu")

    def test_default_still_follows_target_capacity(self):
        for kind in ("mha", "mla"):
            for target_tokens in (16, 20):
                with self.subTest(kind=kind, target=target_tokens):
                    host = self.build(self.device_pool(kind, 8), target_tokens, 0)
                    self.assertEqual(host.logical_size, target_tokens)

    def test_undersized_budget_fails_before_allocating_payload(self):
        for kind, host_cls in (
            ("mha", MHATokenToKVPoolHost),
            ("mla", MLATokenToKVPoolHost),
        ):
            with (
                self.subTest(kind=kind),
                patch.object(host_cls, "_allocate_host_buffer") as allocate,
            ):
                with self.assertRaisesRegex(ValueError, "must cover at least 16"):
                    self.build(self.device_pool(kind, 8), 16, 1e-9)
                allocate.assert_not_called()

    def worker(self, pools, *, nextn_layers=1, fused=False):
        drafts = tuple(
            SimpleNamespace(
                token_to_kv_pool=pool,
                runner_ids=SimpleNamespace(is_fused_draft=fused),
                model_config=SimpleNamespace(
                    num_nextn_predict_layers=nextn_layers,
                    hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                ),
            )
            for pool in pools
        )
        return SimpleNamespace(
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    token_to_kv_pool=self.device_pool("mha", 8),
                )
            ),
            _draft_model_runners=lambda: drafts,
        )

    def test_explicit_budget_uses_sidecar_without_packed_target_layers(self):
        for kind in ("mha", "mla"):
            pool = self.device_pool(kind, 8)
            worker = self.worker((pool,))
            original = BaseSpecWorker._build_hicache_draft_plan(worker)
            self.assertEqual(original.mode, HiCacheDraftMode.PACKED)
            with get_memory().override(hicache_draft_size=1e-6):
                explicit = BaseSpecWorker._build_hicache_draft_plan(worker)
            self.assertEqual(explicit.mode, HiCacheDraftMode.SIDECAR)
            self.assertEqual(explicit.device_pools, (pool,))
            self.assertEqual(
                worker.target_worker.model_runner.mtp_draft_device_pools, ()
            )

    def test_unsupported_draft_plans_fail_before_pool_construction(self):
        pool = self.device_pool("mha", 8)
        cases = [
            (self.worker(()), "exactly one"),
            (self.worker((pool, pool)), "exactly one"),
            (self.worker((pool,), fused=True), "separate draft pool"),
            (self.worker((SimpleNamespace(layer_num=1),)), "full-attention"),
        ]
        with get_memory().override(hicache_draft_size=1e-6):
            for worker, message in cases:
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    BaseSpecWorker._build_hicache_draft_plan(worker)


if __name__ == "__main__":
    unittest.main()
