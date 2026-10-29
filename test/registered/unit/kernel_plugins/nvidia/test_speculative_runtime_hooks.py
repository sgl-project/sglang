"""CPU tests for resolving speculative provider hooks at the binding boundary."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.kernels.plugin_kernels.nvidia import spec
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia import speculative_runtime as runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


def _defaults() -> spec.SpeculativeRuntimeHooks:
    return runtime.default_speculative_runtime_hooks()


class TestSpeculativeRuntimeHooks(CustomTestCase):
    def test_master_switch_reads_only_the_runtime_env(self):
        for value in (False, True):
            with (
                self.subTest(value=value),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(value),
                patch.object(
                    binding,
                    "megakernel_enabled",
                    side_effect=AssertionError("must not read the CLI switch"),
                ) as cli_enabled,
            ):
                self.assertIs(
                    envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get(), value
                )
                cli_enabled.assert_not_called()

    def test_switch_off_returns_stock_identity_without_provider_work(self):
        defaults = _defaults()
        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=lambda _: (_ for _ in ()).throw(
                AssertionError("factory must not run")
            )
        )
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
            patch.object(
                binding,
                "built_megakernel",
                side_effect=AssertionError("off must not inspect a provider"),
            ),
            patch.object(binding, "megakernel_agree") as agree,
        ):
            selected = binding.bind_speculative_runtime_hooks(model, defaults)
        self.assertIs(selected, defaults)
        agree.assert_not_called()

    def test_missing_provider_keeps_exact_defaults(self):
        defaults = _defaults()
        models = (torch.nn.Module(), torch.nn.Module())
        models[1].__dict__["_megakernel"] = SimpleNamespace(widths=())
        for model in models:
            with (
                self.subTest(has_built=binding.built_megakernel(model) is not None),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                patch.object(binding, "_gather_tp", return_value=[False]),
            ):
                selected = binding.bind_speculative_runtime_hooks(model, defaults)
            self.assertIs(selected, defaults)

    def test_factory_wraps_defaults_and_changes_only_local_keys(self):
        defaults = _defaults()
        custom_keys = lambda view: torch.zeros(
            view.values.shape[0], dtype=torch.int64, device=view.values.device
        )

        def factory(stock):
            return spec.SpeculativeRuntimeHooks(
                post_verify=spec.PostVerifyStrategy(
                    run=stock.post_verify.run,
                    local_keys=custom_keys,
                )
            )

        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=factory
        )
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", return_value=[True]),
        ):
            selected = binding.bind_speculative_runtime_hooks(model, defaults)
        self.assertIs(selected.post_verify.run, defaults.post_verify.run)
        self.assertIs(selected.post_verify.local_keys, custom_keys)

    def test_factory_cannot_replace_the_runtime_runner(self):
        defaults = _defaults()

        def factory(stock):
            return msgspec.structs.replace(
                stock,
                post_verify=msgspec.structs.replace(
                    stock.post_verify,
                    run=lambda io, local_keys: stock.post_verify(io),
                ),
            )

        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=factory
        )
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", return_value=[True]),
            self.assertRaisesRegex(ValueError, "cannot replace.*runner"),
        ):
            binding.bind_speculative_runtime_hooks(model, defaults)

    def test_factory_must_return_a_callable_local_key_policy(self):
        defaults = _defaults()

        def factory(stock):
            return msgspec.structs.replace(
                stock,
                post_verify=msgspec.structs.replace(stock.post_verify, local_keys=None),
            )

        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=factory
        )
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", return_value=[True]),
            self.assertRaisesRegex(TypeError, "local_keys must be callable"),
        ):
            binding.bind_speculative_runtime_hooks(model, defaults)

    def test_provider_presence_must_agree_across_ranks(self):
        defaults = _defaults()
        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=lambda stock: stock
        )
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", return_value=[True, False]),
            self.assertRaisesRegex(RuntimeError, "factory presence differ"),
        ):
            binding.bind_speculative_runtime_hooks(model, defaults)

    def test_defaults_are_safe_before_the_sharded_runtime_lands(self):
        defaults = _defaults()
        view = spec.ShardedLogitsView(
            values=torch.ones((2, 4)),
            shard=spec.VocabShardView(start=0, valid=4, vocab_size=4),
            temperatures=None,
            seeds=None,
            splits=1,
        )
        # Exercise the standalone binding tree even when this test is also run
        # against a composed runtime+binding checkout.
        with patch.dict(
            sys.modules,
            {"sglang.srt.speculative.sharded_verify": None},
        ):
            self.assertIsNone(defaults.post_verify.local_keys(view))

    def test_default_post_verify_uses_the_stock_signature(self):
        io = SimpleNamespace(
            device=torch.device("cpu"),
            is_idle=False,
            seq_lens=torch.tensor([3]),
            sampling_info=SimpleNamespace(),
            req_pool_indices=torch.tensor([0]),
            return_logprob=False,
            verify_tree=object(),
            logits=object(),
            grammar_mask=None,
            sparse_target_max_top_k=None,
            watermark=None,
            watermark_full_mask=True,
            simulated_accept_length=None,
            simulated_accept_length_rng=None,
            sparse_bonus_argmax=False,
        )
        expected = (
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        with (
            patch(
                "sglang.srt.speculative.eagle_utils.LOCAL_KEYS_SUPPORTED",
                False,
                create=True,
            ),
            patch(
                "sglang.srt.speculative.eagle_utils.eagle_sample",
                return_value=expected,
            ) as sample,
        ):
            output = runtime.default_post_verify(io, lambda view: None)
        self.assertEqual(
            sample.call_args.kwargs,
            {
                "uno_target_max_top_k": None,
                "watermark_state": None,
                "watermark_full_mask": True,
                "simulated_acc_len": None,
                "simulated_acc_len_rng": None,
                "sparse_bonus_argmax": False,
            },
        )
        self.assertTrue(torch.equal(output.token_ids, expected[0]))


if __name__ == "__main__":
    import unittest

    unittest.main()
