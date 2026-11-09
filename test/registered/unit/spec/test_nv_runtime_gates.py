import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNativeEagerVerify(CustomTestCase):
    def test_master_selects_native_sampling_or_runtime_hooks(self):
        root = Path(__file__).resolve().parents[4] / "python/sglang/srt/speculative"
        for module in ("eagle_worker_common", "ngram_worker"):
            tree = ast.parse((root / (module + ".py")).read_text())
            gate = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.If)
                and ast.unparse(n.test)
                == "envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()"
                and any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id == "eagle_sample"
                    for branch in n.orelse
                    for c in ast.walk(branch)
                )
            )
            for enabled in (False, True):
                with self.subTest(module=module, enabled=enabled):
                    batch = SimpleNamespace(
                        sampling_info=object(),
                        seq_lens=object(),
                        req_pool_indices=object(),
                        device="cpu",
                        return_logprob=False,
                        forward_mode=SimpleNamespace(is_idle=lambda: False),
                    )
                    runner = SimpleNamespace(
                        watermark_state=object(),
                        attn_backend=SimpleNamespace(verify_mask=None),
                    )
                    target = SimpleNamespace(model_runner=runner)
                    outputs = (object(), object(), object())
                    hooks = SimpleNamespace(
                        post_verify=Mock(
                            return_value=SimpleNamespace(
                                token_ids=outputs[0],
                                accepted_lengths=outputs[1],
                                accepted_indices=outputs[2],
                            )
                        )
                    )
                    sample, make_io = (
                        Mock(return_value=outputs),
                        Mock(return_value=object()),
                    )
                    scope = dict(
                        envs=envs,
                        batch=batch,
                        verify_input=object(),
                        logits_output=object(),
                        grammar_mask=object(),
                        target_worker=target,
                        self=SimpleNamespace(
                            model_runner=runner, _speculative_runtime_hooks=hooks
                        ),
                        speculative_runtime_hooks=hooks,
                        make_post_verify_io=make_io,
                        eagle_sample=sample,
                        post_verify_graphs=None,
                        uno_target_max_top_k=17,
                        simulated_acc_len_rng=object(),
                        sparse_bonus_argmax=True,
                    )
                    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(
                        enabled
                    ):
                        exec(
                            compile(
                                ast.Module(body=[gate], type_ignores=[]),
                                "<verify-gate>",
                                "exec",
                            ),
                            scope,
                        )
                    self.assertEqual(
                        (scope["predict"], scope["accept_lens"], scope["accept_index"]),
                        outputs,
                    )
                    if enabled:
                        sample.assert_not_called()
                        make_io.assert_called_once()
                        hooks.post_verify.assert_called_once_with(make_io.return_value)
                    else:
                        make_io.assert_not_called()
                        hooks.post_verify.assert_not_called()
                        kwargs = dict(watermark_state=runner.watermark_state)
                        if module == "eagle_worker_common":
                            kwargs.update(
                                uno_target_max_top_k=17,
                                watermark_full_mask=True,
                                simulated_acc_len_rng=scope["simulated_acc_len_rng"],
                                sparse_bonus_argmax=True,
                            )
                        sample.assert_called_once_with(
                            scope["verify_input"],
                            batch,
                            scope["logits_output"],
                            scope["grammar_mask"],
                            **kwargs,
                        )


if __name__ == "__main__":
    unittest.main()
