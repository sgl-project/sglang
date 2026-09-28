import tempfile
import unittest

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.configs.iquest_q1 import IQuestQ1Config, IQuestQ1MTPConfig
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestIQuestQ1ArgumentResolution(CustomTestCase):
    def setUp(self):
        reset_context()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_context)
        IQuestQ1Config(dtype="bfloat16").save_pretrained(self.directory.name)

    def test_default_attention_backend_is_fa3(self):
        args = ServerArgs(
            model_path=self.directory.name,
            device="cuda",
            speculative_draft_attention_backend="triton",
        )
        args.resolve_once()
        self.assertEqual(resolved_view(args).attention_backend, "fa3")

    def test_architecture_selects_auto_parsers_without_overriding_explicit_choice(self):
        from sglang.srt.parser.template_detection import _architecture_auto_parsers

        args = ServerArgs(model_path=self.directory.name, device="cuda")
        self.assertEqual(
            _architecture_auto_parsers(args, ("reasoning_parser", "tool_call_parser")),
            {"reasoning_parser": "iquest_q1", "tool_call_parser": "iquest_q1"},
        )

    def test_mtp_validation_skips_other_models_drafts(self):
        from transformers import LlamaConfig

        from sglang.srt.arg_groups.speculative_hook import _handle_iquest_q1_mtp_draft

        with (
            tempfile.TemporaryDirectory() as target_directory,
            tempfile.TemporaryDirectory() as draft_directory,
        ):
            LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(
                target_directory
            )
            args = ServerArgs(
                model_path=target_directory,
                speculative_algorithm="EAGLE",
                speculative_draft_model_path=draft_directory,
                device="cuda",
            )
            self.assertFalse(_handle_iquest_q1_mtp_draft(args))

    def test_speculation_requires_an_independent_checkpoint(self):
        for draft_path in (None, self.directory.name):
            with self.subTest(draft_path=draft_path):
                reset_context()
                with self.assertRaisesRegex(ValueError, "independent draft"):
                    ServerArgs(
                        model_path=self.directory.name,
                        device="cuda",
                        speculative_algorithm="EAGLE",
                        speculative_draft_model_path=draft_path,
                    ).resolve_once()

    def test_target_checkpoint_cannot_be_used_as_separate_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            IQuestQ1Config().save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "MTP draft checkpoint"):
                ServerArgs(
                    model_path=self.directory.name,
                    speculative_draft_model_path=directory,
                    speculative_algorithm="EAGLE",
                    device="cuda",
                ).resolve_once()

    def test_mtp_draft_defaults_and_requested_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            IQuestQ1MTPConfig(
                target_config=IQuestQ1Config().to_dict(),
                sliding_window=512,
                swa_rope_theta=10000.0,
            ).save_pretrained(directory)
            for algorithm, steps, backend in (
                ("EAGLE", None, None),
                ("EAGLE", 2, "fa3"),
            ):
                with self.subTest(algorithm=algorithm, steps=steps, backend=backend):
                    reset_context()
                    args = ServerArgs(
                        model_path=self.directory.name,
                        speculative_draft_model_path=directory,
                        speculative_algorithm=algorithm,
                        speculative_num_steps=steps,
                        speculative_draft_attention_backend=backend,
                        device="cuda",
                    )
                    args.resolve_once()
                    cfg = resolved_view(args)
                    depth = 7 if steps is None else steps
                    self.assertEqual(cfg.speculative_num_steps, depth)
                    self.assertEqual(cfg.speculative_num_draft_tokens, depth + 1)
                    self.assertEqual(cfg.speculative_eagle_topk, 1)
                    self.assertEqual(cfg.speculative_draft_model_path, directory)
                    self.assertEqual(cfg.attention_backend, "fa3")

    def test_mtp_draft_rejects_aux_hidden_and_incompatible_dimensions(self):
        for config_kwargs, args_kwargs, message in (
            ({}, {"speculative_algorithm": "EAGLE3"}, "serial EAGLE"),
            (
                {},
                {"speculative_draft_attention_backend": "trtllm_mha"},
                "requires FA3",
            ),
            ({"hidden_size": 64}, {}, "hidden size and vocabulary"),
            ({"vocab_size": 32}, {}, "hidden size and vocabulary"),
            ({"num_target_layers": 87}, {}, "target layer count"),
            ({}, {"speculative_num_steps": 0}, "positive"),
            ({}, {"speculative_token_map": "unused"}, "full-vocabulary"),
            ({}, {"speculative_num_draft_tokens": 9}, "verification width"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(config=config_kwargs, args=args_kwargs):
                    reset_context()
                    IQuestQ1MTPConfig(**config_kwargs).save_pretrained(directory)
                    kwargs = {"speculative_algorithm": "EAGLE", **args_kwargs}
                    with self.assertRaisesRegex(ValueError, message):
                        ServerArgs(
                            model_path=self.directory.name,
                            speculative_draft_model_path=directory,
                            device="cuda",
                            **kwargs,
                        ).resolve_once()

    def test_rejects_backends_that_do_not_return_the_lse(self):
        for kwargs in (
            {"attention_backend": "triton"},
            {"decode_attention_backend": "triton"},
        ):
            with self.subTest(**kwargs):
                reset_context()
                with self.assertRaisesRegex(ValueError, "requires FA3"):
                    ServerArgs(
                        model_path=self.directory.name, device="cuda", **kwargs
                    ).resolve_once()

    def test_rejects_fa3_paths_that_do_not_return_the_lse(self):
        for kwargs, message in (
            ({"attn_cp_size": 2, "tp_size": 2}, "context parallelism"),
            (
                {
                    "speculative_algorithm": "EAGLE",
                    "speculative_num_steps": 1,
                    "speculative_eagle_topk": 2,
                    "speculative_num_draft_tokens": 2,
                },
                "speculative-eagle-topk 1",
            ),
        ):
            with self.subTest(**kwargs):
                reset_context()
                with self.assertRaisesRegex(ValueError, message):
                    ServerArgs(
                        model_path=self.directory.name, device="cuda", **kwargs
                    ).resolve_once()

    def test_rejects_multi_layer_eagle(self):
        with self.assertRaisesRegex(ValueError, "multi-layer EAGLE"):
            args = ServerArgs(
                model_path=self.directory.name,
                device="cuda",
                speculative_algorithm="EAGLE",
                enable_multi_layer_eagle=True,
                speculative_num_steps=1,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=2,
            )
            args.resolve_once()


if __name__ == "__main__":
    unittest.main()
