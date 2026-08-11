"""Unit tests for ModelConfig.get_draft_hf_config — draft/target read parity.

The draft checkpoint's hf config must be read exactly as the target's is. A
draft read that drops one of the inputs fails, or resolves different values,
only on the draft side: the reported bug was the Gemma4 alias probe dropping
model_config_parser, so a checkpoint whose target config comes from a
registered plugin parser died in the hf parser's Hub lookup instead.

Parity is structural -- both reads go through ModelConfig._read_hf_config --
so what is tested here is the other half: that each server_args input reaches
that helper the way from_server_args would select it for a draft ModelConfig,
and that a failure is attributed to the right cause.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from transformers import PretrainedConfig

from sglang.srt.arg_groups.speculative_hook import (
    _draft_is_gemma4_assistant,
    _handle_dflash,
    _resolve_speculative_algorithm_alias,
)
from sglang.srt.configs.model_config import DraftConfigReadError, ModelConfig
from sglang.srt.configs.model_config_parser_registry import (
    _MODEL_CONFIG_PARSER_REGISTRY,
    ModelConfigParserBase,
    register_model_config_parser,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _draft_server_args(**overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; the fields
    # below are set directly (same pattern as unit/spec/test_spec_*.py). Using a
    # real ServerArgs makes field renames break these tests instead of leaving
    # them green while get_draft_hf_config raises AttributeError at startup.
    server_args = ServerArgs(model_path="dummy")
    fields = dict(
        # model_path="dummy" returns from __post_init__ before
        # _handle_missing_default_values auto-detects a device, so device stays
        # None and any handler that inspects it raises AttributeError.
        device="cuda",
        speculative_draft_model_path="/models/draft",
        speculative_draft_model_revision="main",
        revision=None,
        trust_remote_code=True,
        json_model_override_args='{"num_hidden_layers": 2}',
        model_config_parser="my_plugin_parser",
        decrypted_draft_config_file=None,
    )
    fields.update(overrides)
    for key, value in fields.items():
        setattr(server_args, key, value)
    return server_args


class TestDraftHfConfigResolution(CustomTestCase):
    def setUp(self):
        # _resolve_hf_config_path memoizes per process, and the patches below
        # only take effect on a miss, so a leftover entry for one of these
        # paths would silently make the resolution assertions vacuous.
        ModelConfig._resolve_hf_config_path.cache_clear()

    def _captured_call(self, **overrides):
        """(args, kwargs) get_config saw, for one get_draft_hf_config call."""
        with patch("sglang.srt.configs.model_config.get_config") as mock_get_config:
            mock_get_config.return_value = PretrainedConfig(architectures=["Probe"])
            result = ModelConfig.get_draft_hf_config(_draft_server_args(**overrides))
        mock_get_config.assert_called_once()
        self.assertEqual(result.architectures, ["Probe"])
        return mock_get_config.call_args

    def test_forwards_the_target_read_inputs(self):
        args, kwargs = self._captured_call()
        self.assertEqual(args, ("/models/draft",))
        self.assertEqual(kwargs["trust_remote_code"], True)
        self.assertEqual(kwargs["revision"], "main")
        # Passed as a parsed dict, as ModelConfig.__init__ does.
        self.assertEqual(kwargs["model_override_args"], {"num_hidden_layers": 2})
        # The dropped input behind the reported bug: without it, get_config
        # re-resolves model_config_parser="auto", which never routes to a
        # plugin parser.
        self.assertEqual(kwargs["model_config_parser"], "my_plugin_parser")

    def test_returns_an_isolated_copy_of_the_cached_config(self):
        # get_config memoizes process-wide, and consumers reach into nested
        # dicts (dspark_config/dflash_config are returned by reference), so a
        # caller normalizing one field must not corrupt every later read.
        cached = PretrainedConfig(architectures=["Probe"])
        with patch(
            "sglang.srt.configs.model_config.get_config", return_value=cached
        ) as mock_get_config:
            first = ModelConfig.get_draft_hf_config(_draft_server_args())
            second = ModelConfig.get_draft_hf_config(_draft_server_args())
        self.assertEqual(mock_get_config.call_count, 2)
        for result in (first, second):
            self.assertIsNot(result, cached)
        self.assertIsNot(first, second)

    def test_revision_falls_back_to_the_target_revision(self):
        # Mirrors from_server_args' `model_revision or server_args.revision`.
        # handle_speculative_decoding pins the draft revision to "main" only
        # when --speculative-draft-model-path was given; the bundled DSpark and
        # DeepSeek/GLM MTP paths default that path later, so they reach every
        # subsequent read with the fallback live.
        _, kwargs = self._captured_call(
            speculative_draft_model_revision=None, revision="abc123"
        )
        self.assertEqual(kwargs["revision"], "abc123")

    def test_decrypted_draft_config_becomes_the_configuration_file(self):
        _, kwargs = self._captured_call(
            decrypted_draft_config_file="  /tmp/decrypted/config.json  "
        )
        self.assertEqual(kwargs["_configuration_file"], "/tmp/decrypted/config.json")

    def test_no_configuration_file_when_undecrypted(self):
        for value in (None, "", "   "):
            with self.subTest(decrypted_draft_config_file=value):
                _, kwargs = self._captured_call(decrypted_draft_config_file=value)
                self.assertNotIn("_configuration_file", kwargs)

    def test_object_store_draft_path_is_resolved_before_the_read(self):
        # ModelConfig.__init__ normalizes the path before reading; a draft read
        # that skips it hands an s3:// URI to the hf parsers and lands on a
        # different cache key than the draft ModelConfig built later.
        with patch(
            "sglang.srt.configs.model_config.is_runai_obj_uri", return_value=True
        ), patch(
            "sglang.srt.configs.model_config.ObjectStorageModel.get_path",
            return_value="/local/pulled/draft",
        ):
            args, _ = self._captured_call(
                speculative_draft_model_path="s3://bucket/draft"
            )
        self.assertEqual(args, ("/local/pulled/draft",))

    def test_remote_draft_path_is_pulled_once_per_process(self):
        # The resolution sits outside get_config's cache, so an unmemoized
        # helper re-pulls the config over the network on every read. Each pull
        # also strands a tempdir and another link on the signal-handler chain
        # that pins it, and argument resolution reads the draft several times
        # before any ModelConfig exists.
        client = MagicMock()
        client.get_local_dir.return_value = "/local/remote-pull"
        with patch(
            "sglang.srt.configs.model_config.is_runai_obj_uri", return_value=False
        ), patch("sglang.srt.utils.is_remote_url", return_value=True), patch(
            "sglang.srt.connector.create_remote_connector", return_value=client
        ) as create_connector:
            for _ in range(3):
                args, _ = self._captured_call(
                    speculative_draft_model_path="remote://bucket/draft"
                )
        # Every read still gets the pulled local dir...
        self.assertEqual(args, ("/local/remote-pull",))
        # ...but only the first one paid for it.
        create_connector.assert_called_once()
        client.pull_files.assert_called_once()


class TestDraftReadFailureAttribution(CustomTestCase):
    def test_read_failure_names_the_draft_path_and_keeps_the_cause(self):
        cause = OSError(
            "Repo id must be in the form 'repo_name' or 'namespace/repo_name'"
        )
        with patch(
            "sglang.srt.configs.model_config.get_config", side_effect=cause
        ), self.assertRaises(DraftConfigReadError) as ctx:
            ModelConfig.get_draft_hf_config(_draft_server_args())
        message = str(ctx.exception)
        self.assertIn("speculative-draft-model-path", message)
        self.assertIn("/models/draft", message)
        # Callers that swallow this log it with %s, which drops __cause__, so
        # the cause has to survive in the message text itself.
        self.assertIn("Repo id must be in the form", message)
        self.assertIs(ctx.exception.__cause__, cause)

    def test_read_failure_does_not_name_the_decrypted_config_path(self):
        # The message reaches warning-level logs, and the path locates
        # decrypted plaintext on disk.
        path = "/tmp/decrypted/secret-config.json"
        with patch(
            "sglang.srt.configs.model_config.get_config", side_effect=OSError("boom")
        ), self.assertRaises(DraftConfigReadError) as ctx:
            ModelConfig.get_draft_hf_config(
                _draft_server_args(decrypted_draft_config_file=path)
            )
        message = str(ctx.exception)
        self.assertNotIn(path, message)
        self.assertIn("decrypted_draft_config_file=<set>", message)

    def test_malformed_model_override_args_is_not_a_draft_read_failure(self):
        with patch("sglang.srt.configs.model_config.get_config") as mock_get_config:
            with self.assertRaises(json.JSONDecodeError):
                ModelConfig.get_draft_hf_config(
                    _draft_server_args(json_model_override_args="{not json")
                )
        mock_get_config.assert_not_called()

    def test_a_broken_config_parser_is_not_reported_as_an_unreadable_draft(self):
        # A bug inside a registered parser must not be laundered into
        # "checkpoint unreadable", which callers downgrade to a warning.
        with patch(
            "sglang.srt.configs.model_config.get_config",
            side_effect=TypeError("parse() got an unexpected keyword argument"),
        ), self.assertRaises(TypeError):
            ModelConfig.get_draft_hf_config(_draft_server_args())


class TestDflashBlockSizeFallback(CustomTestCase):
    """Which failures _handle_dflash may downgrade to a default draft window."""

    def _dflash_server_args(self, **overrides):
        return _draft_server_args(
            speculative_algorithm="DFLASH",
            speculative_num_draft_tokens=None,
            speculative_dflash_block_size=None,
            # Pinned so _resolve_dflash_draft_attention_backend takes its
            # no-op path: its trtllm_mha branch reads the draft config again,
            # which is not what these tests are about.
            speculative_draft_attention_backend="triton",
            **overrides,
        )

    def test_unreadable_draft_falls_back_to_the_default_block_size(self):
        server_args = self._dflash_server_args()
        with patch(
            "sglang.srt.configs.model_config.get_config",
            side_effect=OSError("no such checkpoint"),
        ):
            _handle_dflash(server_args)
        self.assertEqual(server_args.speculative_num_draft_tokens, 16)

    def test_malformed_model_override_args_is_fatal(self):
        # Regression: the parse used to sit above this try block. Moving it
        # into the read put it back under `except`, so a quoting error in
        # --json-model-override-args was reported as an unreadable draft
        # checkpoint and the launch continued with a 16-token window before
        # dying later inside ModelConfig.__init__'s own json.loads.
        server_args = self._dflash_server_args(json_model_override_args="{not json")
        with patch("sglang.srt.configs.model_config.get_config") as mock_get_config:
            with self.assertRaises(json.JSONDecodeError):
                _handle_dflash(server_args)
        mock_get_config.assert_not_called()


class TestGemma4DraftAliasResolution(CustomTestCase):
    """The reported bug as behavior: a draft only the plugin parser can read.

    The draft paths below do not exist on disk, so any read that reaches the hf
    parser raises. Each test uses a distinct path because get_config is
    memoized process-wide.
    """

    def setUp(self):
        self._saved_registry = dict(_MODEL_CONFIG_PARSER_REGISTRY)

    def tearDown(self):
        _MODEL_CONFIG_PARSER_REGISTRY.clear()
        _MODEL_CONFIG_PARSER_REGISTRY.update(self._saved_registry)

    def _server_args_with_parsed_draft(self, *, draft_path, architecture, algorithm):
        parser_name = f"fake_parser_{Path(draft_path).name}"

        @register_model_config_parser(parser_name)
        class _FakeParser(ModelConfigParserBase):
            def parse(self, model, trust_remote_code, revision=None, **kwargs):
                return PretrainedConfig(architectures=[architecture])

        return _draft_server_args(
            speculative_draft_model_path=draft_path,
            model_config_parser=parser_name,
            speculative_algorithm=algorithm,
            json_model_override_args="{}",
        )

    def test_gemma4_assistant_draft_is_detected_through_the_plugin_parser(self):
        server_args = self._server_args_with_parsed_draft(
            draft_path="/nonexistent/gemma4-assistant-draft",
            architecture="Gemma4AssistantForCausalLM",
            algorithm="NEXTN",
        )
        self.assertTrue(_draft_is_gemma4_assistant(server_args))

    def test_non_gemma4_draft_is_not_detected(self):
        server_args = self._server_args_with_parsed_draft(
            draft_path="/nonexistent/llama-draft",
            architecture="LlamaForCausalLM",
            algorithm="NEXTN",
        )
        self.assertFalse(_draft_is_gemma4_assistant(server_args))

    def test_no_draft_read_for_algorithms_that_ignore_the_answer(self):
        # The probe used to read the draft config for every algorithm. Now that
        # the read honors --model-config-parser, a target-only parser would
        # turn an unused answer into a fatal startup error.
        server_args = _draft_server_args(speculative_algorithm="DSPARK")
        with patch("sglang.srt.configs.model_config.get_config") as mock_get_config:
            self.assertFalse(_draft_is_gemma4_assistant(server_args))
        mock_get_config.assert_not_called()

    def test_no_draft_read_without_a_draft_path(self):
        server_args = _draft_server_args(
            speculative_algorithm="EAGLE", speculative_draft_model_path=None
        )
        with patch("sglang.srt.configs.model_config.get_config") as mock_get_config:
            self.assertFalse(_draft_is_gemma4_assistant(server_args))
        mock_get_config.assert_not_called()

    def test_gemma4_draft_promotes_nextn_to_frozen_kv_mtp(self):
        for algorithm in ("NEXTN", "EAGLE"):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(
                    _resolve_speculative_algorithm_alias(
                        speculative_algorithm=algorithm, is_gemma4_draft=True
                    ),
                    "FROZEN_KV_MTP",
                )

    def test_gemma4_draft_rejects_eagle3(self):
        with self.assertRaisesRegex(ValueError, "EAGLE3 is"):
            _resolve_speculative_algorithm_alias(
                speculative_algorithm="EAGLE3", is_gemma4_draft=True
            )

    def test_non_gemma4_draft_keeps_the_nextn_to_eagle_alias(self):
        self.assertEqual(
            _resolve_speculative_algorithm_alias(
                speculative_algorithm="NEXTN", is_gemma4_draft=False
            ),
            "EAGLE",
        )

    def test_other_algorithms_pass_through(self):
        self.assertEqual(
            _resolve_speculative_algorithm_alias(
                speculative_algorithm="DSPARK", is_gemma4_draft=False
            ),
            "DSPARK",
        )


if __name__ == "__main__":
    unittest.main()
