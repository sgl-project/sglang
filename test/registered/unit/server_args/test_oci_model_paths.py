"""`oci://` model references resolve to a local path before anything else runs.

The scheme is explicit on purpose: a bare `registry/name:tag` is the same shape
as a HuggingFace repo id, so sniffing would hijack existing `--model-path
org/model` deployments. These tests pin that, the output contract llmman
promises, and that every other reference shape is left untouched.
"""

import json
import os
import tempfile
import unittest
import unittest.mock

from sglang.srt.utils import llmman
from sglang.srt.utils.oci_utils import is_oci_uri, resolve_oci_model, strip_oci_scheme
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOciScheme(CustomTestCase):
    def test_recognizes_the_oci_scheme(self):
        self.assertTrue(is_oci_uri("oci://ghcr.io/org/model:tag"))
        self.assertTrue(is_oci_uri("OCI://ghcr.io/org/model:tag"))

    def test_leaves_every_other_reference_shape_alone(self):
        # A bare HF repo id must never be claimed.
        for value in (
            "Qwen/Qwen3-0.6B",
            "ghcr.io/org/model:tag",
            "/local/path/to/model",
            "s3://bucket/key",
            "gs://bucket/key",
            "az://container/key",
            "",
            None,
        ):
            self.assertFalse(is_oci_uri(value), value)

    def test_accepts_pathlib_input(self):
        import pathlib

        self.assertFalse(is_oci_uri(pathlib.Path("/local/model")))

    def test_strips_the_scheme_only_when_present(self):
        self.assertEqual(
            strip_oci_scheme("oci://ghcr.io/org/model:tag"), "ghcr.io/org/model:tag"
        )
        self.assertEqual(
            strip_oci_scheme("OCI://ghcr.io/org/model:tag"), "ghcr.io/org/model:tag"
        )
        self.assertEqual(strip_oci_scheme("Qwen/Qwen3-0.6B"), "Qwen/Qwen3-0.6B")


class TestResolveOutputContract(CustomTestCase):
    """`llmman resolve --no-pull` reports where the daemon's pull landed."""

    def test_parses_the_documented_contract(self):
        with tempfile.TemporaryDirectory() as path:
            line = json.dumps(
                {
                    "reference": "ghcr.io/org/model:tag",
                    "path": path,
                    "format": "safetensors",
                }
            )
            self.assertEqual(llmman.parse_resolve_output(line, "ref"), path)

    def test_tolerates_trailing_newline_and_leaked_diagnostics(self):
        with tempfile.TemporaryDirectory() as path:
            out = f'pulling blobs...\n{json.dumps({"path": path})}\n'
            self.assertEqual(llmman.parse_resolve_output(out, "ref"), path)

    def test_ignores_unknown_fields_so_the_contract_can_grow(self):
        with tempfile.TemporaryDirectory() as path:
            line = json.dumps(
                {"path": path, "format": "gguf", "mmproj": "/x", "future": 1}
            )
            self.assertEqual(llmman.parse_resolve_output(line, "ref"), path)

    def test_rejects_malformed_output(self):
        for out in (
            "",
            "   \n\n",
            "not json",
            '["a", "list"]',
            '{"no_path": 1}',
            '{"path": ""}',
            '{"path": 3}',
            '{"path": "/nonexistent/xyzzy"}',
        ):
            with self.assertRaises(RuntimeError):
                llmman.parse_resolve_output(out, "ref")


class TestEndpointResolution(CustomTestCase):
    def test_parses_every_llmman_host_form(self):
        cases = {
            "": "http://127.0.0.1:17434",
            "1.2.3.4:9999": "http://1.2.3.4:9999",
            "1.2.3.4": "http://1.2.3.4:17434",
            "http://1.2.3.4:9999/ignored": "http://1.2.3.4:9999",
            # A wildcard bind is meaningful to the server but not to a client.
            "0.0.0.0:9999": "http://127.0.0.1:9999",
            "[::]:9999": "http://[::1]:9999",
        }
        for value, want in cases.items():
            with unittest.mock.patch.dict(os.environ, {llmman.HOST_ENV: value}):
                self.assertEqual(llmman.endpoint(), want, value)

    def test_binary_default_and_override(self):
        with unittest.mock.patch.dict(os.environ, {llmman.BIN_ENV: ""}):
            self.assertEqual(llmman.llmman_bin(), "llmman")
        with unittest.mock.patch.dict(os.environ, {llmman.BIN_ENV: "/opt/llmman"}):
            self.assertEqual(llmman.llmman_bin(), "/opt/llmman")


class TestResolveOciModel(CustomTestCase):
    def test_rejects_an_empty_reference_without_touching_the_daemon(self):
        with self.assertRaises(ValueError):
            resolve_oci_model("oci://")
        with self.assertRaises(ValueError):
            resolve_oci_model("oci://   ")

    def test_strips_the_scheme_before_handing_off_to_llmman(self):
        with unittest.mock.patch(
            "sglang.srt.utils.oci_utils.llmman.pull_and_resolve",
            return_value="/resolved",
        ) as acquire:
            self.assertEqual(
                resolve_oci_model("oci://ghcr.io/org/model:tag"), "/resolved"
            )
        self.assertEqual(acquire.call_args[0][0], "ghcr.io/org/model:tag")
        self.assertIsNotNone(acquire.call_args[1]["progress"])


class TestModelPathHook(CustomTestCase):
    """The hook rewrites only oci:// fields, and pulls each image once."""

    def _run_hook(self, **fields):
        from sglang.srt.arg_groups.model_path_hook import resolve_oci_model_paths

        server_args = unittest.mock.Mock()
        cfg = unittest.mock.Mock(**fields)
        declared = []
        with unittest.mock.patch(
            "sglang.srt.arg_groups.model_path_hook.resolving_view", return_value=cfg
        ), unittest.mock.patch(
            "sglang.srt.arg_groups.model_path_hook.declare_resolution",
            side_effect=lambda _sa, _src, **kw: declared.append(kw),
        ), unittest.mock.patch(
            "sglang.srt.arg_groups.model_path_hook.resolve_oci_model",
            side_effect=lambda ref: f"/resolved/{ref.rsplit('/', 1)[-1]}",
        ) as resolver:
            resolve_oci_model_paths(server_args)
        return declared, resolver

    def test_no_oci_reference_declares_nothing(self):
        declared, resolver = self._run_hook(
            model_path="Qwen/Qwen3-0.6B",
            tokenizer_path="Qwen/Qwen3-0.6B",
            speculative_draft_model_path=None,
        )
        self.assertEqual(declared, [])
        resolver.assert_not_called()

    def test_rewrites_the_model_path(self):
        declared, _ = self._run_hook(
            model_path="oci://ghcr.io/org/model:tag",
            tokenizer_path=None,
            speculative_draft_model_path=None,
        )
        self.assertEqual(declared, [{"model_path": "/resolved/model:tag"}])

    def test_pulls_a_shared_reference_once(self):
        ref = "oci://ghcr.io/org/model:tag"
        declared, resolver = self._run_hook(
            model_path=ref,
            tokenizer_path=ref,
            speculative_draft_model_path=None,
        )
        self.assertEqual(resolver.call_count, 1)
        self.assertEqual(
            declared,
            [
                {"model_path": "/resolved/model:tag"},
                {"tokenizer_path": "/resolved/model:tag"},
            ],
        )

    def test_leaves_a_non_oci_tokenizer_alone(self):
        declared, _ = self._run_hook(
            model_path="oci://ghcr.io/org/model:tag",
            tokenizer_path="Qwen/Qwen3-0.6B",
            speculative_draft_model_path=None,
        )
        self.assertEqual(declared, [{"model_path": "/resolved/model:tag"}])


if __name__ == "__main__":
    unittest.main()
