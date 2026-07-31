import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
HARNESS_PATH = REPO_ROOT / "scripts" / "frontend_api_parity.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


class FakeResponse:
    def __init__(
        self,
        *,
        status=200,
        content_type="application/json",
        body=None,
        text=None,
        lines=None,
    ):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._body = body
        self.text = text if text is not None else json.dumps(body)
        self._lines = lines or []

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body

    def iter_lines(self, decode_unicode=False):
        del decode_unicode
        return iter(self._lines)


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


class TestFrontendApiParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = _load_module("frontend_api_parity", HARNESS_PATH)

    def make_spec(self, *, normalize_paths=None, tolerance=0):
        return {
            "schema_version": 1,
            "cases": [
                {
                    "name": "completion",
                    "request": {
                        "method": "POST",
                        "path": "/v1/completions",
                        "json": {"prompt": "hello"},
                    },
                    "response_mode": "json",
                    "normalize_paths": normalize_paths or [],
                    "float_tolerance": tolerance,
                }
            ],
        }

    def test_case_spec_rejects_duplicate_names_and_bad_tolerance(self):
        spec = self.make_spec()
        spec["cases"].append(copy.deepcopy(spec["cases"][0]))
        with self.assertRaisesRegex(self.harness.HarnessError, "duplicate case"):
            self.harness.validate_case_spec(spec)

        spec = self.make_spec(tolerance=-1)
        with self.assertRaisesRegex(self.harness.HarnessError, "non-negative"):
            self.harness.validate_case_spec(spec)

    def test_sse_parser_preserves_order_and_wire_fields(self):
        lines = [
            ": heartbeat",
            "event: completion",
            "id: event-1",
            'data: {"text":"hel"}',
            "",
            "data: first",
            "data: second",
            "",
            "retry: 250",
            "data: [DONE]",
        ]

        self.assertEqual(
            self.harness.parse_sse_lines(lines),
            [
                {
                    "data": {"text": "hel"},
                    "event": "completion",
                    "id": "event-1",
                },
                {"data": "first\nsecond"},
                {"data": "[DONE]", "retry": 250},
            ],
        )

    def test_normalization_supports_wildcards_and_escaped_pointer_tokens(self):
        response = {
            "body": {
                "items": [{"id": "one"}, {"id": "two"}],
                "a/b": {"~key": "unstable"},
            }
        }

        normalized = self.harness.normalize_response(
            response, ["/body/items/*/id", "/body/a~1b/~0key"]
        )

        sentinel = self.harness.NORMALIZED_VALUE
        self.assertEqual(
            normalized,
            {
                "body": {
                    "items": [{"id": sentinel}, {"id": sentinel}],
                    "a/b": {"~key": sentinel},
                }
            },
        )
        self.assertEqual(response["body"]["items"][0]["id"], "one")

        with self.assertRaisesRegex(self.harness.HarnessError, "matched no"):
            self.harness.normalize_response(response, ["/body/missing"])
        self.assertEqual(
            self.harness.normalize_response(
                response, ["/body/missing"], require_match=False
            ),
            response,
        )

    def test_capture_normalizes_response_without_storing_cli_headers(self):
        response = FakeResponse(
            body={"id": "request-123", "created": 123, "choices": []}
        )
        session = FakeSession(response)
        spec = self.make_spec(normalize_paths=["/body/id", "/body/created"])

        snapshot = self.harness.capture_snapshot(
            self.harness.validate_case_spec(spec),
            base_url="http://localhost:30000/",
            label="python",
            revision="abc123",
            cli_headers={"Authorization": "Bearer secret"},
            timeout=5,
            session=session,
        )

        request = snapshot["cases"][0]["request"]
        self.assertNotIn("headers", request)
        self.assertEqual(
            snapshot["cases"][0]["response"]["body"],
            {
                "id": self.harness.NORMALIZED_VALUE,
                "created": self.harness.NORMALIZED_VALUE,
                "choices": [],
            },
        )
        method, url, kwargs = session.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "http://localhost:30000/v1/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")

    def test_candidate_capture_allows_missing_normalized_fields(self):
        response = FakeResponse(status=404, body={"detail": "Not Found"})
        session = FakeSession(response)
        spec = self.make_spec(normalize_paths=["/body/id"])

        snapshot = self.harness.capture_snapshot(
            self.harness.validate_case_spec(spec),
            base_url="http://localhost:30000",
            label="rust",
            revision=None,
            cli_headers={},
            timeout=5,
            require_normalization_match=False,
            session=session,
        )

        self.assertEqual(
            snapshot["cases"][0]["response"]["body"], {"detail": "Not Found"}
        )

    def test_compare_is_exact_by_default_and_honors_explicit_tolerance(self):
        reference = {
            "schema_version": 1,
            "case_spec_sha256": "same",
            "cases": [
                {
                    "name": "completion",
                    "request": {"path": "/v1/completions"},
                    "float_tolerance": 0,
                    "response": {"body": {"logprob": -0.125}},
                }
            ],
        }
        actual = copy.deepcopy(reference)
        actual["cases"][0]["response"]["body"]["logprob"] = -0.124

        exact_differences = self.harness.compare_snapshots(reference, actual)
        self.assertTrue(any("logprob" in item for item in exact_differences))

        reference["cases"][0]["float_tolerance"] = 0.01
        self.assertEqual(self.harness.compare_snapshots(reference, actual), [])

    def test_compare_detects_sse_order_and_case_spec_changes(self):
        reference = {
            "schema_version": 1,
            "case_spec_sha256": "python-spec",
            "cases": [
                {
                    "name": "stream",
                    "request": {"path": "/v1/completions"},
                    "float_tolerance": 0,
                    "response": {
                        "events": [{"data": {"text": "a"}}, {"data": "[DONE]"}]
                    },
                }
            ],
        }
        actual = copy.deepcopy(reference)
        actual["case_spec_sha256"] = "rust-spec"
        actual["cases"][0]["response"]["events"].reverse()

        differences = self.harness.compare_snapshots(reference, actual)

        self.assertTrue(any("case spec digest differs" in item for item in differences))
        self.assertTrue(any("events[0]" in item for item in differences))

    def test_selected_compare_can_use_a_full_reference_snapshot(self):
        reference = {
            "schema_version": 1,
            "case_spec_sha256": "same",
            "cases": [{"name": "one"}, {"name": "two"}],
        }

        filtered = self.harness.select_snapshot_cases(reference, ["two"])

        self.assertEqual([case["name"] for case in filtered["cases"]], ["two"])
        self.assertEqual(len(reference["cases"]), 2)
        with self.assertRaisesRegex(self.harness.HarnessError, "missing selected"):
            self.harness.select_snapshot_cases(reference, ["three"])

    def test_snapshot_round_trip_is_stable(self):
        snapshot = {
            "schema_version": 1,
            "case_spec_sha256": "same",
            "source": {"label": "python"},
            "cases": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "snapshot.json"
            self.harness.write_snapshot(path, snapshot)
            loaded = self.harness.load_json(path)

        self.assertEqual(loaded, snapshot)


if __name__ == "__main__":
    unittest.main()
