import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


def attributes(**values):
    return [
        {"key": key, "value": {"stringValue": str(value)}}
        for key, value in values.items()
    ]


def span(span_id, name, parent=None, **attrs):
    result = {
        "traceId": "0" * 31 + "1",
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": "10000000",
        "endTimeUnixNano": "13000000",
        "attributes": attributes(**attrs),
    }
    if parent is not None:
        result["parentSpanId"] = parent
    return result


def resource(service, spans):
    return {
        "resource": {"attributes": attributes(**{"service.name": service})},
        "scopeSpans": [{"spans": spans}],
    }


def engine_resource(parent=None):
    return resource(
        "sglang",
        [
            span(
                "0000000000000101",
                "request",
                parent=parent,
                module="sglang::request",
                rid="rid",
            ),
            span(
                "0000000000000102",
                "thread",
                parent="0000000000000101",
                pid="7",
                host_id="host",
                thread_label="scheduler",
            ),
            span("0000000000000103", "operation", parent="0000000000000102"),
        ],
    )


class TestConvertOtelToPerfetto(unittest.TestCase):
    def convert(self, input_data):
        script = Path(__file__).resolve().parents[1] / "convert_otel_2_perfetto.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "otel.json"
            output = root / "perfetto.json"
            source.write_text(input_data, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "-i", str(source), "-o", str(output)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            return json.loads(output.read_text(encoding="utf-8"))

    def test_array_whitespace_and_jsonl_have_identical_output(self):
        data = {"resourceSpans": [engine_resource()]}
        array = json.dumps([data])
        expected = self.convert(array)
        for text in (
            " \t\r\n" + array,
            "\n" + json.dumps([data], indent=2),
            json.dumps(data) + "\n",
            "\n \n" + json.dumps(data) + "\n\n",
        ):
            with self.subTest(input_data=text):
                self.assertEqual(self.convert(text), expected)
        self.assertEqual(
            [event["name"] for event in expected if event["ph"] == "X"],
            ["operation"],
        )

    def test_smg_spans_are_emitted_once_without_internal_child_field(self):
        parent = span("0000000000000001", "smg_parent")
        child = span("0000000000000002", "smg_child", parent="0000000000000001")
        for spans in ([parent], [parent, child]):
            with self.subTest(span_count=len(spans)):
                text = json.dumps([{"resourceSpans": [resource("smg", spans)]}])
                events = self.convert(text)
                complete = [event for event in events if event["ph"] == "X"]
                self.assertEqual(
                    Counter(event["name"] for event in complete),
                    Counter(item["name"] for item in spans),
                )
                self.assertTrue(all(event["pid"] == "smg" for event in complete))
                self.assertEqual(len({event["tid"] for event in complete}), len(spans))
                self.assertTrue(all(event["dur"] == 2999.0 for event in complete))

    def test_whitespace_prefixed_mixed_trace_preserves_smg_to_engine_flow(self):
        resources = [
            resource(
                "smg",
                [
                    span("0000000000000001", "smg_parent"),
                    span("0000000000000002", "smg_child", parent="0000000000000001"),
                ],
            ),
            engine_resource(parent="0000000000000001"),
        ]
        events = self.convert(
            " \n" + json.dumps([{"resourceSpans": resources}], indent=2)
        )
        complete = [event for event in events if event["ph"] == "X"]
        self.assertEqual(
            Counter(event["name"] for event in complete),
            Counter(["smg_parent", "smg_child", "operation"]),
        )
        flows = [event for event in events if event["ph"] in ("s", "f")]
        self.assertEqual(len(flows), 2)
        start = next(event for event in flows if event["ph"] == "s")
        finish = next(event for event in flows if event["ph"] == "f")
        self.assertEqual(start["id"], finish["id"])
        for flow, name in ((start, "smg_parent"), (finish, "operation")):
            target = next(event for event in complete if event["name"] == name)
            self.assertEqual(
                (flow["pid"], flow["tid"], flow["ts"]),
                (target["pid"], target["tid"], target["ts"]),
            )


if __name__ == "__main__":
    unittest.main()
