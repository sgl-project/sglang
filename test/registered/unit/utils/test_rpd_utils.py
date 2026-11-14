"""CPU-only RPD SQLite to compressed Chrome trace regression tests."""

import gzip
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRpdUtils(CustomTestCase):
    def export(self, text, trace_format, *, description=None, optional_tables=False):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "trace.rpd"
            output = Path(tmp) / "trace.json.gz"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE rocpd_string (id INTEGER PRIMARY KEY, string TEXT);
                    CREATE TABLE rocpd_api (
                        id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER,
                        start INTEGER, end INTEGER, apiName_id INTEGER, args_id INTEGER
                    );
                    CREATE TABLE rocpd_op (
                        id INTEGER PRIMARY KEY, gpuId INTEGER, queueId INTEGER,
                        start INTEGER, end INTEGER, opType_id INTEGER, description_id INTEGER
                    );
                    CREATE TABLE rocpd_api_ops (id INTEGER, api_id INTEGER, op_id INTEGER);
                    """
                )
                connection.executemany(
                    "INSERT INTO rocpd_string VALUES (?, ?)",
                    [
                        (1, "UserMarker"),
                        (2, text),
                        (3, "hipLaunchKernel"),
                        (4, text if description is None else description),
                    ],
                )
                connection.executemany(
                    "INSERT INTO rocpd_api VALUES (?, 10, 20, ?, ?, ?, 2)",
                    [
                        (1, 1000, 1000, 1),  # Instant marker.
                        (2, 2000, 10000, 1),  # Range enclosing the first op.
                        (3, 3000, 4000, 3),
                        (4, 20000, 30000, 1),  # New range flushes the GPU frame.
                        (5, 21000, 22000, 3),
                    ],
                )
                connection.executemany(
                    "INSERT INTO rocpd_op VALUES (?, 0, 7, ?, ?, 2, 4)",
                    [(1, 5000, 6000), (2, 23000, 24000)],
                )
                connection.executemany(
                    "INSERT INTO rocpd_api_ops VALUES (?, ?, ?)",
                    [(1, 3, 1), (2, 5, 2)],
                )
                if optional_tables:
                    connection.executescript(
                        """
                        CREATE TABLE rocpd_hsaApi (
                            id INTEGER, pid INTEGER, tid INTEGER, start INTEGER,
                            end INTEGER, apiName_id INTEGER, args_id INTEGER
                        );
                        INSERT INTO rocpd_hsaApi VALUES (1, 30, 40, 3000, 4000, 2, 2);
                        CREATE TABLE rocpd_graphLaunchapi (api_ptr_id INTEGER, graphExec TEXT);
                        INSERT INTO rocpd_graphLaunchapi VALUES (3, '0x1234');
                        CREATE TABLE rocpd_monitor (
                            deviceId INTEGER, monitorType TEXT, start INTEGER,
                            end INTEGER, value INTEGER
                        );
                        """
                    )
                    connection.execute(
                        "INSERT INTO rocpd_monitor VALUES (0, ?, 1000, 30000, 42)",
                        (text,),
                    )
                connection.commit()
            bounds = {"start": None, "end": None} if optional_tables else {}
            rpd_to_chrome_trace(database, output, format=trace_format, **bounds)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
        return trace["traceEvents"] if trace_format == "object" else trace

    def test_text_round_trip(self):
        for trace_format in ("object", "array"):
            for text in (
                "ordinary marker",
                'marker "quoted"',
                r"C:\tmp\new",
                "line\n\ttab\rreturn\bbackspace\fformfeed\x01",
                "marker \u03bb \u6d4b\u8bd5",
            ):
                with self.subTest(trace_format=trace_format, text=text):
                    events = self.export(text, trace_format)
                    markers = [
                        e
                        for e in events
                        if e.get("name") == text and e.get("pid") == "10"
                    ]
                    self.assertEqual([e["ph"] for e in markers], ["i", "X", "X"])
                    self.assertTrue(all(e["args"]["desc"] == text for e in markers))
                    apis = [e for e in events if e.get("name") == "hipLaunchKernel"]
                    self.assertEqual(len(apis), 2)
                    self.assertTrue(all(e["args"]["desc"] == text for e in apis))
                    ops = [
                        e for e in events if e.get("pid") == "0" and e.get("ph") == "X"
                    ]
                    self.assertEqual(len(ops), 3)
                    self.assertTrue(all(e["name"] == text for e in ops))
                    self.assertEqual(
                        [e["args"]["desc"] for e in ops],
                        [text, text, "UserMarker frame: 1 ops"],
                    )
                    self.assertEqual([e["ts"] for e in apis], ["3.0", "21.0"])
                    self.assertEqual([e["dur"] for e in apis], ["1.0", "1.0"])
                    self.assertEqual(sum(e.get("cat") == "api_op" for e in events), 4)
                    counters = [e for e in events if e.get("ph") == "C"]
                    self.assertTrue(counters)
                    self.assertTrue(
                        all(isinstance(e["ts"], (int, float)) for e in counters)
                    )

    def test_empty_description_uses_op_type(self):
        for trace_format in ("object", "array"):
            with self.subTest(trace_format=trace_format):
                text = 'op "quoted" \\ path'
                events = self.export(text, trace_format, description="")
                ops = [
                    e
                    for e in events
                    if e.get("pid") == "0" and e.get("args", {}).get("desc") == text
                ]
                self.assertEqual(len(ops), 2)
                self.assertTrue(all(e["name"] == text for e in ops))

    def test_optional_event_text(self):
        for trace_format in ("object", "array"):
            with self.subTest(trace_format=trace_format):
                text = 'label "quoted" \\ path\n\t'
                events = self.export(text, trace_format, optional_tables=True)
                hsa = [e for e in events if e.get("pid") == "30" and e.get("ph") == "X"]
                self.assertEqual(len(hsa), 1)
                self.assertEqual(hsa[0]["name"], text)
                self.assertEqual(hsa[0]["args"], {"desc": text})
                monitors = [
                    e for e in events if e.get("ph") == "C" and e.get("name") == text
                ]
                self.assertEqual(len(monitors), 2)
                self.assertTrue(all(e["args"] == {text: 42} for e in monitors))
                graphs = [e for e in events if e.get("name") == "Graph 0x1234"]
                self.assertEqual(len(graphs), 1)
                self.assertEqual(graphs[0]["args"], {"kernels": "1"})


if __name__ == "__main__":
    unittest.main()
