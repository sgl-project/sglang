import json
import struct
import tempfile
import unittest
from pathlib import Path

from sglang.srt.models.qwen4_ple_nvme import (
    IoUringPageRowReader,
    MMapRowReader,
    PLEManifest,
    RowLocation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class TestQwen4PLENvmeManifest(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.snapshot = Path(self.temp.name)
        prefix = "model.layers.2.ple.ple_embedding.ngram_embedding"
        self.names = [f"{prefix}.shard_{index}.weight" for index in range(2)]
        _write_safetensors(
            self.snapshot / "model-00001-of-00002.safetensors",
            {self.names[0]: ("F8_E4M3", (2, 4), bytes(range(8)))},
        )
        _write_safetensors(
            self.snapshot / "model-00002-of-00002.safetensors",
            {self.names[1]: ("F8_E4M3", (1, 4), bytes(range(8, 12)))},
        )
        index = {
            "metadata": {},
            "weight_map": {
                self.names[0]: "model-00001-of-00002.safetensors",
                self.names[1]: "model-00002-of-00002.safetensors",
            },
        }
        (self.snapshot / "model.safetensors.index.json").write_text(json.dumps(index))

    def tearDown(self):
        self.temp.cleanup()

    def test_maps_global_rows_to_exact_safetensors_ranges(self):
        manifest = PLEManifest.from_snapshot(self.snapshot, expected_shards=2)
        self.assertEqual(manifest.dtype, "F8_E4M3")
        self.assertEqual(manifest.embedding_dim, 4)
        self.assertEqual(manifest.total_rows, 3)
        self.assertEqual(manifest.row_bytes, 4)

        reader = MMapRowReader(manifest)
        self.addCleanup(reader.close)
        self.assertEqual(
            reader.read_rows([2, 0, 1]),
            [bytes(range(8, 12)), bytes(range(4)), bytes(range(4, 8))],
        )

    def test_rejects_noncontiguous_shards(self):
        index_path = self.snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        second = index["weight_map"].pop(self.names[1])
        index["weight_map"][self.names[1].replace("shard_1", "shard_2")] = second
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            PLEManifest.from_snapshot(self.snapshot)

    def test_enumerates_every_page_spanned_by_a_row(self):
        reader = object.__new__(IoUringPageRowReader)
        reader.page_size = 4096
        path = self.snapshot / "model-00001-of-00002.safetensors"
        self.assertEqual(
            reader._page_keys(RowLocation(path=path, offset=100, nbytes=9000)),
            ((path, 0), (path, 4096), (path, 8192)),
        )


if __name__ == "__main__":
    unittest.main()
