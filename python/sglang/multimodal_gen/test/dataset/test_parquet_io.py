import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sgl_diffusion.dataset.dataloader.parquet_io import (
    ParquetDatasetWriter,
    records_to_table,
)


def test_records_to_table_types():
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vae_latent_bytes", pa.binary()),
            pa.field("vae_latent_shape", pa.list_(pa.int64())),
            pa.field("duration_sec", pa.float64()),
            pa.field("width", pa.int64()),
        ]
    )
    records = [
        {
            "id": "a",
            "vae_latent_bytes": b"\x00\x01",
            "vae_latent_shape": [1, 2, 3],
            "duration_sec": 1.5,
            "width": 640,
        }
    ]

    table = records_to_table(records, schema)
    assert table.schema == schema
    assert table.num_rows == 1
    cols = {name: table.column(name).to_pylist()[0] for name in schema.names}
    assert cols["id"] == "a"
    assert isinstance(cols["vae_latent_bytes"], (bytes, bytearray))
    assert cols["vae_latent_shape"] == [1, 2, 3]
    assert abs(cols["duration_sec"] - 1.5) < 1e-6
    assert cols["width"] == 640


def test_writer_flush_and_remainder(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string())])
    records = [{"id": str(i)} for i in range(25)]
    table = records_to_table(records, schema)

    out_dir = tmp_path / "out"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=10)
    writer.append_table(table)
    written = writer.flush(num_workers=1)
    assert written == 20

    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 2
    total_rows = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total_rows == 20

    # Append remainder to complete another chunk
    extra = records_to_table([{"id": str(i)} for i in range(5)], schema)
    writer.append_table(extra)
    written2 = writer.flush(num_workers=1)
    assert written2 == 10
    files2 = sorted(out_dir.rglob("*.parquet"))
    assert len(files2) == 3
    total_rows2 = sum(pq.read_table(str(f)).num_rows for f in files2)
    assert total_rows2 == 30


def test_writer_flush_write_remainder(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string())])
    # 25 rows, 10 per file => 2 full files + 1 remainder(5)
    records = [{"id": str(i)} for i in range(25)]
    table = records_to_table(records, schema)

    out_dir = tmp_path / "out_last"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=10)
    writer.append_table(table)
    # First flush writes 20
    written1 = writer.flush(num_workers=1)
    assert written1 == 20
    # Final flush with remainder
    written2 = writer.flush(num_workers=1, write_remainder=True)
    assert written2 == 5
    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 3
    total_rows = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total_rows == 25


def test_writer_parallel_workers(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string())])
    # 40 rows, 10 per file => 4 files
    records = [{"id": str(i)} for i in range(40)]
    table = records_to_table(records, schema)

    out_dir = tmp_path / "out_parallel"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=10)
    writer.append_table(table)
    written = writer.flush(num_workers=2)
    assert written == 40

    # Ensure files exist under worker subdirs
    worker_dirs = [
        p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("worker_")
    ]
    assert len(worker_dirs) >= 1
    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 4
    total_rows = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total_rows == 40
