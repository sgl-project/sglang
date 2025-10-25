from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from sgl_diffusion.runtime.pipelines.pipeline_batch_info import PreprocessBatch
from sgl_diffusion.runtime.workflow.preprocess.components import (
    ParquetDatasetSaver,
)


def _simple_record_creator(batch: PreprocessBatch) -> list[dict]:
    # batch.latents will be converted to numpy by the saver before this call
    assert isinstance(batch.latents, np.ndarray)
    num = len(batch.video_file_name)
    records = []
    for i in range(num):
        arr = batch.latents[i]
        records.append(
            {
                "id": batch.video_file_name[i],
                "data_bytes": arr.tobytes(),
                "data_shape": list(arr.shape),
            }
        )
    return records


def test_parquet_dataset_saver_flush_and_last(tmp_path: Path):
    # Schema for the simple record creator
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("data_bytes", pa.binary()),
            pa.field("data_shape", pa.list_(pa.int64())),
        ]
    )

    B = 5
    # Build a minimal PreprocessBatch
    batch = PreprocessBatch(
        data_type=Datatype,
        latents=torch.randn(B, 2),
        prompt_embeds=[torch.randn(B, 1, 1)],
        # Attention mask should be integer dtype in real pipelines
        prompt_attention_mask=[torch.ones(B, 1, dtype=torch.int64)],
    )
    batch.video_file_name = [f"vid_{i}" for i in range(B)]

    saver = ParquetDatasetSaver(
        flush_frequency=10,  # higher than B to avoid auto-flush
        samples_per_file=3,
        schema=schema,
        record_creator=_simple_record_creator,
    )

    out_dir = tmp_path / "saver_out"
    saver.save_and_write_parquet_batch(batch, str(out_dir))
    # First flush: should write one full file (3 rows), keep 2 in buffer
    saver.flush_tables()
    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 1
    assert pq.read_table(str(files[0])).num_rows == 3

    # Final flush: write remainder 2 rows
    saver.flush_tables(write_remainder=True)
    files2 = sorted(out_dir.rglob("*.parquet"))
    assert len(files2) == 2
    total = sum(pq.read_table(str(f)).num_rows for f in files2)
    assert total == 5
