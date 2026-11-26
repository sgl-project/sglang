import json
import struct
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import PullWeightsReqInput  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_puller  # noqa: E402
from sglang.srt.weight_sync import local_checkpoint  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _pull_on_host(rank, root, retry):
    # One local filesystem namespace per rank; both see the same published files.
    root = Path(root)
    local = root / f"host-{rank}"
    dist.init_process_group(
        "gloo",
        init_method=(root / f"rendezvous-{int(retry)}").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        replies = []
        puller = weight_puller.SchedulerWeightPuller(
            tp_cpu_group=dist.group.WORLD,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda output, req: replies.append(output)
                )
            ),
        )
        req = PullWeightsReqInput(
            local_checkpoint_dir=str(local), source_dir=str(root), target_version=1
        )
        model = SimpleNamespace(
            model_path=str(root / "base"),
            download_dir=None,
            revision=None,
            custom_pull_weights_pre_read_hook=None,
        )
        with ExitStack() as st:
            for name, value in (
                ("get_model", model),
                (
                    "get_parallel",
                    SimpleNamespace(enable_dp_attention_local_control_broadcast=False),
                ),
                (
                    "get_exec",
                    SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=False)),
                ),
                ("download_weights_from_hf", model.model_path),
            ):
                st.enter_context(patch.object(weight_puller, name, return_value=value))
            if rank == 1 and not retry:
                # Fail after the first file has actually been copied to host 1.
                st.enter_context(
                    patch.object(
                        local_checkpoint,
                        "_drop_page_cache",
                        side_effect=OSError("injected host 1 I/O failure"),
                    )
                )
            if rank == 0 and retry:
                st.enter_context(
                    patch.object(
                        local_checkpoint.shutil,
                        "copy2",
                        side_effect=AssertionError("completed host copied again"),
                    )
                )
            assert puller.handle(req) is None
            deadline = time.monotonic() + 30
            while not replies and time.monotonic() < deadline:
                puller.check_pending()
                time.sleep(0.005)
            assert len(replies) == 1
            reply = replies[0]
            assert reply.success == retry, reply.message
            if not retry:
                assert "injected host 1 I/O failure" in reply.message
            version = local_checkpoint._read_applied_version(str(local))
            assert version == (1 if retry or rank == 0 else None)
            if retry or rank == 0:
                for source in (root / "weight_v000001").iterdir():
                    assert (local / source.name).read_bytes() == source.read_bytes()
            else:
                assert len([path for path in local.iterdir() if path.is_file()]) == 1
    finally:
        dist.destroy_process_group()


class TestPartialPullRetry(CustomTestCase):
    def test_restarted_ranks_converge_after_one_host_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            published = Path(scratch) / "weight_v000001"
            published.mkdir()
            weight_map = {}
            for shard in range(2):
                name = f"weight{shard}"
                filename = f"model-{shard + 1:05d}-of-00002.safetensors"
                weight_map[name] = filename
                header = json.dumps(
                    {name: {"dtype": "U8", "shape": [256], "data_offsets": [0, 256]}}
                ).encode()
                header += b" " * (-len(header) % 8)
                (published / filename).write_bytes(
                    struct.pack("<Q", len(header)) + header + bytes([shard]) * 256
                )
            (published / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
            for retry in (False, True):
                # Fresh scheduler processes on retry can use only persisted state.
                mp.spawn(_pull_on_host, args=(scratch, retry), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
