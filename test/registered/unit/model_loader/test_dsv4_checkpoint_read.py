"""CPU tests for the DSV4.1 per-rank checkpoint reader."""

from __future__ import annotations

import os
import struct
import json

import torch

from sglang.srt.layers.engram import engram_shard_rows
from sglang.srt.model_loader.dsv4_checkpoint_read import (
    DSV4CheckpointReader,
    contiguous_owned_expert_ids,
    expert_id_in_checkpoint_name,
    is_engram_embed_table,
    owned_expert_ids_from_physical_map,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_MODEL = os.environ.get(
    "DSV41_CHECKPOINT", os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
)


class _Slice:
    def __init__(self, rows: int, cols: int):
        self._shape = [rows, cols]
        self.requested = None

    def get_shape(self):
        return self._shape

    def __getitem__(self, item):
        self.requested = item
        rows = item.stop - item.start
        self.tensor = torch.arange(rows * self._shape[1], dtype=torch.float32).view(
            rows, self._shape[1]
        )
        return self.tensor


class _Handle:
    def __init__(self, rows: int = 8, cols: int = 4):
        self.full = []
        self.slice = _Slice(rows, cols)

    def get_tensor(self, name):
        self.full.append(name)
        return torch.empty(1)

    def get_slice(self, name):
        return self.slice


def _reader(**kwargs) -> DSV4CheckpointReader:
    defaults = dict(
        owned_expert_ids=frozenset(range(2)),
        n_routed=4,
        tp_rank=0,
        tp_size=1,
        mtp_only=False,
        narrow_engram=False,
    )
    defaults.update(kwargs)
    return DSV4CheckpointReader(**defaults)


class TestDsv4CheckpointRead(CustomTestCase):
    def test_expert_id_parse(self):
        self.assertEqual(
            expert_id_in_checkpoint_name("layers.6.ffn.experts.17.w1.weight"), 17
        )
        self.assertEqual(
            expert_id_in_checkpoint_name("mtp.1.ffn.experts.3.w2.scale"), 3
        )
        self.assertIsNone(expert_id_in_checkpoint_name("layers.6.ffn.shared_experts.w1.weight"))
        self.assertTrue(is_engram_embed_table("layers.1.engram.embed.weight"))
        self.assertFalse(is_engram_embed_table("layers.1.engram.wkv.weight"))

    def test_contiguous_split_and_physical_map(self):
        self.assertEqual(contiguous_owned_expert_ids(384, 1, 8), frozenset(range(48, 96)))
        self.assertIsNone(contiguous_owned_expert_ids(384, 0, 1))
        self.assertIsNone(contiguous_owned_expert_ids(100, 0, 8))
        # Two layers, 4 physical, ep=2. Rank 1 owns physical 2 and 3.
        # Layer 0 maps them to logical 2, 3. Layer 1 maps them to logical 3, 1.
        mapping = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 1]])
        owned = owned_expert_ids_from_physical_map(mapping, ep_rank=1, ep_size=2, n_routed=4)
        self.assertEqual(owned, frozenset({1, 2, 3}))
        self.assertIsNone(
            owned_expert_ids_from_physical_map(mapping, ep_rank=0, ep_size=3, n_routed=4)
        )

    def test_skips_remote_experts_and_keeps_the_rest(self):
        reader = _reader()
        handle = _Handle()
        self.assertIsNone(reader("layers.0.ffn.experts.2.w1.weight", handle))
        self.assertEqual(handle.full, [])
        kept = reader("layers.0.ffn.experts.1.w1.weight", handle)
        self.assertIsNotNone(kept)
        # Id past n_routed is not a routed expert this filter understands.
        reader("layers.0.ffn.experts.9.w1.weight", handle)
        reader("layers.0.ffn.shared_experts.w1.weight", handle)
        self.assertEqual(
            handle.full,
            [
                "layers.0.ffn.experts.1.w1.weight",
                "layers.0.ffn.experts.9.w1.weight",
                "layers.0.ffn.shared_experts.w1.weight",
            ],
        )
        self.assertEqual(reader.expert_ids_read, {1})

    def test_draft_ignores_the_target_checkpoint(self):
        reader = _reader(mtp_only=True, owned_expert_ids=frozenset({0}))
        handle = _Handle()
        self.assertIsNone(reader("layers.6.ffn.experts.0.w1.weight", handle))
        self.assertIsNone(reader("layers.1.engram.embed.weight", handle))
        self.assertIsNotNone(reader("mtp.0.ffn.experts.0.w1.weight", handle))
        self.assertIsNone(reader("mtp.0.ffn.experts.1.w1.weight", handle))
        self.assertEqual(handle.full, ["mtp.0.ffn.experts.0.w1.weight"])

    def test_engram_slice_uses_the_same_rows_as_the_module(self):
        reader = _reader(narrow_engram=True, tp_rank=1, tp_size=2, owned_expert_ids=None)
        handle = _Handle(rows=100, cols=8)
        tensor = reader("layers.1.engram.embed.scale", handle)
        start, end = engram_shard_rows(100, 1, 2)
        self.assertEqual(handle.slice.requested, slice(start, end))
        self.assertEqual(tuple(tensor.shape), (end - start, 8))
        # The reader must hand back the mmap view. Cloning the shard OOMs
        # this host: eight ranks each private-copy ~12 GiB onto a 64 GiB node.
        self.assertEqual(tensor.data_ptr(), handle.slice.tensor.data_ptr())
        self.assertEqual(handle.full, [])

    def test_finish_raises_when_an_owned_expert_was_never_read(self):
        reader = _reader()
        handle = _Handle()
        reader("layers.0.ffn.experts.0.w1.weight", handle)
        with self.assertRaises(RuntimeError):
            reader.finish()

    def test_finish_accepts_a_complete_local_set(self):
        reader = _reader()
        handle = _Handle()
        reader("layers.0.ffn.experts.0.w1.weight", handle)
        reader("layers.0.ffn.experts.1.w1.weight", handle)
        reader.finish()

    def test_live_checkpoint_rank_keeps_only_its_experts(self):
        index = os.path.join(_MODEL, "model.safetensors.index.json")
        if not os.path.isfile(index):
            self.skipTest("DeepSeek-V4.1-Flash checkpoint is not on this machine")
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        owned = contiguous_owned_expert_ids(384, ep_rank=3, ep_size=8)
        self.assertIsNotNone(owned)
        kept_ids = set()
        remote_kept = []
        for name in weight_map:
            expert_id = expert_id_in_checkpoint_name(name)
            if expert_id is None or expert_id >= 384:
                continue
            if expert_id in owned:
                kept_ids.add(expert_id)
            else:
                remote_kept.append(name)
        self.assertEqual(kept_ids, set(owned))
        # The filter drops these; the checkpoint still contains them.
        self.assertGreater(len(remote_kept), 0)

        # Byte accounting from safetensors headers only.
        files = {}
        for name, filename in weight_map.items():
            files.setdefault(filename, []).append(name)
        kept_bytes = 0
        skipped_bytes = 0

        def nbytes(meta):
            n = 1
            for dim in meta["shape"]:
                n *= dim
            return n * {
                "F16": 2,
                "BF16": 2,
                "F32": 4,
                "F8_E4M3": 1,
                "F8_E8M0": 1,
                "U8": 1,
                "I8": 1,
                "I32": 4,
                "I64": 8,
            }.get(meta["dtype"], 1)

        root = _MODEL
        for filename, names in files.items():
            path = os.path.join(root, filename)
            with open(path, "rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            for name in names:
                meta = header[name]
                size = nbytes(meta)
                expert_id = expert_id_in_checkpoint_name(name)
                drop_expert = (
                    expert_id is not None
                    and expert_id < 384
                    and expert_id not in owned
                )
                drop_engram = is_engram_embed_table(name)
                if drop_expert:
                    skipped_bytes += size
                elif drop_engram:
                    # Rank keeps one TP slice. The other 7/8 are not read.
                    rows = meta["shape"][0]
                    start, end = engram_shard_rows(rows, 3, 8)
                    frac = (end - start) / rows
                    kept_bytes += int(size * frac)
                    skipped_bytes += size - int(size * frac)
                else:
                    kept_bytes += size
        # Remote experts alone are the bulk of what used to be read and discarded.
        self.assertGreater(skipped_bytes, kept_bytes * 3)

    def test_engram_slice_does_not_read_the_whole_table(self):
        path = os.path.join(_MODEL, "model-00048-of-00048.safetensors")
        if not os.path.isfile(path):
            self.skipTest("engram shard is not on this machine")
        import time

        import safetensors

        started = time.perf_counter()
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            view = handle.get_slice("layers.14.engram.embed.weight")
            rows = int(view.get_shape()[0])
            start, _end = engram_shard_rows(rows, 0, 8)
            shard = view[start : start + 4]
        elapsed = time.perf_counter() - started
        self.assertEqual(tuple(shard.shape), (4, 256))
        self.assertLess(elapsed, 5.0)
