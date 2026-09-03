import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _EnvField:
    def __init__(self, name):
        self.name = name

    def get(self):
        return os.getenv(self.name)


fake_environ = types.ModuleType("sglang.srt.environ")
fake_environ.envs = SimpleNamespace(
    SGLANG_STATE_OBSERVER_DIR=_EnvField("SGLANG_STATE_OBSERVER_DIR"),
    SGLANG_STATE_OBSERVER_RID_PREFIX=_EnvField("SGLANG_STATE_OBSERVER_RID_PREFIX"),
)
sys.modules["sglang.srt.environ"] = fake_environ
module_path = (
    Path(__file__).parents[4]
    / "python/sglang/srt/observability/state_mapping_observer.py"
)
module_spec = importlib.util.spec_from_file_location(
    "state_mapping_observer_under_test", module_path
)
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
StateMappingObserver = module.StateMappingObserver


class _Mode(Enum):
    TARGET_VERIFY = auto()


class _Algorithm(Enum):
    EAGLE = auto()


@dataclass(frozen=True)
class FakeParallelState:
    tp_rank: int = 3
    tp_size: int = 8
    pp_rank: int = 1
    pp_size: int = 2
    dp_rank: int = 0
    dp_size: int = 1
    attn_tp_rank: int = 3
    attn_tp_size: int = 8
    attn_cp_rank: int = 0
    attn_cp_size: int = 1
    attn_dcp_rank: int = 3
    attn_dcp_size: int = 8
    attn_dp_rank: int = 0
    attn_dp_size: int = 1
    moe_ep_rank: int = 3
    moe_ep_size: int = 8
    moe_dp_rank: int = 0
    moe_dp_size: int = 1
    gpu_id: int = 3


class FakeTensor:
    def __init__(self, values, shape=None):
        self.values = values if isinstance(values, list) else [values]
        self.shape = tuple(shape if shape is not None else [len(self.values)])

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_args):
        return self

    def tolist(self):
        return self.values

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, positions = key
            return FakeTensor([self.values[row][position] for position in positions])
        value = self.values[key]
        return FakeTensor(value) if isinstance(value, list) else value


class FakeDSAPool:
    page_size = 64
    index_head_dim = 128
    index_kpool = 4
    index_kpool_compress = True
    kpool_use_compress = True
    tail_extra_slots = 2
    index_key_cache = object()


class FakeHybridPool:
    page_size = 64
    use_mla = True
    use_dsa = True
    full_attention_layer_id_mapping = {0: 0, 3: 1}

    def __init__(self):
        self.full_kv_pool = FakeDSAPool()


class FakeReqPool:
    mamba_map = {1: 0, 2: 1}

    def __init__(self):
        self.req_to_token = FakeTensor(
            [[0] * 8, [0] * 8, [0] * 8, [101, 102, 103, 104, 105, 106, 107, 108]],
            shape=[4, 8],
        )
        self.req_index_to_mamba_index_mapping = FakeTensor([0, 0, 0, 9])


class FakeRunner:
    def __init__(self, start, end, req_pool, kv_pool):
        self.layer_info = SimpleNamespace(start_layer=start, end_layer=end)
        self.model = SimpleNamespace()
        self.req_to_token_pool = req_pool
        self.token_to_kv_pool = kv_pool


class FakeSpecWorker:
    def __init__(self, runners):
        self._runners = runners

    def _draft_model_runners(self):
        return tuple(self._runners)


def _make_scheduler():
    req_pool = FakeReqPool()
    target_pool = FakeHybridPool()
    target_runner = FakeRunner(0, 24, req_pool, target_pool)
    draft_runner = FakeRunner(24, 25, req_pool, FakeHybridPool())
    return SimpleNamespace(
        ps=FakeParallelState(),
        world_group=SimpleNamespace(rank=11, local_rank=3),
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: target_pool),
        tp_worker=SimpleNamespace(model_runner=target_runner),
        model_worker=FakeSpecWorker([draft_runner]),
        disaggregation_mode=SimpleNamespace(name="DECODE"),
    )


def _make_batch(rid="glm53-state-proof-001"):
    kv = SimpleNamespace(
        req_pool_idx=3,
        kv_committed_len=5,
        kv_allocated_len=7,
        cache_protected_len=2,
        mamba_pool_idx=FakeTensor([9]),
        mamba_ping_pong_track_buffer=FakeTensor([9, 10]),
        mamba_next_track_idx=1,
        mamba_last_track_idx=0,
        mamba_last_track_seqlen=6,
    )
    req = SimpleNamespace(
        rid=rid, origin_input_ids=[41, 42, 43, 44], seqlen=7, kv=kv
    )
    spec = SimpleNamespace(
        topk_p=FakeTensor([123456.0] * 4, [1, 4]),
        topk_index=FakeTensor([123456] * 4, [1, 4]),
        hidden_states=FakeTensor([123456.0] * 16, [1, 16]),
        bonus_tokens=FakeTensor([123456], [1]),
        dsa_topk_indices=FakeTensor([123456] * 4, [1, 4]),
        draft_probs=None,
    )
    return SimpleNamespace(
        reqs=[req],
        forward_iter=17,
        spec_algorithm=_Algorithm.EAGLE,
        forward_mode=_Mode.TARGET_VERIFY,
        spec_info=spec,
        mamba_track_indices=FakeTensor([9]),
        mamba_track_buffer_indices=[1],
    )


class TestStateMappingObserver(unittest.TestCase):
    def _env(self, output_dir, prefix="glm53-state-proof-"):
        return patch.dict(
            os.environ,
            {
                "SGLANG_STATE_OBSERVER_DIR": output_dir,
                "SGLANG_STATE_OBSERVER_RID_PREFIX": prefix,
                "SGLANG_STATE_OBSERVER_VARIANT": "pd-pp2-bf16-mtp",
                "SGLANG_STATE_OBSERVER_ROLE": "Decode",
                "SGLANG_SOURCE_COMMIT": "commit-id",
                "SGLANG_SOURCE_TREE": "tree-id",
                "SGLANG_SOURCE_ARCHIVE_SHA256": "archive-id",
                "POD_NAME": "decode-pod",
                "POD_UID": "pod-uid",
                "NODE_NAME": "node-181",
            },
            clear=False,
        )

    def test_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "SGLANG_STATE_OBSERVER_DIR": "",
                "SGLANG_STATE_OBSERVER_RID_PREFIX": "",
            },
        ):
            self.assertIsNone(StateMappingObserver.from_scheduler(_make_scheduler()))
            self.assertEqual(os.listdir(tmpdir), [])

    def test_non_matching_rid_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir, self._env(tmpdir):
            observer = StateMappingObserver.from_scheduler(_make_scheduler())
            observer.observe(_make_batch("ordinary-request"), "pre_forward")
            self.assertEqual(os.listdir(tmpdir), [])

    def test_matching_rid_records_concrete_pre_and_post_state(self):
        with tempfile.TemporaryDirectory() as tmpdir, self._env(tmpdir):
            observer = StateMappingObserver.from_scheduler(_make_scheduler())
            batch = _make_batch()
            next_draft = SimpleNamespace(
                topk_p=FakeTensor([123456.0] * 4, [1, 4]),
                topk_index=FakeTensor([123456] * 4, [1, 4]),
                hidden_states=FakeTensor([123456.0] * 16, [1, 16]),
                bonus_tokens=FakeTensor([123456], [1]),
                dsa_topk_indices=FakeTensor([123456] * 4, [1, 4]),
                draft_probs=None,
            )
            observer.observe(batch, "pre_forward")
            observer.observe(
                batch, "post_forward", SimpleNamespace(next_draft_input=next_draft)
            )

            files = os.listdir(tmpdir)
            self.assertEqual(files, ["state-mapping-Decode-rank-11.jsonl"])
            with open(os.path.join(tmpdir, files[0]), encoding="utf-8") as stream:
                raw = stream.read()
            rows = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual([row["event"]["phase"] for row in rows], ["pre_forward", "post_forward"])
            row = rows[1]
            self.assertEqual(row["request"]["rid"], "glm53-state-proof-001")
            self.assertEqual(row["request"]["req_pool_idx"], 3)
            self.assertEqual(row["request"]["token_slots"], {"positions": [0, 4, 6], "slot_ids": [101, 105, 107]})
            self.assertEqual(row["target"]["layers"], [0, 24])
            self.assertEqual(row["draft"]["runners"][0]["layers"], [24, 25])
            self.assertTrue(row["draft"]["shares_target_req_pool"])
            self.assertEqual(row["dsa"]["tail_state_indices"], [3, 4, 2, 0, 1, 6])
            self.assertEqual(row["mamba"]["layer_mapping"], {"1": 0, "2": 1})
            self.assertEqual(row["mamba"]["pool_idx"], [9])
            self.assertEqual(row["mamba"]["ping_pong_slots"], [9, 10])
            self.assertEqual(row["speculative"]["next_draft_input_shapes"]["hidden_states"], [1, 16])
            self.assertEqual(row["context"]["pod_uid"], "pod-uid")
            self.assertNotIn("123456", raw)
            self.assertNotIn("0x", raw)


if __name__ == "__main__":
    unittest.main()
