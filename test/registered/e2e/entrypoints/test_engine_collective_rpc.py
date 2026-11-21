"""Engine.collective_rpc and save_sharded_model under each parallel layout."""

import glob
import os
import shutil
import tempfile
import unittest

from huggingface_hub import snapshot_download

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=360, stage="nightly", runner_config="2-gpu-large")

# Its attention splits heads by attn_tp_size, as DP attention needs; Llama splits by tp_size.
MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
SAMPLING = {"temperature": 0, "max_new_tokens": 8}


class TestEngineCollectiveRpc(CustomTestCase):
    def _run_layout(self, num_shards, port, **parallel):
        """Check that every call gets its own reply, the ranks' error reaches
        the caller, and the save writes one file set per shard; return the save
        directory."""
        # Each engine gets its own ports: one closed in this process may not be free yet.
        engine = sgl.Engine(
            model_path=MODEL, mem_fraction_static=0.5, port=port, **parallel
        )
        try:
            # Two errors in turn: a stale reply from an earlier call would
            # answer a later one with the other error.
            for _ in range(2):
                with self.assertRaisesRegex(
                    RuntimeError, "allowed: save_remote_model, save_sharded_model"
                ):
                    engine.collective_rpc("feed_watchdog")
                with self.assertRaisesRegex(RuntimeError, "'url'"):
                    engine.collective_rpc("save_remote_model")

            path = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, path, ignore_errors=True)
            engine.save_sharded_model(path=path, pattern=None, max_size=None)
            shards = {
                os.path.basename(f).split("-part-")[0]
                for f in glob.glob(os.path.join(path, "*.safetensors"))
            }
            self.assertEqual(len(shards), num_shards, shards)
            return path
        finally:
            engine.shutdown()

    def test_attention_dp(self):
        self._run_layout(num_shards=2, port=31000, tp_size=2, attn_dp_size=2)

    def test_attention_dp_local_control_broadcast(self):
        self._run_layout(
            num_shards=2,
            port=32000,
            tp_size=2,
            attn_dp_size=2,
            enable_dp_attention_local_control_broadcast=True,
        )

    def test_data_parallel_saves_one_copy(self):
        self._run_layout(num_shards=1, port=33000, dp_size=2)

    def test_pipeline_parallel_save_loads_back(self):
        path = self._run_layout(num_shards=2, port=34000, pp_size=2)
        metadata = snapshot_download(
            MODEL, ignore_patterns=["*.safetensors", "*.bin", "*.pt"]
        )
        shutil.copytree(metadata, path, dirs_exist_ok=True)
        engine = sgl.Engine(
            model_path=path,
            load_format="sharded_state",
            mem_fraction_static=0.5,
            port=35000,
            pp_size=2,
        )
        try:
            # Loading raises on a missing or unexpected key, so stages that
            # overwrote each other's shards would fail here.
            self.assertTrue(engine.generate(PROMPT, SAMPLING)["text"])
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
