import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class _BaseTestDynamicEPLB(CustomTestCase):
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-deepep-moe",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                # TODO pr-chain: enable later
                # "--enable-expert-distribution-metrics",
                # TODO auto determine these flags
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "static",
                *cls.extra_args,
            ],
            env={
                "SGL_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestDynamicEPLBSimple(_BaseTestDynamicEPLB):
    pass


class TestDynamicEPLBMultiChunk(_BaseTestDynamicEPLB):
    extra_args = ["--eplb-rebalance-layers-per-chunk", "1"]


class TestStaticEPLB(CustomTestCase):
    def test_save_expert_distribution_and_init_expert_location(self):
        os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "0"

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                enable_deepep_moe=True,
                deepep_mode="normal",
                disable_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
                tp_size=2,
                dp_size=2,
                log_level="info",
                # TODO pr-chain: enable later
                # enable_expert_distribution_metrics=True,
            )

            print(f"Action: start engine")
            os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = tmp_dir
            engine = sgl.Engine(
                **engine_kwargs,
                disable_overlap_schedule=True,
            )
            engine.start_expert_distribution_record()
            self._assert_engine_generate_correct(engine)

            print(f"Action: dump_expert_distribution_record")
            engine.dump_expert_distribution_record()
            snapshot_path = list(Path(tmp_dir).glob("*.pt"))[0]
            assert snapshot_path is not None
            print(f"{snapshot_path=}")

            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

            print(f"Action: start engine with init_expert_location")
            engine = sgl.Engine(
                **engine_kwargs,
                init_expert_location=str(snapshot_path),
                port=21000,
                # TODO auto determine these flags
                ep_dispatch_algorithm="static",
            )
            self._assert_engine_generate_correct(engine)
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

    def _assert_engine_generate_correct(self, engine: sgl.Engine):
        output = engine.generate(
            prompt=["1+1=2, 2+2=4", "One plus one is two, two plus two is four"],
            sampling_params=dict(max_new_tokens=8, temperature=0.0),
        )
        print(f"engine.generate {output=}")
        self.assertEqual(
            [x["text"] for x in output],
            [", 4+4=8,", ", four plus four is eight, eight"],
        )


if __name__ == "__main__":
    unittest.main()
