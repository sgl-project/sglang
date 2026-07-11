import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def get_a2a_backend_config():
    # On Blackwell or machines where DeepEP is hard to compile, set
    # SGLANG_EPLB_TEST_MOE_A2A_BACKEND=flashinfer to use FlashInfer A2A.
    moe_a2a_backend = os.environ.get("SGLANG_EPLB_TEST_MOE_A2A_BACKEND", "deepep")
    args = ["--moe-a2a-backend", moe_a2a_backend]
    kwargs = {"moe_a2a_backend": moe_a2a_backend}
    if moe_a2a_backend == "deepep":
        args.extend(["--deepep-mode", "normal"])
        kwargs["deepep_mode"] = "normal"
    elif moe_a2a_backend == "flashinfer":
        args.extend(["--moe-runner-backend", "flashinfer_cutlass"])
        kwargs["moe_runner_backend"] = "flashinfer_cutlass"
    return args, kwargs


def get_a2a_backend_args():
    return get_a2a_backend_config()[0]


def get_a2a_backend_kwargs():
    return get_a2a_backend_config()[1]


class _BaseTestDynamicEPLB(CustomTestCase):
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
            envs.SGLANG_EXPERT_LOCATION_UPDATER_CANARY.override(True),
        ):
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
                    *get_a2a_backend_args(),
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
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(5)

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
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                disable_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
                tp_size=2,
                dp_size=2,
                log_level="info",
                # TODO pr-chain: enable later
                # enable_expert_distribution_metrics=True,
            )
            engine_kwargs.update(get_a2a_backend_kwargs())

            print(f"Action: start engine")
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)
            engine = sgl.Engine(
                **engine_kwargs,
                disable_overlap_schedule=True,
                port=11000,
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
            engine = None
            time.sleep(5)

            print(f"Action: start engine with init_expert_location")
            engine = sgl.Engine(
                **engine_kwargs,
                init_expert_location=str(snapshot_path),
                # TODO auto determine these flags
                ep_dispatch_algorithm="static",
                port=12000,
            )
            self._assert_engine_generate_correct(engine)
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine
            engine = None
            time.sleep(5)

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
