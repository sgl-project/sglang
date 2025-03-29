import multiprocessing
import multiprocessing as mp
import os
import random
import time
import traceback
import unittest
from multiprocessing import Process

import requests
import torch
from openai import OpenAI
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available
from sglang.test.runners import (
    HFRunner,
    SRTRunner,
    check_close_model_outputs,
    get_dtype_str,
)
from sglang.test.test_utils import CustomTestCase, is_in_ci

_MAX_NEW_TOKENS = 8
_PROMPTS = ["1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=", "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="]
_TORCH_DTYPE = torch.float16

# Set to false to temporarily debug issues unrelated to weight update
_ENABLE_UPDATE_WEIGHTS = True

CI_MODELS = ALL_OTHER_MODELS = [
    dict(
        model_path="Qwen/Qwen2.5-1.5B",
        tp_size=2,
    )
]


# TODO Ask: this is extracted from PortArgs.init_new, is it allowed to extract it, i.e. touch that old code
def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


PORT = find_available_port(2345)


class TestVerlEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        multiprocessing.set_start_method("spawn")

    def assert_fragment_e2e_execution(
        self,
        index: int,
        model_path: str,
        mem_fraction_static: float = 0.4,
        tp_size: int = 2,
        tight_memory: bool = False,
        prefill_tolerance: float = 0.1,
        decode_tolerance: float = 0.1,
    ):
        master_port = find_available_port(23456)

        print(f"assert_fragment_e2e_execution START {index=} {model_path=}")

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(tp_size):
            p = Process(
                target=_run_subprocess,
                kwargs=dict(
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    master_port=master_port,
                    output_writer=output_writer,
                    model_path=model_path,
                    mem_fraction_static=mem_fraction_static,
                    tight_memory=tight_memory,
                    prefill_tolerance=prefill_tolerance,
                    decode_tolerance=decode_tolerance,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(tp_size):
            self.assertTrue(
                output_reader.recv(),
                f"Subprocess has error, please see logs above. ({index=} {model_path=})",
            )

        for p in processes:
            p.join()

    def test_ci_models(self):
        for index, model_info in enumerate(CI_MODELS):
            self.assert_fragment_e2e_execution(index=index, **model_info)

    def test_others(self):
        if is_in_ci():
            return

        for index, model_info in enumerate(ALL_OTHER_MODELS):
            self.assert_fragment_e2e_execution(index=index, **model_info)


def _run_subprocess(
    tp_rank: int,
    tp_size: int,
    master_port: int,
    output_writer,
    model_path: str,
    mem_fraction_static: float,
    tight_memory: bool,
    prefill_tolerance: float,
    decode_tolerance: float,
):
    try:
        print(f"subprocess[{tp_rank=}] Start {os.environ.get('CUDA_VISIBLE_DEVICES')=}")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(rank=tp_rank, world_size=tp_size)
        torch.cuda.set_device(tp_rank)

        mesh_kwargs = dict(mesh_shape=(tp_size, 1), mesh_dim_names=["tp", "pp"])
        inference_device_mesh_device = init_device_mesh("cuda", **mesh_kwargs)
        inference_device_mesh_cpu = init_device_mesh("cpu", **mesh_kwargs)
        print(
            f"subprocess[{tp_rank=}] {inference_device_mesh_device=} {inference_device_mesh_cpu=}"
        )

        # hf model is used for comparison
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=_TORCH_DTYPE, trust_remote_code=True
        ).cuda()
        hf_tokenizer = get_tokenizer(model_path, trust_remote_code=True)

        hf_outputs = HFRunner.forward_generation_raw(
            base_model=hf_model,
            prompts=_PROMPTS,
            max_new_tokens=_MAX_NEW_TOKENS,
            tokenizer=hf_tokenizer,
            lora_paths=None,
            torch_dtype=_TORCH_DTYPE,
            output_str_only=False,
        )
        print(
            f"subprocess[{tp_rank=}] call hf.forward {hf_outputs=}",
            flush=True,
        )

        if _ENABLE_UPDATE_WEIGHTS:
            if tight_memory:
                hf_model.cpu()
                torch.cuda.empty_cache()

            # test update weights
            print(f"subprocess[{tp_rank=}] get_fsdp_state_dict", flush=True)
            fsdp_state_dict = _get_fsdp_state_dict(hf_model=hf_model, tp_size=tp_size)

        engine = VerlEngine(
            model_path=model_path,
            load_format="dummy" if _ENABLE_UPDATE_WEIGHTS else "auto",
            mem_fraction_static=mem_fraction_static,
            random_seed=42,
            trust_remote_code=True,
            dtype=get_dtype_str(_TORCH_DTYPE),
            device_mesh_cpu=inference_device_mesh_cpu["tp"],
            launch_server=True,
            server_args=ServerArgs(
                model_path=model_path,
                tp_size=tp_size,
                port=PORT,
                mem_fraction_static=0.5,
            ),
        )
        print(f"subprocess[{tp_rank=}] {engine=}", flush=True)

        # test direct generate API
        print(f"subprocess[{tp_rank=}] testing direct generate API")
        direct_response = engine.generate(
            prompt="Hello, world!",
            sampling_params={"temperature": 0.7, "max_new_tokens": 20},
        )
        print(f"Direct generate response: {direct_response}")

        # test memory occupation APIs
        print(f"subprocess[{tp_rank=}] testing memory occupation APIs")
        engine.release_memory_occupation()
        print("Memory released")
        time.sleep(1)
        engine.resume_memory_occupation()
        print("Memory resumed")
        time.sleep(1)
        # openai API test for reference
        torch.distributed.barrier()
        if tp_rank == 0:
            client = OpenAI(api_key="None", base_url=f"http://localhost:{PORT}/v1")
            print(client.models.list().data[0].id)
            url = f"http://localhost:{PORT}/generate"
            data = {"text": "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="}
            response = requests.post(url, json=data)
            print(response.json())
        if _ENABLE_UPDATE_WEIGHTS:
            print(f"subprocess[{tp_rank=}] call update_weights_from_tensor", flush=True)
            # check_tensor = [(k, v) for k, v in fsdp_state_dict.items()][0]
            # update_tensor = [check_tensor[0], torch.zeros_like(check_tensor[1])]
            engine.update_weights_from_tensor(
                [(k, v) for k, v in fsdp_state_dict.items()]
            )
            # if tp_rank == 0:
            #     response = requests.get(
            #         f"http://localhost:{PORT}/get_weights_by_name",
            #         json={"name": list(fsdp_state_dict.keys())[0], "truncate_size": 5},
            #         timeout=20,
            #     )
            #     print(response.json())

        execution_ok = True

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()

    engine.shutdown()
    print(f"subprocess[{tp_rank=}] end", flush=True)


# Adapted from https://github.com/volcengine/verl/blob/main/tests/rollout/run_fsdp_vllm.py
def _get_fsdp_state_dict(hf_model, tp_size: int):
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(tp_size,), mesh_dim_names=["fsdp"]
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    fsdp_model = FSDP(
        hf_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )
    print(f"{fsdp_model=}")

    FSDP.set_state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
    )

    return fsdp_model.state_dict()


if __name__ == "__main__":
    unittest.main()
