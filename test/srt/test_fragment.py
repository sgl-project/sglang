import multiprocessing
import multiprocessing as mp
import os
import traceback
import unittest
from multiprocessing import Process

import torch
from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server.engine_fragment import EngineFragment
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import is_in_ci
from test.srt.models.test_generation_models import check_close_model_outputs, CI_MODELS, ALL_OTHER_MODELS
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

_TP_SIZE = 2
_MAX_NEW_TOKENS = 8
_PROMPTS = ["Today is a sunny day and I like", "I have a very good idea on"]


class TestFragment(unittest.TestCase):
    def assert_fragment_e2e_execution(self, index: int, model_path: str):
        multiprocessing.set_start_method("spawn")

        # different port for maximum isolation
        nccl_port = 12345 + index
        master_port = 23456 + index

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(_TP_SIZE):
            p = Process(
                target=_run_subprocess,
                args=(tp_rank, master_port, nccl_port, output_writer, model_path),
            )
            p.start()
            processes.append(p)

        for _ in range(_TP_SIZE):
            self.assertTrue(output_reader.recv(), 'Subprocess has error, please see logs above')

        for p in processes:
            p.join()

    # def test_ci_models(self):
    #     for index, model_case in enumerate(CI_MODELS):
    #         self.assert_fragment_e2e_execution(index=index, model_path=model_case.model_path)
    #
    # def test_others(self):
    #     if is_in_ci():
    #         return
    #
    #     for index, model_case in enumerate(ALL_OTHER_MODELS):
    #         self.assert_fragment_e2e_execution(index=index, model_path=model_case.model_path)
    #
    def test_temp(self):
        self.assert_fragment_e2e_execution(index=0, model_path="meta-llama/Llama-3.2-1B-Instruct")


def _run_subprocess(tp_rank: int, master_port: int, nccl_port: int, output_writer, model_path: str):
    try:
        print(f"subprocess[{tp_rank=}] Start {os.environ['CUDA_VISIBLE_DEVICES']=}")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(rank=tp_rank, world_size=_TP_SIZE)

        mesh_kwargs = dict(mesh_shape=(_TP_SIZE, 1), mesh_dim_names=["tp", "pp"])
        inference_device_mesh_device = init_device_mesh("cuda", **mesh_kwargs)
        inference_device_mesh_cpu = init_device_mesh("cpu", **mesh_kwargs)
        print(
            f"subprocess[{tp_rank=}] {inference_device_mesh_device=} {inference_device_mesh_cpu=}"
        )

        fragment = EngineFragment(
            model_path=model_path,
            load_format='dummy',
            mem_fraction_static=0.1,
            tp_size=_TP_SIZE,
            random_seed=42,
            # fragment args
            tp_rank=tp_rank,
            gpu_id=tp_rank,
            nccl_port=nccl_port,
            parallel_process_groups=ParallelProcessGroups.from_devices_meshes(
                device_mesh_device=inference_device_mesh_device,
                device_mesh_cpu=inference_device_mesh_cpu,
                dim_tp="tp",
                dim_pp="pp",
            ),
        )
        print(f"subprocess[{tp_rank=}] {fragment=}", flush=True)

        # hf model is used for comparison
        with torch.device("cuda"):
            hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            hf_model.to(torch.bfloat16)
        hf_model.cuda()
        hf_tokenizer = get_tokenizer(model_path)

        hf_outputs = HFRunner.forward_generation_raw(
            prompts=_PROMPTS,
            max_new_tokens=_MAX_NEW_TOKENS,
            base_model=hf_model,
            tokenizer=hf_tokenizer,
            lora_paths=None,
            torch_dtype=torch.float16,
            output_str_only=False,
        )

        # test update weights
        fsdp_state_dict = _get_fsdp_state_dict(hf_model=hf_model)
        print(
            f"subprocess[{tp_rank=}] call update_weights_from_tensor ({list(fsdp_state_dict.keys())=})",
            flush=True,
        )
        fragment.update_weights_from_tensor(
            [(k, v) for k, v in fsdp_state_dict.items()]
        )

        srt_outputs = SRTRunner.forward_generation_raw(
            prompts=_PROMPTS,
            max_new_tokens=_MAX_NEW_TOKENS,
            lora_paths=None,
            engine=fragment,
        )
        print(
            f"subprocess[{tp_rank=}] call srt.forward {srt_outputs=}",
            flush=True,
        )

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=5e-2,
            decode_tolerance=5e-2,
            rouge_l_tolerance=1,
        )

        execution_ok = False

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = True

    output_writer.send(execution_ok)
    output_writer.close()

    fragment.shutdown()
    print(f"subprocess[{tp_rank=}] end", flush=True)


# Adapted from https://github.com/volcengine/verl/blob/main/tests/rollout/run_fsdp_vllm.py
def _get_fsdp_state_dict(hf_model):
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(_TP_SIZE,), mesh_dim_names=["fsdp"]
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

    state_dict = fsdp_model.state_dict()
    # small models have weight tieing, thus we skip it
    return {k: v for k, v in state_dict.items() if k not in ["lm_head.weight"]}


if __name__ == "__main__":
    unittest.main()
