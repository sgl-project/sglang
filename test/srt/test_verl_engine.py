import multiprocessing
import multiprocessing as mp
import os
import random
import traceback
import unittest
from multiprocessing import Process

import torch
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
from sglang.srt.utils import is_port_available
from sglang.test.runners import (
    HFRunner,
    SRTRunner,
    check_close_model_outputs,
    get_dtype_str,
)
from sglang.test.test_utils import CustomTestCase, find_available_port, is_in_ci

_MAX_NEW_TOKENS = 8
_PROMPTS = ["1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=", "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="]
_TORCH_DTYPE = torch.float16

# Set to false to temporarily debug issues unrelated to weight update
_ENABLE_UPDATE_WEIGHTS = True
# _ENABLE_UPDATE_WEIGHTS = False

# TODO maybe we should add more other models? should we keep it in sync with test_generation_models.py?
CI_MODELS = [
    dict(model_path="meta-llama/Llama-3.1-8B-Instruct"),
    # Fail to run gemma-2-2b after transformers==4.48.3 -> 4.50.0
    # dict(model_path="google/gemma-2-2b"),
]
ALL_OTHER_MODELS = [
    dict(model_path="meta-llama/Llama-3.2-1B-Instruct"),
    dict(model_path="Qwen/Qwen2-1.5B"),
    dict(
        model_path="Qwen/Qwen2.5-14B-Instruct",
        mem_fraction_static=0.4,
        tp_size=8,
        tight_memory=True,
        decode_tolerance=1.3,
    ),  # test_generation_models.py same config (qwen + tp=8) gives 1.22 decode error
    dict(model_path="HuggingFaceTB/SmolLM-135M-Instruct", tp_size=3),
    dict(model_path="allenai/OLMo-1B-0724-hf"),
    dict(
        model_path="THUDM/glm-4-9b-chat",
        mem_fraction_static=0.1,
        tp_size=8,
        tight_memory=True,
    ),
    dict(model_path="allenai/OLMo-2-1124-7B-Instruct"),
    dict(
        model_path="ibm-granite/granite-3.0-2b-instruct",
        prefill_tolerance=0.22,
        decode_tolerance=0.22,
    ),
    # Fail to run these models in test_generation_models.py, need to fix that first
    # dict(model_path="openai-community/gpt2"),
    # dict(model_path="microsoft/Phi-3-small-8k-instruct"),
]


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

    # def test_adhoc(self):
    #     self.assert_fragment_e2e_execution(index=0, model_path="meta-llama/Llama-3.2-1B-Instruct")


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
        )
        print(f"subprocess[{tp_rank=}] {engine=}", flush=True)

        if _ENABLE_UPDATE_WEIGHTS:
            print(f"subprocess[{tp_rank=}] call update_weights_from_tensor", flush=True)
            engine.update_weights_from_tensor(
                [(k, v) for k, v in fsdp_state_dict.items()]
            )

        for enable_batch in [False, True]:
            if enable_batch:
                fn = SRTRunner.batch_forward_generation_raw
            else:
                fn = SRTRunner.forward_generation_raw

            srt_outputs = fn(
                prompts=_PROMPTS,
                max_new_tokens=_MAX_NEW_TOKENS,
                lora_paths=None,
                engine=engine,
            )
            print(
                f"subprocess[{tp_rank=}] call srt.forward {enable_batch=} {srt_outputs=}",
                flush=True,
            )

            check_close_model_outputs(
                hf_outputs=hf_outputs,
                srt_outputs=srt_outputs,
                prefill_tolerance=prefill_tolerance,
                decode_tolerance=decode_tolerance,
                rouge_l_tolerance=1,
                check_logprobs=not enable_batch,
                debug_text=f"{enable_batch=} {tp_rank=}",
            )

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
