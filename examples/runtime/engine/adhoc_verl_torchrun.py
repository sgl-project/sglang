# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import List

import torch
import torch.nn.functional as F
from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.server.engine_fragment import EngineFragment
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


def main():
    assert torch.cuda.is_available(), 'CUDA must be present to run FSDP vLLM example'
    local_rank, rank, world_size = initialize_global_process_group()

    # NOTE MODIFIED path-related logic
    # local_cache_path = '~/.cache/verl/rlhf'
    # local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'Qwen/Qwen2-7B-Instruct'
    local_model_path = hdfs_path
    # from verl.utils.fs import copy_local_path_from_hdfs
    # local_model_path = copy_local_path_from_hdfs(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
        actor_model.to(torch.bfloat16)

    max_prompt_length = 16
    response_length = 32
    preencode_prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors='pt', padding=True,
                        padding_side='left')  # NOTE MODIFIED add
    input_ids = prompts['input_ids']
    attention_mask = prompts['attention_mask']
    # from verl.utils.torch_functional import pad_sequence_to_length
    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True).cuda()
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True).cuda()

    from transformers import GenerationConfig
    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        # max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        # renormalize_logits=True,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=False)  # may OOM when use_cache = True
    seq = output.sequences
    response = seq[:, max_prompt_length:]

    print(f'hf response: {tokenizer.batch_decode(response)}')

    tensor_model_parallel_size = 4
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    fsdp_model = FSDP(actor_model,
                      use_orig_params=True,
                      auto_wrap_policy=None,
                      device_id=torch.cuda.current_device(),
                      sharding_strategy=ShardingStrategy.FULL_SHARD,
                      mixed_precision=mixed_precision,
                      cpu_offload=CPUOffload(offload_params=False),
                      sync_module_states=False,
                      device_mesh=device_mesh)

    FSDP.set_state_dict_type(fsdp_model,
                             state_dict_type=StateDictType.SHARDED_STATE_DICT,
                             state_dict_config=ShardedStateDictConfig())

    state_dict = fsdp_model.state_dict()

    if rank == 0:
        lines = ['------------------------ state_dict ------------------------']
        for k, v in state_dict.items():
            v_local = v.to_local()
            lines.append(f'{k}\t: {v.shape=} {v_local.shape=} {v.dtype=} {v_local.dtype=} {type(v)=} {type(v_local)=}')
        print('\n'.join(lines))

    # NOTE MODIFIED
    # sampling_params = SamplingParams(temperature=0,
    #                                  top_p=1,
    #                                  n=1,
    #                                  max_tokens=response_length,
    #                                  logprobs=1,
    #                                  ignore_eos=True,
    #                                  detokenize=False)
    sampling_params = dict(temperature=0,
                           top_p=1,
                           n=1,
                           max_new_tokens=response_length,
                           ignore_eos=True)

    tp_size, dp_size = 4, 1
    kwargs = dict(mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=["tp", "dp", "pp"])
    inference_device_mesh_device = init_device_mesh("cuda", **kwargs)
    inference_device_mesh_cpu = init_device_mesh("cpu", **kwargs)
    print(f"{inference_device_mesh_device=} {inference_device_mesh_cpu=}")

    print(actor_model_config)
    # llm = LLM(model=None,
    #           tokenizer=tokenizer,
    #           model_hf_config=actor_model_config,
    #           tensor_parallel_size=tensor_model_parallel_size,
    #           enforce_eager=True,
    #           dtype='bfloat16',
    #           load_format='dummy_dtensor',
    #           gpu_memory_utilization=0.1,
    #           trust_remote_code=True)
    changed_model_path = local_model_path.replace('-Instruct', '')
    assert changed_model_path != local_model_path
    print(f'{changed_model_path=}')
    llm = EngineFragment(
        model_path=changed_model_path,  # use model of same type but different weight to test update_weights
        tp_size=tensor_model_parallel_size,
        dtype='bfloat16',
        mem_fraction_static=0.1,
        nccl_port=12345,
        tp_rank=rank,
        gpu_id=rank,
        parallel_process_groups=ParallelProcessGroups.from_devices_meshes(
            device_mesh_device=inference_device_mesh_device,
            device_mesh_cpu=inference_device_mesh_cpu,
            dim_tp="tp",
            dim_pp="pp",
        ),
    )

    # most naive way
    t = time.time()
    state_dict_full = {k: v.full_tensor() for k, v in state_dict.items()}
    print(f'gather full tensor: {time.time() - t:.2f}')
    llm.update_weights_from_tensor([(k, v) for k, v in state_dict_full.items()])
    print(f'gather + update weights: {time.time() - t:.2f}')

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    idx_list = []
    batch_size = input_ids.shape[0]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # from verl.workers.rollout.vllm_rollout.vllm_rollout import _pre_process_inputs
    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))
    print('start generation')
    # outputs = llm.generate(prompt_token_ids=idx_list, sampling_params=sampling_params, use_tqdm=False)
    outputs = llm.generate(input_ids=idx_list, sampling_params=sampling_params)

    # vllm_output = outputs[0].cuda()
    if torch.distributed.get_rank() == 0:
        print(f'hf response: {tokenizer.batch_decode(response)}')
        # print(f'vllm response: {tokenizer.batch_decode(vllm_output)}')
        print(f'vllm response: {[o["text"] for o in outputs]}')

    llm.shutdown()


# COPIED FROM verl
def initialize_global_process_group(timeout_second=36000):
    import torch.distributed
    from datetime import timedelta

    # NOTE MODIFIED should provide backend=None to have nccl+gloo
    # torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


# COPIED FROM verl
def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


# COPIED FROM verl
# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if __name__ == "__main__":
    main()
