import datetime
import os
import sys

from sglang import Engine
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerFragmentArgs
from transformers import AutoTokenizer


# TODO big refactor sglang system after poc
def run():
    # build distributed world
    local_rank, rank, world_size = initialize_global_process_group()

    def _log(text):
        t = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'[{t}] [rank={rank}] {text}')

    _log(f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}')

    # hack
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    # torch.cuda.set_device(rank)
    # _log(f"now {os.environ['CUDA_VISIBLE_DEVICES']=}")

    dp_size, tp_size = 1, 4
    assert world_size == dp_size * tp_size, f'{world_size=}'

    dp_rank, tp_rank = divmod(rank, tp_size)
    _log(f'{dp_rank=} {tp_rank=}')

    # model_name, mem_fraction_static = "meta-llama/Llama-3.2-1B-Instruct", 0.1
    model_name, mem_fraction_static = "meta-llama/Llama-3.1-70B-Instruct", 0.9

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # `sync_model_weights` not in this PR
    # # build device mesh for training engine.
    # device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    # fsdp_model = FSDP(actor_model, ..., device_mesh = device_mesh)
    # FSDP.set_state_dict_type(fsdp_model,
    #                          state_dict_type=StateDictType.SHARDED_STATE_DICT,
    #                          state_dict_config=ShardedStateDictConfig())
    # # get sharded model state dict
    # state_dict = fsdp_model.state_dict()
    # # sync weights between actor and rollout, support several format: DTensor and Megatron (sharded)
    # inference_engine.sync_model_weights(actor_weights=state_dict, load_format='dtensor')
    #
    # name = sys.argv[1]

    # [Optional] build device mesh for inference engine
    # gen_device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, tp_size), mesh_dim_names=['dp', 'tp'])
    # build inference engine
    inference_engine = Engine(
        model_path=model_name,
        mem_fraction_static=mem_fraction_static,
        tp_size=tp_size,
        dp_size=dp_size,
        fragment=ServerFragmentArgs(
            tp_rank=tp_rank,
            nccl_port=12345,
        ),
    )
    _log(f'{inference_engine=}')

    # moved to above
    # # [Optional] update parallel state in SGLang for 3D-HybridEngine
    # inference_engine.update_parallel_state(TP=device_mesh["tp"])

    input_text = "Today is a sunny day and I like"
    # input_ids = hf_tokenizer(input_text)['input_ids']
    # _log(f'{input_ids=}')

    # generate sequence, it would be better if the output is a list of Tensor not list of list[str]
    output = inference_engine.generate(
        prompt=input_text,
        sampling_params=SamplingParams(max_new_tokens=16),
    )
    print(f'{output=}')

    _log('exit')

    # already done in old PR, waiting for merging
    # # offload kvcache after generation
    # inference_engine.free_kvcache()  # inference_engine.init_kvcache()
    # # offload model
    # inference_engine.offload_model_weights()  # inference_engine.load_model_weights(), we can simply re-init them


# NOTE COPIED FROM verl
def initialize_global_process_group(timeout_second=36000):
    import torch.distributed
    from datetime import timedelta
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        print(f'call torch.cuda.set_device({local_rank=})')
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


if __name__ == '__main__':
    run()
