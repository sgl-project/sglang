import datetime
import os
import sys

from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.server.engine_fragment import EngineFragment
from torch.distributed.device_mesh import init_device_mesh


def run():
    """
    Example command:
    ```
    torchrun --nproc_per_node=4 offline_batch_inference_torchrun.py
    ```
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def _log(text):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] [rank={rank}] {text}")

    _log(
        f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}'
    )

    tp_size = 4
    dp_size = 2
    assert world_size == tp_size * dp_size

    device_mesh_kwargs = dict(mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=['tp', 'dp', 'pp'])
    device_mesh_device = init_device_mesh('cuda', **device_mesh_kwargs)
    device_mesh_cpu = init_device_mesh('cpu', **device_mesh_kwargs)
    _log(f"{device_mesh_device=} {device_mesh_cpu=}")

    tp_rank = device_mesh_device.get_local_rank('tp')
    dp_rank = device_mesh_device.get_local_rank('dp')
    _log(f"{tp_rank=} {tp_size=} ; {dp_rank=} {dp_size=}")

    model_name, mem_fraction_static = "meta-llama/Llama-3.2-1B-Instruct", 0.1
    # model_name, mem_fraction_static = "meta-llama/Llama-3.1-70B-Instruct", 0.9 # test large models

    for k in [
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_DEBUG",
        "OMP_NUM_THREADS",
        "RANK",
        "ROLE_NAME",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "WORLD_SIZE",
    ]:
        del os.environ[k]

    fragment = EngineFragment(
        model_path=model_name,
        mem_fraction_static=mem_fraction_static,
        tp_size=tp_size,
        tp_rank=tp_rank,
        nccl_port=23456,
        gpu_id=local_rank,
        parallel_process_groups=ParallelProcessGroups.from_devices_meshes(
            device_mesh_device=device_mesh_device,
            device_mesh_cpu=device_mesh_cpu,
            dim_tp='tp',
            dim_pp='pp',
        ),
    )
    _log(f"{fragment=}")

    prompt_all = [
        ["1+1=2, 1+2=3, 1+3=4, 1+4=", "9-1=8, 8-1=7, 7-1="],
        ["2*1=2, 2*2=4, 2*3=", "8/2=4, 6/2="],
    ]
    prompt = prompt_all[dp_rank]

    output = fragment.generate(
        prompt=prompt,
        sampling_params=dict(max_new_tokens=16, temperature=0.0),
    )
    _log(f"{prompt=} {output=}")

    fragment.shutdown()
    _log(f"End script")


if __name__ == "__main__":
    run()
