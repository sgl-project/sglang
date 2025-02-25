import datetime
import os
import sys

from sglang.srt.entrypoints.verl_engine import VerlEngine
from torch.distributed.device_mesh import init_device_mesh


def run():
    """
    Example command:
    ```
    torchrun --nproc_per_node=8 offline_batch_inference_torchrun.py
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
    # TODO
    # TODO temp
    # TODO
    dp_size = 1
    assert world_size == tp_size * dp_size

    device_mesh_kwargs = dict(
        mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=["tp", "dp", "pp"]
    )
    device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
    _log(f"{device_mesh_cpu=}")

    tp_rank = device_mesh_cpu.get_local_rank("tp")
    dp_rank = device_mesh_cpu.get_local_rank("dp")
    _log(f"{tp_rank=} {tp_size=} ; {dp_rank=} {dp_size=}")

    model_name, mem_fraction_static = "meta-llama/Llama-3.2-1B-Instruct", 0.1
    # model_name, mem_fraction_static = "meta-llama/Llama-3.1-70B-Instruct", 0.9 # test large models

    # TODO remove this
    for k in [
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
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
        if k in os.environ:
            del os.environ[k]

    fragment = VerlEngine(
        model_path=model_name,
        mem_fraction_static=mem_fraction_static,
        first_rank_in_node=tp_rank == 0,
        device_mesh_cpu=device_mesh_cpu['tp'],
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
