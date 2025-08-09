import json
import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.server_args import prepare_server_args


def save_expert_location(expert_location, rank, output_dir):
    """Saves the expert location metadata to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"expert_metadata_rank_{rank}.json")
    data_to_save = {
        "physical_to_logical_map": expert_location.physical_to_logical_map.cpu()
        .numpy()
        .tolist(),
        "logical_to_all_physical_map": expert_location.logical_to_all_physical_map.cpu()
        .numpy()
        .tolist(),
        "logical_to_all_physical_map_num_valid": expert_location.logical_to_all_physical_map_num_valid.cpu()
        .numpy()
        .tolist(),
    }
    if expert_location.logical_to_rank_dispatch_physical_map is not None:
        data_to_save["logical_to_rank_dispatch_physical_map"] = (
            expert_location.logical_to_rank_dispatch_physical_map.cpu().numpy().tolist()
        )

    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"Saved expert location metadata for rank {rank} to {file_path}")


def worker(rank, world_size, server_args, fault_tolerant):
    """The worker function for each process."""
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"  # Use a free port

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model_config = ModelConfig.from_server_args(server_args)

    if fault_tolerant:
        print("Generating fault-tolerant expert distribution...")

        @dataclass
        class CustomExpertLocationMetadata:
            physical_to_logical_map: torch.Tensor
            logical_to_all_physical_map: torch.Tensor
            logical_to_all_physical_map_num_valid: torch.Tensor
            logical_to_rank_dispatch_physical_map: torch.Tensor
            num_physical_experts: int

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )
        num_logical_experts = model_config_for_expert_location.num_logical_experts
        # Assuming all layers have experts for distribution purposes.
        # The mapping will be the same for all layers with MoE.
        num_layers_with_experts = model_config_for_expert_location.num_layers
        ep_size = world_size

        # Create a deterministic mapping of logical experts to ranks.
        # Each logical expert is mapped to 2 ranks for fault tolerance.
        physical_slots_on_ranks = [[] for _ in range(ep_size)]
        for logical_idx in range(num_logical_experts):
            rank1 = logical_idx % ep_size
            rank2 = (logical_idx + 1) % ep_size

            if rank1 == rank2:  # Only happens if ep_size is 1
                physical_slots_on_ranks[rank1].append(logical_idx)
            else:
                physical_slots_on_ranks[rank1].append(logical_idx)
                physical_slots_on_ranks[rank2].append(logical_idx)

        # Calculate global physical indices
        num_physical_experts_on_rank = [len(x) for x in physical_slots_on_ranks]
        rank_offsets = [0] * ep_size
        for i in range(1, ep_size):
            rank_offsets[i] = rank_offsets[i - 1] + num_physical_experts_on_rank[i - 1]

        num_total_physical_experts = sum(num_physical_experts_on_rank)
        max_replicas = 2 if ep_size > 1 else 1

        # Create global maps (same for all ranks)
        logical_to_all_physical_map = torch.full(
            (num_layers_with_experts, num_logical_experts, max_replicas),
            -1,
            dtype=torch.int32,
        )
        logical_to_all_physical_map_num_valid = torch.zeros(
            (num_layers_with_experts, num_logical_experts), dtype=torch.int32
        )

        for logical_idx in range(num_logical_experts):
            rank1 = logical_idx % ep_size
            rank2 = (logical_idx + 1) % ep_size

            # Get global physical index for replica on rank1
            phys_slot_idx_on_rank1 = physical_slots_on_ranks[rank1].index(logical_idx)
            global_phys_idx1 = rank_offsets[rank1] + phys_slot_idx_on_rank1

            logical_to_all_physical_map[:, logical_idx, 0] = global_phys_idx1
            logical_to_all_physical_map_num_valid[:, logical_idx] = 1

            if rank1 != rank2:
                # Get global physical index for replica on rank2
                phys_slot_idx_on_rank2 = physical_slots_on_ranks[rank2].index(
                    logical_idx
                )
                global_phys_idx2 = rank_offsets[rank2] + phys_slot_idx_on_rank2
                logical_to_all_physical_map[:, logical_idx, 1] = global_phys_idx2
                logical_to_all_physical_map_num_valid[:, logical_idx] = 2

        # Create maps for the current rank
        my_physical_slots = physical_slots_on_ranks[rank]
        num_my_physical_experts = len(my_physical_slots)

        physical_to_logical_map = torch.full(
            (num_layers_with_experts, num_my_physical_experts), -1, dtype=torch.int32
        )
        if num_my_physical_experts > 0:
            physical_to_logical_map[:, :] = torch.tensor(
                my_physical_slots, dtype=torch.int32
            ).unsqueeze(0)

        logical_to_rank_dispatch_physical_map = torch.full(
            (num_layers_with_experts, num_logical_experts), -1, dtype=torch.int32
        )
        for phys_idx_on_rank, logical_idx in enumerate(my_physical_slots):
            global_phys_idx = rank_offsets[rank] + phys_idx_on_rank
            logical_to_rank_dispatch_physical_map[:, logical_idx] = global_phys_idx

        expert_location = CustomExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=logical_to_rank_dispatch_physical_map,
            num_physical_experts=num_total_physical_experts,
        )
    else:
        print("Generating trivial expert distribution...")
        expert_location = ExpertLocationMetadata.init_trivial(server_args, model_config)

    # Export the object to disk
    output_dir = os.environ.get("EPLB_LOCATION_DIR", "/tmp/expert_location_metadata")
    save_expert_location(expert_location, rank, output_dir)

    # Clean up the process group
    dist.destroy_process_group()


def main():
    # A bit of a hack to get the fault_tolerant flag without changing server_args
    fault_tolerant = "--fault-tolerant" in sys.argv
    if fault_tolerant:
        # remove the flag so prepare_server_args doesn't see it
        sys.argv.remove("--fault-tolerant")

    server_args = prepare_server_args(sys.argv[1:])
    world_size = server_args.ep_size

    mp.spawn(
        worker,
        args=(world_size, server_args, fault_tolerant),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
