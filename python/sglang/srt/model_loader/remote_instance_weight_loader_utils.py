# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import socket
import time
from typing import List

import requests

from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


def local_instance_id() -> str:
    """This client's identity token inside the send-weights group name.

    It is only ever interpolated into a group name -- never dialed -- so all that
    matters is that every call site agrees. Both the request that asks the seed to
    create the group and the client's own group construction must derive it here:
    init_custom_process_group wraps the rendezvous store in
    PrefixStore(group_name, store), so two spellings of this token put the seed
    and the client in disjoint key namespaces, and the NCCL unique-ID handshake
    then blocks forever with no timeout and no error.
    """
    return NetworkAddress.resolve_host(socket.gethostname())


def _raise_for_seed_refusal(response, what: str, seed_url: str) -> None:
    """Surface a seed-side refusal instead of letting the client wait it out.

    Both trigger calls run fire-and-forget on a worker thread -- they have to, the
    seed does not answer the group-creation request until the client has joined the
    barrier. So nothing downstream ever looks at the response, and a seed that
    refuses (a group port already taken by an outbound connection, a group name it
    does not know, a model that does not match) leaves the client sitting in the
    group build until the process-group timeout fires tens of minutes later, with
    no log line naming a cause anywhere. Log the seed's own message.
    """
    if response.status_code == 200:
        return
    detail = response.text
    try:
        body = response.json()
        detail = body.get("message") or body.get("detail") or detail
    except ValueError:
        pass
    raise RuntimeError(
        f"Seed instance {seed_url} refused {what} "
        f"(HTTP {response.status_code}): {detail}"
    )


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = NetworkAddress(
        remote_instance_weight_loader_seed_instance_ip,
        remote_instance_weight_loader_seed_instance_service_port,
    ).to_url()
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        response = requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
        _raise_for_seed_refusal(
            response, "send-weights group creation", seed_instance_service_url
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = NetworkAddress(
        remote_instance_weight_loader_seed_instance_ip,
        remote_instance_weight_loader_seed_instance_service_port,
    ).to_url()
    try:
        response = requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
        _raise_for_seed_refusal(
            response, "the weight transfer", seed_instance_service_url
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None


def register_memory_region(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    for name, weight in model.named_parameters():
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    # Register merged memory blocks that hold weights.
    for weight_block in weight_blocks_for_reg_mr:
        address, size = weight_block
        ret = transfer_engine.register_memory(address, size)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"
            )

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict
