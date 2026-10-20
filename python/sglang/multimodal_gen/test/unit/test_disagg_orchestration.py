# SPDX-License-Identifier: Apache-2.0
"""Exercise the real head's transfer protocol with lightweight worker peers."""

import base64
import pickle
import time

import pytest
import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.orchestrator import DiffusionServer
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import send_tensors
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    decode_transfer_msg,
    encode_transfer_msg,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
def test_pipeline_transfer_and_slot_reuse(tmp_path, split, preallocated):
    """Every hop reaches its owner; repeated requests and errors return capacity."""
    context = zmq.Context()
    sockets = []

    def socket(kind):
        sock = context.socket(kind)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.LINGER, 0)
        sockets.append(sock)
        return sock

    def bind(sock):
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        return f"tcp://127.0.0.1:{port}"

    roles = (
        [RoleType.PROMPT_ENCODER, RoleType.IMAGE_ENCODER]
        if split
        else [RoleType.ENCODER]
    )
    roles += [RoleType.DENOISER, RoleType.DECODER]
    work = {role: socket(zmq.PULL) for role in roles}
    endpoints = {role: bind(sock) for role, sock in work.items()}
    holders = [socket(zmq.PULL) for _ in range(6)]
    frontend, *results = [bind(sock) for sock in holders]
    for holder in holders:
        holder.close()
    server = DiffusionServer(
        frontend_endpoint=frontend,
        encoder_work_endpoints=[] if split else [endpoints[RoleType.ENCODER]],
        denoiser_work_endpoints=[endpoints[RoleType.DENOISER]],
        decoder_work_endpoints=[endpoints[RoleType.DECODER]],
        encoder_result_endpoint=results[0],
        denoiser_result_endpoint=results[1],
        decoder_result_endpoint=results[2],
        prompt_encoder_work_endpoints=[endpoints[RoleType.PROMPT_ENCODER]]
        if split
        else None,
        image_encoder_work_endpoints=[endpoints[RoleType.IMAGE_ENCODER]]
        if split
        else None,
        prompt_encoder_result_endpoint=results[3],
        image_encoder_result_endpoint=results[4],
    )
    try:
        server.start()
        assert server.wait_ready(5)
        result = {role: socket(zmq.PUSH) for role in roles}
        for role, sock in result.items():
            sock.connect(server._pools[role].result_endpoint)
            sock.send_multipart(
                encode_transfer_msg(
                    dict(
                        msg_type="transfer_register",
                        role=role.value,
                        work_endpoint=endpoints[role],
                        session_id=role.value,
                        pool_ptr=1000,
                        preallocated_slots=[
                            dict(offset=0, size=64, slot_id=0, addr=1000)
                        ]
                        if preallocated
                        else [],
                    )
                )
            )
        deadline = time.monotonic() + 5
        while not all(pool.peers for pool in server._pools.values()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        client = socket(zmq.REQ)
        client.connect(frontend)
        reference = tmp_path / "head-only.png"
        reference.write_bytes(b"head-local image upload")
        for iteration in range(3 if split else 2):
            request_id = str(iteration)
            image_path = [str(reference)] if iteration else None
            client.send_pyobj(
                [Req(request_id=request_id, prompt="test", image_path=image_path)]
            )
            req = pickle.loads(work[roles[0]].recv_multipart()[-1])[0]
            if split and image_path:
                assert (
                    base64.b64decode(req.image_path[0].split(",", 1)[1])
                    == reference.read_bytes()
                )
            fail_image = split and iteration == 2
            active_roles = [
                role for role in roles if image_path or role != RoleType.IMAGE_ENCODER
            ]
            for sender, receiver in zip(active_roles, active_roles[1:]):
                if sender == RoleType.IMAGE_ENCODER and fail_image:
                    result[sender].send_multipart(
                        encode_transfer_msg(
                            dict(
                                msg_type="transfer_done",
                                request_id=request_id,
                                error="invalid reference",
                            )
                        )
                    )
                    break
                result[sender].send_multipart(
                    encode_transfer_msg(
                        dict(
                            msg_type="transfer_done"
                            if sender == RoleType.DENOISER
                            else "transfer_staged",
                            staged_for_decoder=sender == RoleType.DENOISER,
                            request_id=request_id,
                            data_size=64,
                            manifest={"payload": sender.value},
                            scalar_fields={"seed": 42, "image_path": image_path},
                        )
                    )
                )
                if sender.is_input_encoder:
                    # A duplicate must not enqueue a second transfer or kill the head.
                    result[sender].send_multipart(
                        encode_transfer_msg(
                            dict(
                                msg_type="transfer_staged",
                                request_id=request_id,
                            )
                        )
                    )
                if not preallocated:
                    alloc = decode_transfer_msg(work[receiver].recv_multipart())
                    assert alloc["source_role"] == sender.value
                    result[receiver].send_multipart(
                        encode_transfer_msg(
                            dict(
                                msg_type="transfer_allocated",
                                request_id=request_id,
                                session_id=receiver.value,
                                pool_ptr=1000,
                                slot_offset=0,
                            )
                        )
                    )
                push = decode_transfer_msg(work[sender].recv_multipart())
                assert push["dest_session_id"] == receiver.value
                result[sender].send_multipart(
                    encode_transfer_msg(
                        dict(
                            msg_type="transfer_pushed",
                            request_id=request_id,
                        )
                    )
                )
                result[sender].send_multipart(
                    encode_transfer_msg(
                        dict(
                            msg_type="transfer_pushed",
                            request_id=request_id,
                        )
                    )
                )
                ready = decode_transfer_msg(work[receiver].recv_multipart())
                assert ready["manifest"] == {"payload": sender.value}
                assert ready["scalar_fields"]["seed"] == 42
            if not fail_image:
                send_tensors(
                    result[RoleType.DECODER],
                    {"output": torch.ones(1, 3, 2, 2)},
                    {"request_id": request_id},
                )
            output = client.recv_pyobj()
            if fail_image:
                assert "invalid reference" in output.error
            else:
                torch.testing.assert_close(output.output, torch.ones(1, 3, 2, 2))
            assert not server._assignments
            expected = [4, 2, 4, 4] if split else [4, 2, 4]
            assert [pool.free_slots[0] for pool in server._pools.values()] == expected
    finally:
        server.stop()
        for sock in sockets:
            sock.close(linger=0)
        context.destroy(linger=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
