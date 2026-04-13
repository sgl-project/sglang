# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiffusionServer pool-based pipeline orchestrator."""

import time
import unittest

import zmq

from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
    DiffusionServer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocatedMsg,
    TransferPushedMsg,
    TransferRegisterMsg,
    TransferStagedMsg,
    decode_transfer_msg,
    encode_transfer_msg,
)


class _MockReq:
    """Minimal mock request for testing."""

    def __init__(self, request_id: str = "test-001"):
        self.request_id = request_id
        self.is_warmup = False


from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
)


class TestDiffusionServerInit(unittest.TestCase):
    """Test DiffusionServer initialization."""

    def test_basic_init(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19900",
            encoder_work_endpoints=["tcp://127.0.0.1:19901"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19902"],
            decoder_work_endpoints=["tcp://127.0.0.1:19903"],
            encoder_result_endpoint="tcp://127.0.0.1:19904",
            denoiser_result_endpoint="tcp://127.0.0.1:19905",
            decoder_result_endpoint="tcp://127.0.0.1:19906",
        )
        self.assertEqual(server._num_encoders, 1)
        self.assertEqual(server._num_denoisers, 1)
        self.assertEqual(server._num_decoders, 1)

    def test_get_stats(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19910",
            encoder_work_endpoints=["tcp://127.0.0.1:19911"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19912"],
            decoder_work_endpoints=["tcp://127.0.0.1:19913"],
            encoder_result_endpoint="tcp://127.0.0.1:19914",
            denoiser_result_endpoint="tcp://127.0.0.1:19915",
            decoder_result_endpoint="tcp://127.0.0.1:19916",
        )
        stats = server.get_stats()
        self.assertEqual(stats["role"], "diffusion_server")
        self.assertEqual(stats["num_encoders"], 1)
        self.assertEqual(stats["num_denoisers"], 1)
        self.assertEqual(stats["num_decoders"], 1)
        self.assertEqual(stats["pending_requests"], 0)
        # Capacity-aware stats
        self.assertEqual(stats["encoder_free_slots"], [4])
        self.assertEqual(stats["denoiser_free_slots"], [2])
        self.assertEqual(stats["decoder_free_slots"], [4])
        self.assertEqual(stats["encoder_tta_depth"], 0)
        self.assertEqual(stats["denoiser_tta_depth"], 0)
        self.assertEqual(stats["decoder_tta_depth"], 0)

    def test_custom_capacity(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19920",
            encoder_work_endpoints=["tcp://127.0.0.1:19921", "tcp://127.0.0.1:19922"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19923"],
            decoder_work_endpoints=["tcp://127.0.0.1:19924"],
            encoder_result_endpoint="tcp://127.0.0.1:19925",
            denoiser_result_endpoint="tcp://127.0.0.1:19926",
            decoder_result_endpoint="tcp://127.0.0.1:19927",
            encoder_capacity=8,
            denoiser_capacity=3,
            decoder_capacity=6,
        )
        self.assertEqual(server._encoder_free_slots, [8, 8])
        self.assertEqual(server._denoiser_free_slots, [3])
        self.assertEqual(server._decoder_free_slots, [6])


class TestDiffusionServerTransferInit(unittest.TestCase):
    """Test DiffusionServer transfer mode initialization."""

    def test_transfer_mode_init(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19950",
            encoder_work_endpoints=["tcp://127.0.0.1:19951"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19952"],
            decoder_work_endpoints=["tcp://127.0.0.1:19953"],
            encoder_result_endpoint="tcp://127.0.0.1:19954",
            denoiser_result_endpoint="tcp://127.0.0.1:19955",
            decoder_result_endpoint="tcp://127.0.0.1:19956",
        )
        self.assertTrue(server._transfer_mode)
        self.assertEqual(len(server._transfer_state), 0)
        self.assertEqual(len(server._encoder_peers), 0)

    def test_transfer_stats(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19960",
            encoder_work_endpoints=["tcp://127.0.0.1:19961"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19962"],
            decoder_work_endpoints=["tcp://127.0.0.1:19963"],
            encoder_result_endpoint="tcp://127.0.0.1:19964",
            denoiser_result_endpoint="tcp://127.0.0.1:19965",
            decoder_result_endpoint="tcp://127.0.0.1:19966",
        )
        stats = server.get_stats()
        self.assertTrue(stats["transfer_mode"])
        self.assertEqual(stats["transfer_active_transfers"], 0)
        self.assertEqual(stats["encoder_peers"], 0)


class TestDiffusionServerTransferProtocol(unittest.TestCase):
    """Test transfer protocol message handling in DiffusionServer."""

    def test_transfer_register(self):
        """Test instance registration with DS."""
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19970",
            encoder_work_endpoints=["tcp://127.0.0.1:19971"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19972"],
            decoder_work_endpoints=["tcp://127.0.0.1:19973"],
            encoder_result_endpoint="tcp://127.0.0.1:19974",
            denoiser_result_endpoint="tcp://127.0.0.1:19975",
            decoder_result_endpoint="tcp://127.0.0.1:19976",
        )

        # Register an encoder
        reg_msg = TransferRegisterMsg(
            role="encoder",
            session_id="enc-session-0",
            pool_ptr=0x7F000000,
            pool_size=16 * 1024 * 1024,
        )
        frames = encode_transfer_msg(reg_msg)
        server._handle_transfer_result(frames, "encoder")

        self.assertIn(0, server._encoder_peers)
        self.assertEqual(server._encoder_peers[0]["session_id"], "enc-session-0")
        self.assertEqual(server._encoder_peers[0]["pool_ptr"], 0x7F000000)

    def test_transfer_staged_and_alloc(self):
        """Test encoder staged → DS selects denoiser → sends alloc."""
        ctx = zmq.Context()
        # We need live sockets to capture the alloc message DS sends to denoiser
        denoiser_work_ep = "tcp://127.0.0.1:19980"
        denoiser_work_pull = ctx.socket(zmq.PULL)
        denoiser_work_pull.bind(denoiser_work_ep)

        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19981",
            encoder_work_endpoints=["tcp://127.0.0.1:19982"],
            denoiser_work_endpoints=[denoiser_work_ep],
            decoder_work_endpoints=["tcp://127.0.0.1:19983"],
            encoder_result_endpoint="tcp://127.0.0.1:19984",
            denoiser_result_endpoint="tcp://127.0.0.1:19985",
            decoder_result_endpoint="tcp://127.0.0.1:19986",
        )
        server.start()
        time.sleep(0.3)  # Let sockets connect

        try:
            # Submit a request
            server._tracker.submit("r1")
            server._tracker.transition(
                "r1", RequestState.ENCODER_RUNNING, encoder_instance=0
            )

            # Simulate encoder sending transfer_staged
            staged_msg = TransferStagedMsg(
                request_id="r1",
                data_size=4096,
                manifest={"latents": [{"offset": 0, "shape": [4], "dtype": "float32"}]},
                session_id="enc-0",
                pool_ptr=0x1000,
                slot_offset=0,
            )
            frames = encode_transfer_msg(staged_msg)
            server._handle_transfer_result(frames, "encoder")

            # DS should have sent transfer_alloc to denoiser
            alloc_frames = denoiser_work_pull.recv_multipart(flags=0)
            alloc_msg = decode_transfer_msg(alloc_frames)
            self.assertEqual(alloc_msg["msg_type"], "transfer_alloc")
            self.assertEqual(alloc_msg["request_id"], "r1")
            self.assertEqual(alloc_msg["data_size"], 4096)

            # Verify transfer state
            self.assertIn("r1", server._transfer_state)
            self.assertEqual(server._transfer_state["r1"].sender_session_id, "enc-0")
        finally:
            server.stop()
            denoiser_work_pull.close()
            ctx.destroy(linger=0)

    def test_transfer_full_e2e_handshake(self):
        """Test full transfer handshake: staged → alloc → allocated → push → pushed → ready."""
        ctx = zmq.Context()
        enc_work_ep = "tcp://127.0.0.1:19990"
        den_work_ep = "tcp://127.0.0.1:19991"

        enc_work_pull = ctx.socket(zmq.PULL)
        enc_work_pull.bind(enc_work_ep)
        den_work_pull = ctx.socket(zmq.PULL)
        den_work_pull.bind(den_work_ep)

        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19992",
            encoder_work_endpoints=[enc_work_ep],
            denoiser_work_endpoints=[den_work_ep],
            decoder_work_endpoints=["tcp://127.0.0.1:19993"],
            encoder_result_endpoint="tcp://127.0.0.1:19994",
            denoiser_result_endpoint="tcp://127.0.0.1:19995",
            decoder_result_endpoint="tcp://127.0.0.1:19996",
        )
        server.start()
        time.sleep(0.3)

        try:
            # Setup: submit request with encoder running
            server._tracker.submit("r1")
            server._tracker.transition(
                "r1", RequestState.ENCODER_RUNNING, encoder_instance=0
            )

            # Step 1: Encoder staged
            staged = TransferStagedMsg(
                request_id="r1",
                data_size=2048,
                manifest={"t": [{"offset": 0, "shape": [512], "dtype": "float32"}]},
                session_id="enc-sess",
                pool_ptr=0x1000,
                slot_offset=0,
            )
            server._handle_transfer_result(encode_transfer_msg(staged), "encoder")

            # Step 2: Denoiser receives alloc
            alloc_frames = den_work_pull.recv_multipart()
            alloc = decode_transfer_msg(alloc_frames)
            self.assertEqual(alloc["msg_type"], "transfer_alloc")

            # Step 3: Denoiser sends allocated
            allocated = TransferAllocatedMsg(
                request_id="r1",
                session_id="den-sess",
                pool_ptr=0x2000,
                slot_offset=0,
                slot_size=2048,
            )
            server._handle_transfer_result(encode_transfer_msg(allocated), "denoiser")

            # Step 4: Encoder receives push command
            push_frames = enc_work_pull.recv_multipart()
            push = decode_transfer_msg(push_frames)
            self.assertEqual(push["msg_type"], "transfer_push")
            self.assertEqual(push["dest_session_id"], "den-sess")
            self.assertEqual(push["dest_addr"], 0x2000)  # pool_ptr + slot_offset
            self.assertEqual(push["transfer_size"], 2048)

            # Step 5: Encoder sends pushed (RDMA done)
            pushed = TransferPushedMsg(request_id="r1")
            server._handle_transfer_result(encode_transfer_msg(pushed), "encoder")

            # Step 6: Denoiser receives ready
            ready_frames = den_work_pull.recv_multipart()
            ready = decode_transfer_msg(ready_frames)
            self.assertEqual(ready["msg_type"], "transfer_ready")
            self.assertEqual(ready["request_id"], "r1")
            self.assertIn("t", ready["manifest"])
        finally:
            server.stop()
            enc_work_pull.close()
            den_work_pull.close()
            ctx.destroy(linger=0)


if __name__ == "__main__":
    unittest.main()
