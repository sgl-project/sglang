import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock

import grpc
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from sglang.srt.entrypoints.grpc_server import SGLangSchedulerServicer
from sglang.srt.grpc import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from sglang.srt.server_args import ServerArgs


class TestGetTokenizer(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # valid tiny model from HF
        cls.model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
        cls.model_path = snapshot_download(cls.model_id)

    async def asyncSetUp(self):
        self.server = grpc.aio.server()
        self.mock_manager = MagicMock()

        # Minimal server args
        self.server_args = ServerArgs(
            model_path=self.model_path,
            host="127.0.0.1",
            port=0,  # Unused for internal logic but good for completeness
            tokenizer_path=self.model_path,
        )

        self.servicer = SGLangSchedulerServicer(
            request_manager=self.mock_manager,
            server_args=self.server_args,
            model_info={},  # pass in Minimal info if needed, or empty dict
            scheduler_info={},
        )

        sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server(
            self.servicer, self.server
        )

        # Bind to a random port
        port = self.server.add_insecure_port("127.0.0.1:0")
        await self.server.start()
        self.address = f"127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.server.stop(0)

    async def test_get_tokenizer(self):
        print(f"Connecting to test server at {self.address}")
        async with grpc.aio.insecure_channel(self.address) as channel:
            stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)

            request = sglang_scheduler_pb2.GetTokenizerRequest()

            # Stream response
            file_chunks = []
            metadata_received = False

            async for response in stub.GetTokenizer(request):
                if response.HasField("metadata"):
                    print(f"Received metadata: {response.metadata.model_identifier}")
                    metadata_received = True
                elif response.HasField("file_chunk"):
                    file_chunks.append(response.file_chunk.data)

            self.assertTrue(metadata_received, "Should have received metadata")
            self.assertTrue(len(file_chunks) > 0, "Should have received file chunks")

            # Reassemble zip
            zip_content = b"".join(file_chunks)
            print(f"Total zip size: {len(zip_content)} bytes")

            # verify zip content
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "tokenizer.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_content)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Check that essential files exist
                self.assertTrue(
                    os.path.exists(os.path.join(temp_dir, "tokenizer.json"))
                )
                self.assertTrue(
                    os.path.exists(os.path.join(temp_dir, "tokenizer_config.json"))
                )

                # Try loading with AutoTokenizer
                print("Loading tokenizer with transformers...")
                tokenizer = AutoTokenizer.from_pretrained(temp_dir)

                test_text = "Hello, world!"
                tokens = tokenizer.encode(test_text)
                decoded = tokenizer.decode(tokens)

                print(f"Original: {test_text}")
                print(f"Tokens: {tokens}")
                print(f"Decoded: {decoded}")

                # Basic sanity check (decoded text might not match exactly for random models but shouldn't error)
                self.assertIsInstance(tokens, list)
                self.assertGreater(len(tokens), 0)


if __name__ == "__main__":
    unittest.main()
