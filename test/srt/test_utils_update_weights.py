import asyncio
import os
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.weight_sync.utils import update_weights
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class AsyncEngine(Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def update_weights_from_tensor(self, update_weights_request):
        return await self.tokenizer_manager.update_weights_from_tensor(
            update_weights_request, None
        )


def is_distributed_available():
    """Check if distributed training environment is available"""
    required_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    return all(var in os.environ for var in required_vars)


def setup_single_process_distributed():
    """Setup distributed environment for single process testing"""
    if not is_distributed_available():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        os.environ["LOCAL_RANK"] = "0"


class TestUtilsUpdateWeights(unittest.TestCase):
    """Test class for utils.update_weights function"""

    @classmethod
    def setUpClass(cls):
        """Setup distributed environment and test fixtures for the entire test class"""
        cls.setup_distributed()
        cls.setup_test_engine()
        cls.setup_test_model()
        cls.setup_device_mesh()

    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests"""
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()

        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()

    @classmethod
    def setup_distributed(cls):
        """Setup distributed environment for testing"""
        setup_single_process_distributed()

        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
            except Exception as e:
                raise unittest.SkipTest(
                    f"Could not initialize distributed backend: {e}"
                )

        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(cls.rank % torch.cuda.device_count())

        # Set up environment variables
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    @classmethod
    def setup_test_engine(cls):
        """Setup test engine"""
        if cls.rank == 0:
            cls.engine = AsyncEngine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                dtype="bfloat16",
                mem_fraction_static=0.3,
                enable_memory_saver=True,
                tp_size=cls.world_size,
                disable_cuda_graph=False,
            )
        else:
            cls.engine = None

    @classmethod
    def setup_test_model(cls):
        """Load test model"""
        try:
            cls.model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )
        except Exception as e:
            raise unittest.SkipTest(f"Could not load test model: {e}")

    @classmethod
    def setup_device_mesh(cls):
        """Create device mesh for testing"""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available for device mesh")

        cls.device_mesh_key = "tp"
        cls.mesh = init_device_mesh(
            "cuda", (cls.world_size,), mesh_dim_names=(cls.device_mesh_key,)
        )

    def create_test_params_batch(self, model, num_params=64):
        """Create a batch of test parameters from the model"""
        param_names = []
        test_tensors = []

        # Get first few parameters from the model for testing
        for i, (name, tensor) in enumerate(model.named_parameters()):
            if i >= num_params:
                break
            param_names.append(name)
            # Create test tensor with known values, matching original shape and dtype
            test_tensor = torch.full_like(tensor, 1.5, dtype=tensor.dtype).cuda()
            test_tensors.append(test_tensor)

        return list(zip(param_names, test_tensors))

    def test_utils_update_weights(self):
        """Test basic functionality of utils.update_weights"""

        async def async_test():
            # Create test parameters batch
            params_batch = self.create_test_params_batch(self.model, num_params=2)

            # Test the utils.update_weights function
            result = await update_weights(
                engine=self.engine,
                params_batch=params_batch,
                device_mesh_key=self.device_mesh_key,
                device_mesh=self.mesh,
                load_format=None,
            )

            self.assertIn("Success", result)

        # Run the async test
        asyncio.run(async_test())


if __name__ == "__main__":
    unittest.main()
