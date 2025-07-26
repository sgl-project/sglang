import asyncio
import os

import pytest
import torch
import torch.distributed as dist
from loguru import logger
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


class TestUtilsUpdateWeights:
    """Test class for utils.update_weights function"""

    @pytest.fixture(scope="class")
    def setup_distributed(self):
        """Setup distributed environment for testing"""
        setup_single_process_distributed()

        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
            except Exception as e:
                pytest.skip(f"Could not initialize distributed backend: {e}")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

        # Set up environment variables
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["CUDA_MODULE_LOADING"] = "AUTO"

        yield rank, world_size

        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.fixture(scope="class")
    def test_engine(self, setup_distributed):
        """Setup test engine"""
        rank, world_size = setup_distributed

        if rank == 0:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            engine = AsyncEngine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                dtype="bfloat16",
                mem_fraction_static=0.3,
                enable_memory_saver=True,
                tp_size=world_size,
                disable_cuda_graph=True,
            )
            yield engine
            engine.shutdown()

        else:
            yield None

    @pytest.fixture(scope="class")
    def test_model(self):
        """Load test model"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )
            return model
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")

    @pytest.fixture(scope="class")
    def device_mesh(self, setup_distributed):
        """Create device mesh for testing"""
        rank, world_size = setup_distributed

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mesh")

        device_mesh_key = "tp"
        mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=(device_mesh_key,)
        )

        return device_mesh_key, mesh

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

    @pytest.mark.asyncio
    async def test_utils_update_weights(
        self, setup_distributed, test_engine, test_model, device_mesh
    ):
        """Test basic functionality of utils.update_weights"""
        rank, world_size = setup_distributed
        device_mesh_key, mesh = device_mesh

        # Create test parameters batch
        params_batch = self.create_test_params_batch(test_model, num_params=2)

        print(
            f"Rank {rank} testing utils.update_weights with {len(params_batch)} parameters"
        )
        # Test the utils.update_weights function
        result = await update_weights(
            engine=test_engine,
            params_batch=params_batch,
            device_mesh_key=device_mesh_key,
            device_mesh=mesh,
            load_format=None,
        )

        assert "Success" in result


if __name__ == "__main__":
    pytest.main([__file__])
