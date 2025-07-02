import os
import unittest
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path

import pynvml
import pytest
import tensorrt as trt
import tensorrt_llm
import torch
from cuda import cuda, nvrtc
from parameterized import parameterized
from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import Session, TensorInfo


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


# ref: https://github.com/NVIDIA/cuda-python/blob/main/examples/extra/jit_program_test.py
def getSMVersion():
    # Init
    (err,) = cuda.cuInit(0)
    ASSERT_DRV(err)

    # Device
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Get target architecture
    err, sm_major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
    )
    ASSERT_DRV(err)
    err, sm_minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
    )
    ASSERT_DRV(err)

    return sm_major * 10 + sm_minor


def getCUDAVersion():
    import subprocess

    try:
        cuda_version = subprocess.run(
            ["nvcc", "--version"], stdout=subprocess.PIPE, universal_newlines=True
        )
        output = cuda_version.stdout.split()
        release_version = output[-4].replace(",", ".").split(".")
        return int(release_version[0]) * 100 + int(release_version[1])
    except Exception as e:
        print(f"Error getting CUDA version: {e}")


skip_pre_ada = pytest.mark.skipif(
    getSMVersion() < 89, reason="This test is not supported in pre-Ada architecture"
)
skip_pre_hopper = pytest.mark.skipif(
    getSMVersion() < 90, reason="This test is not supported in pre-Hopper architecture"
)
skip_pre_blackwell = pytest.mark.skipif(
    getSMVersion() < 100,
    reason="This test is not supported in pre-Blackwell architecture",
)
skip_blackwell = pytest.mark.skipif(
    getSMVersion() == 100, reason="This test is not supported in Blackwell architecture"
)
skip_blackwell_geforce = pytest.mark.skipif(
    getSMVersion() == 120, reason="This test is not supported on SM 120"
)

# If used together with @parameterized, we have to use unittest.skipIf instead of pytest.mark.skipif
skip_pre_ada_unittest = unittest.skipIf(
    getSMVersion() < 89 or (getSMVersion() == 89 and getCUDAVersion() < 1204),
    reason="This test is not supported in pre-Ada architecture, and for Ada we require cuda version >= 12.4",
)
skip_pre_hopper_unittest = unittest.skipIf(
    getSMVersion() < 90, reason="This test is not supported in pre-Hopper architecture"
)
skip_pre_blackwell_unittest = unittest.skipIf(
    getSMVersion() < 100,
    reason="This test is not supported in pre-Blackwell architecture",
)
skip_non_ada_unittest = unittest.skipIf(
    getSMVersion() != 89, reason="This test is only supported in Ada architecture"
)
skip_non_hopper_unittest = unittest.skipIf(
    getSMVersion() != 90, reason="This test is only supported in Hopper architecture"
)
skip_neither_ada_nor_hopper_unittest = unittest.skipIf(
    getSMVersion() != 90 and getSMVersion() != 89,
    reason="This test is only supported in Ada or Hopper architecture",
)

IGNORE_ARCH = os.environ.get("TLLM_TEST_IGNORE_ARCH", False)
force_ampere = pytest.mark.skipif(
    (not IGNORE_ARCH) and (getSMVersion() < 80 or getSMVersion() > 89),
    reason="This test is only enabled in Ampere architecture",
)


def is_bf16(dtype):
    return dtype == "bfloat16" or dtype == "bf16" or dtype == torch.bfloat16


def skip_fp8_pre_ada(use_fp8):
    if use_fp8 and getSMVersion() < 89:
        pytest.skip("FP8 is not supported on pre-Ada architectures")


def skip_blackwell_for_fmha_tests(context_fmha_type, head_size):
    if getSMVersion() == 100 and (
        head_size not in [32, 64, 128] and context_fmha_type != ContextFMHAType.disabled
    ):
        pytest.skip(
            "Context FMHA only supports head sizes [32, 64, 128] currently on blackwell."
        )


def skip_fp4_pre_blackwell(use_fp4):
    if use_fp4 and getSMVersion() < 100:
        pytest.skip("FP4 is not supported on pre-Blackwell architectures")


def skip_bf16_fp32_accum(dtype, context_fmha_type):
    if is_bf16(dtype) and context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
        pytest.skip(
            "bfloat16 Context FMHA will always accumulate on FP32, so it has been tested with ContextFMHAType.enabled"
        )


def skip_num_gpus_less_than(num_gpus: int):
    return pytest.mark.skipif(
        torch.cuda.device_count() < num_gpus,
        reason=f"The test needs at least {num_gpus} GPUs, skipping",
    )


skip_single_gpu = skip_num_gpus_less_than(2)


def compose_decorator(*decorators):

    def composed_decorator(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return composed_decorator


pytest.mark.gpu2 = compose_decorator(skip_single_gpu, pytest.mark.gpu2)
pytest.mark.gpu4 = compose_decorator(skip_num_gpus_less_than(4), pytest.mark.gpu4)


def skip_gpu_memory_less_than(required_memory: int):
    memory = get_total_gpu_memory(0)
    return pytest.mark.skipif(
        required_memory > memory,
        reason=f"Not enough GPU memory for this test (wanted {required_memory}, have {memory})",
    )


skip_gpu_memory_less_than_40gb = skip_gpu_memory_less_than(40 * 1024 * 1024 * 1024)

skip_gpu_memory_less_than_80gb = skip_gpu_memory_less_than(80 * 1024 * 1024 * 1024)

skip_gpu_memory_less_than_138gb = skip_gpu_memory_less_than(138 * 1024 * 1024 * 1024)


def modelopt_installed():
    try:
        # isort: off
        import modelopt.torch.quantization as atq  # NOQA
        from modelopt.torch.export import export_tensorrt_llm_checkpoint  # NOQA

        # isort: on
        return True
    except Exception:
        return False


skip_no_modelopt = unittest.skipIf(
    not modelopt_installed(), reason="Modelopt is not installed"
)


def check_nvlink():
    "check nvlink is active"
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                print(f"Link state is {link_state}")

            except pynvml.NVMLError as error:
                print(
                    f"Device does not seem to support NVLink or there's an issue: {error}"
                )
        pynvml.nvmlShutdown()
        return True

    except pynvml.NVMLError as error:
        print(f"Error initializing NVML or other NVML error: {error}")
        return False


skip_nvlink_inactive = unittest.skipIf(not check_nvlink(), reason="nvlink is inactive.")


# This function names will make all unit tests names to show the values of all parameters in @parameterized.expand
def unittest_name_func(testcase_func, param_num, param):
    expand_params = lambda params: "_".join(
        [expand_params(x) if isinstance(x, (list, tuple)) else str(x) for x in params]
    )
    name = expand_params(param.args)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(name),
    )


def set_input_shape(
    profile, inp: tensorrt_llm.Tensor, shape: tuple, data: torch.Tensor = None
):
    set_input_shapes(profile, inp, shape, shape, shape, data)
    return


def set_input_shapes(
    profile,
    inp: tensorrt_llm.Tensor,
    min_shape: tuple,
    opt_shape: tuple,
    max_shape: tuple,
    data: torch.Tensor = None,
):
    if inp.trt_tensor.is_shape_tensor:
        # For shape tensors, TensorRT expects the full tensor (on CPU), not just shape
        assert (
            data is not None
        ), f"For shape tensor {inp.name}, TensorRT needs the tensor value."
        assert str(data.device) == "cpu", (
            f"Shape tensor's data needs to be on CPU "
            f"(device found={data.device}) for both updating the profile and for execution."
        )
        np_data = data.flatten().numpy()
        profile.set_shape_input(inp.name, np_data, np_data, np_data)
        return
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    return


def create_session(
    builder,
    network,
    precision="float32",
    int8=False,
    fp8=False,
    memory_pool_limit=None,
    optimization_profiles=[],
    quant_mode=QuantMode(0),
):
    """
    This function creates an engine and a tensorrt_llm.runtime.Session for the engine.
    Args:
        network: a tensorrt_llm.Network object
        precision: the precision of the network, choose from ["float32", "float16", "bfloat16"]
        **kwargs: builder flags such as int8, fp8, etc.
    Returns:
        session: a tensorrt_llm.runtime.Session
    """
    builder_config = builder.create_builder_config(
        precision=precision, int8=int8, fp8=fp8, quant_mode=quant_mode
    )
    # Some tests require to set mem pool limit to avoid OOM
    if memory_pool_limit is not None:
        builder_config.trt_builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, memory_pool_limit
        )
    # Some tests include shape tensors, so the optimization profile needs to be feed in explicitly
    if len(optimization_profiles) > 0:
        for profile in optimization_profiles:
            builder_config.trt_builder_config.add_optimization_profile(profile)
    # Disable TF32 for accuracy in testing.
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, "Failed to build engine"
    session = Session.from_serialized_engine(engine)
    return session


def run_session(
    session: Session, inputs, outputs={}, override_shapes={}, override_types={}
):
    """
    The current session object needs to pass in both inputs and outputs bindings.
    For test convenience, create a function that infers output shapes automatically,
    This function is similar to tensorrt_llm.runtime.Session._debug_run, and Polygraphy runner.infer,
    where only input shape is required.
    NOTES:
        1. The outputs dictionary is required for outputs for which the shapes cannot be inferred.
           This function will prioritize to use the tensor in this dictionary.
        2. `override_shapes` can be used to force some input tensors' shape to be different than the passed tensor.
           Required for zero-volume tensors since torch.Tensor.data_ptr() is nullptr for such tensors.
        3. `override_types` can be used to force some input tensors' type to be different than the passed tensor.
           Required for zero-volume tensors since torch.Tensor.data_ptr() is nullptr for such tensors.
    """

    # Prepare output tensors.
    output_info = session.infer_shapes(
        [
            TensorInfo(
                name,
                torch_dtype_to_trt(
                    tensor.dtype if name not in override_types else override_types[name]
                ),
                tensor.shape if name not in override_shapes else override_shapes[name],
            )
            for name, tensor in inputs.items()
        ]
    )

    def create_torch(t):
        if t.dtype == trt.fp4:
            shape = list(t.shape)
            shape[-1] = shape[-1] // 2
            return torch.empty(tuple(shape), dtype=torch.uint8, device="cuda")
        else:
            return torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )

    outputs = {
        t.name: create_torch(t) if t.name not in outputs else outputs[t.name]
        for t in output_info
    }

    # Execute model inference
    stream = torch.cuda.current_stream()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    assert ok, "Engine execution failed"
    stream.synchronize()

    return outputs


def similarity_score(a, b):
    "similar compare a and b"
    return SequenceMatcher(None, a, b).ratio()


def similar(a, b, threshold=0.8):
    "similar compare a and b"
    return similarity_score(a, b) >= threshold


def get_project_root(test_file: str) -> Path:
    return next(
        p
        for p in Path(test_file).resolve().parents
        if (p / "tests").is_dir() and (p / "tensorrt_llm").is_dir()
    )


@contextmanager
def default_dtype(dtype: torch.dtype):
    cur_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(cur_default)


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


def woq_groupwise_gt_matmul(mat1, ref_torch_weights, bias=None):
    ref = torch.matmul(mat1, ref_torch_weights)
    if bias is not None:
        ref += bias
    return ref
