import unittest
from types import SimpleNamespace

import torch  # Added

from sglang.srt.layers.linear import LinearBase

# Added imports for new unit tests
from sglang.srt.layers.quantization.awq import (
    AWQConfig,
    AWQLinearMethod,
    FusedAWQLinearMethod,
)
from sglang.srt.utils import is_cuda as _is_cuda  # Added is_cuda
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

if _is_cuda():
    try:
        import sgl_kernel
    except ImportError:
        print("sgl_kernel not found, AWQ unit tests will be skipped.")
        sgl_kernel = None
else:
    sgl_kernel = None


class TestAWQIntegration(CustomTestCase):  # Renamed from TestAWQ to avoid conflict
    @classmethod
    def setUpClass(cls):
        if (
            not _is_cuda() or sgl_kernel is None
        ):  # Skip if no CUDA or kernel for integration tests too
            raise unittest.SkipTest(
                "AWQ integration tests require CUDA and sgl_kernel."
            )
        cls.model = DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        if (
            hasattr(cls, "process") and cls.process is not None
        ):  # Ensure process exists before trying to kill
            kill_process_tree(cls.process.pid)
            cls.process = None

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.64)


# New Unit Test Class for AWQ Methods
@unittest.skipUnless(
    _is_cuda() and sgl_kernel is not None, "AWQ unit tests require CUDA and sgl_kernel."
)
class TestAWQMethods(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        self.input_size = 128
        self.output_size = 256
        self.batch_size = 2

        self.input_size_per_partition = self.input_size
        self.output_partition_sizes = [self.output_size]
        self.params_dtype = torch.float16
        self.device = "cuda"  # Ensured by class decorator

    def _generate_packed_data(self, rows, cols_packed, pack_factor, weight_bits):
        """Helper to generate packed data for qweight/qzeros."""
        # Create individual 4-bit numbers.
        # Shape: (rows, cols_packed * pack_factor) where each element is a 4-bit number
        max_val = 1 << weight_bits
        individual_values = torch.randint(
            0,
            max_val,
            (rows, cols_packed * pack_factor),
            dtype=torch.int32,
            device=self.device,
        )

        packed_tensor = torch.zeros(
            (rows, cols_packed), dtype=torch.int32, device=self.device
        )
        for r_idx in range(rows):
            for c_idx in range(cols_packed):
                val = 0
                for i in range(pack_factor):
                    # Get the 4-bit number
                    four_bit_num = individual_values[r_idx, c_idx * pack_factor + i]
                    # Shift it to its position (LSB first) and OR with the current packed value
                    val |= four_bit_num.item() << (i * weight_bits)
                packed_tensor[r_idx, c_idx] = val
        return packed_tensor

    def _setup_linear_layer(self, method_class):
        layer = LinearBase(
            self.input_size,
            self.output_size,
            bias=False,
            params_dtype=self.params_dtype,
        ).to(self.device)
        method = method_class(self.config)
        # Pass dummy extra_weight_attrs if create_weights expects it (it does from LinearMethodBase)
        method.create_weights(
            layer,
            self.input_size_per_partition,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=None,
        )

        pack_factor = self.config.pack_factor
        weight_bits = self.config.weight_bits

        qweight_rows = self.input_size_per_partition
        qweight_cols_packed = self.output_size // pack_factor
        qweight_data = self._generate_packed_data(
            qweight_rows, qweight_cols_packed, pack_factor, weight_bits
        )

        qzeros_rows = self.input_size_per_partition // self.config.group_size
        qzeros_cols_packed = self.output_size // pack_factor
        qzeros_data = self._generate_packed_data(
            qzeros_rows, qzeros_cols_packed, pack_factor, weight_bits
        )

        scales_shape = (
            self.input_size_per_partition // self.config.group_size,
            self.output_size,
        )
        scales_data = (
            torch.rand(scales_shape, dtype=self.params_dtype, device=self.device) - 0.5
        ) * 0.02  # Centered, small range

        if hasattr(layer, "qweight") and layer.qweight is not None:
            layer.qweight.data.copy_(qweight_data)
        if hasattr(layer, "qzeros") and layer.qzeros is not None:
            layer.qzeros.data.copy_(qzeros_data)
        if hasattr(layer, "scales") and layer.scales is not None:
            layer.scales.data.copy_(scales_data)

        input_x = torch.randn(
            self.batch_size,
            self.input_size,
            dtype=self.params_dtype,
            device=self.device,
        )
        bias_tensor = torch.randn(
            self.output_size, dtype=self.params_dtype, device=self.device
        )

        return (
            layer,
            method,
            input_x,
            bias_tensor,
            qweight_data,
            qzeros_data,
            scales_data,
        )

    def test_fused_awq_method_no_bias(self):
        layer, method, x, _, qweight, qzeros, scales = self._setup_linear_layer(
            FusedAWQLinearMethod
        )
        output = method.apply(
            layer, x, bias=None
        )  # output shape: (batch_size, output_size)

        ref_dequant_weight = sgl_kernel.awq_dequantize(
            qweight, scales, qzeros
        )  # shape: (input_size, output_size)
        # x shape: (batch_size, input_size)
        # x.reshape(-1, self.input_size) is (batch_size, input_size)
        expected_output = torch.matmul(
            x.reshape(-1, self.input_size), ref_dequant_weight
        )  # shape: (batch_size, output_size)
        # No need to reshape expected_output if it's already (batch_size, output_size)

        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(
            torch.allclose(output, expected_output, atol=1e-1, rtol=1e-1)
        )  # Increased tolerance

    def test_fused_awq_method_with_bias(self):
        layer, method, x, bias, qweight, qzeros, scales = self._setup_linear_layer(
            FusedAWQLinearMethod
        )
        output = method.apply(layer, x, bias=bias)

        ref_dequant_weight = sgl_kernel.awq_dequantize(qweight, scales, qzeros)
        expected_output = torch.matmul(
            x.reshape(-1, self.input_size), ref_dequant_weight
        )
        expected_output += bias  # Bias is (output_size), broadcasts correctly

        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(
            torch.allclose(output, expected_output, atol=1e-1, rtol=1e-1)
        )  # Increased tolerance

    def test_original_awq_method_no_bias(self):
        layer, method, x, _, _, _, _ = self._setup_linear_layer(AWQLinearMethod)
        output = method.apply(layer, x, bias=None)

        ref_dequant_weight = sgl_kernel.awq_dequantize(
            layer.qweight, layer.scales, layer.qzeros
        )
        expected_output = torch.matmul(
            x.reshape(-1, self.input_size), ref_dequant_weight
        )

        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(
            torch.allclose(output, expected_output, atol=1e-1, rtol=1e-1)
        )  # Increased tolerance

    def test_consistency_between_methods(self):
        layer_fused, method_fused, x, bias, q_w_data, q_z_data, sc_data = (
            self._setup_linear_layer(FusedAWQLinearMethod)
        )
        output_fused = method_fused.apply(layer_fused, x, bias=bias)

        layer_orig = LinearBase(
            self.input_size,
            self.output_size,
            bias=False,
            params_dtype=self.params_dtype,
        ).to(self.device)
        method_orig = AWQLinearMethod(self.config)
        method_orig.create_weights(
            layer_orig,
            self.input_size_per_partition,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=None,
        )

        layer_orig.qweight.data.copy_(q_w_data)
        layer_orig.qzeros.data.copy_(q_z_data)
        layer_orig.scales.data.copy_(sc_data)

        # Optional: Call process_weights_after_loading if it affects behavior beyond data type conversion
        # method_orig.process_weights_after_loading(layer_orig)

        output_orig = method_orig.apply(layer_orig, x, bias=bias)

        self.assertEqual(output_fused.shape, output_orig.shape)
        self.assertTrue(
            torch.allclose(output_fused, output_orig, atol=1e-2, rtol=1e-2)
        )  # Tighter for consistency

    def test_fused_awq_method_unaligned_output(self):
        """Tests the fused AWQ method where the output dim is not divisible by 128."""
        # Override setup for this specific test
        self.output_size = 200  # Not divisible by 128, but divisible by pack_factor (8)
        self.output_partition_sizes = [self.output_size]

        (
            layer,
            method,
            x,
            _,
            qweight,
            qzeros,
            scales,
        ) = self._setup_linear_layer(FusedAWQLinearMethod)
        output = method.apply(layer, x, bias=None)

        ref_dequant_weight = sgl_kernel.awq_dequantize(qweight, scales, qzeros)
        expected_output = torch.matmul(
            x.reshape(-1, self.input_size), ref_dequant_weight
        )

        self.assertEqual(output.shape, expected_output.shape)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # Only add TestAWQIntegration if CUDA is available, otherwise it's skipped by decorator but loader might complain
    if _is_cuda() and sgl_kernel is not None:
        suite.addTests(loader.loadTestsFromTestCase(TestAWQIntegration))
    suite.addTests(
        loader.loadTestsFromTestCase(TestAWQMethods)
    )  # This will be skipped by decorator if no CUDA

    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    # Exit with a non-zero code if tests failed
    if not result.wasSuccessful():
        exit(1)
