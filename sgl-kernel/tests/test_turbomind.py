# adapt from https://github.com/InternLM/turbomind/blob/main/example/test_linear.py
import unittest

import torch
import torch.nn as nn
from sgl_kernel import turbomindLinear

from sglang.srt.layers.quantization.turbomind_utils import pack_u4_row, unpack_awq_gemm


class TestInt8Gemm(unittest.TestCase):
    def _i32x8_to_i4x8(self, w):
        """merge 8 integers (range from 0 to 15) into one 32-bit integer."""
        assert w.shape[-1] % 8 == 0
        shape = (w.shape[0], w.numel() // (w.shape[0] * 8), 8)
        shape = shape[:-1] + (1,)
        result = torch.zeros(shape, dtype=w.dtype, device=w.device)
        mask = torch.tensor([15], dtype=w.dtype, device=w.device)
        for i in range(8):
            shift = 4 * (7 - i)
            result[..., 0] |= (w[..., i] & mask) << shift
        result = result.view(w.shape[0], -1)
        return result

    def _makeup_weights(
        self, in_features: int, out_features: int, group_size: int = 128
    ):
        assert out_features % 8 == 0
        qweight = torch.randint(
            0, 16, (in_features, out_features // 8, 8), dtype=torch.int32, device="cuda"
        )
        # print(f'-- makeup qweight: shape {qweight.shape}')
        # print(qweight.view(in_features, -1))
        qweight = self._i32x8_to_i4x8(qweight)
        # print(f'-- merge qweight: shape {qweight.shape}')
        # print(qweight)

        # make up qzeros
        assert in_features % group_size == 0 and in_features // group_size >= 1
        qzeros = torch.randint(
            0,
            16,
            (in_features // group_size, out_features // 8, 8),
            dtype=torch.int32,
            device="cuda",
        )
        # print(f'-- makeup qzero: shape {qzeros.shape}')
        # print(qzeros.view(in_features // group_size, -1))
        qzeros = self._i32x8_to_i4x8(qzeros)
        # print(f'-- merge qzero: shape {qzeros.shape}\n{qzeros}')

        # make up scales
        scales = torch.rand(
            (in_features // group_size, out_features),
            dtype=torch.float16,
            device="cuda",
        )
        # print(f'-- makeup scales: shape {scales.shape}\n{scales}')
        return qweight, qzeros, scales

    def _dequantize(self, group_size: int = 128):
        _qweight = unpack_awq_gemm(self.qweight)
        _qzeros = unpack_awq_gemm(self.qzeros)
        _qzeros = _qzeros.float()
        _qweight = _qweight.float()
        _scales = self.scales.float()
        for i in range(self.qzeros.shape[0]):
            start = i * group_size
            end = start + group_size
            _qweight[start:end] = (
                _qweight[start:end, :] - _qzeros[i : i + 1, :]
            ) * _scales[i : i + 1, :]
        return _qweight.half()

    def _post_init(self):
        assert self.qweight.device.type == "cuda"

        self.qweight = unpack_awq_gemm(self.qweight)
        self.qzeros = unpack_awq_gemm(self.qzeros)
        self.scales = self.scales

        self.qweight = pack_u4_row(self.qweight)
        self.qzeros = self.qzeros.to(torch.half)

        device_id = self.qweight.device.index
        properties = torch.cuda.get_device_properties(device_id)

        def is_16xx_series(name):
            import re

            pattern = r"GTX 16\d\d"
            return bool(re.search(pattern, name))

        simt = is_16xx_series(properties.name)
        self.qweight = self.qweight.contiguous()
        self.scales = self.scales.contiguous()
        self.qzeros = self.qzeros.contiguous()
        self.model.post_init(self.qweight, self.scales, self.qzeros, simt)

    def test_accuracy(self):
        test_cases = [
            (32, 512, 256, 128, 4),
            (1, 1024, 512, 128, 4),
        ]

        for batch_size, in_features, out_features, group_size, w_bit in test_cases:
            with self.subTest(
                msg=f"B{batch_size}_In{in_features}_Out{out_features}_G{group_size}"
            ):

                self.qweight, self.qzeros, self.scales = self._makeup_weights(
                    in_features, out_features, group_size
                )

                x = torch.randn(
                    (batch_size, in_features),
                    device=self.qweight.device,
                    dtype=torch.float16,
                )

                weight = self._dequantize(group_size)
                # print(f'-- dequantization: weight.shape={weight.shape}, weight: \n{weight}')
                ref_linear = nn.Linear(
                    in_features, out_features, bias=False, device="cuda"
                )
                with torch.no_grad():
                    ref_linear.weight = nn.Parameter(weight.T)
                    ref_res = ref_linear(x)
                    print(f"nn.linear.res: {ref_res}")

                self.model = turbomindLinear(
                    in_features,
                    out_features,
                    w_bit,
                    group_size,
                )

                self._post_init()

                stream = torch.cuda.current_stream()
                res = torch.empty(
                    (x.shape[0], out_features),
                    dtype=torch.float16,
                    device=x.device,
                )
                self.model.forward(x, res, stream.cuda_stream)
                out_shape = x.shape[:-1] + (out_features,)
                res = torch.from_dlpack(res).view(out_shape)
                stream.synchronize()

                print(f"turbomind.linear.res: {res}")

                abs_diff = torch.abs(res - ref_res).float()
                rel_diff = abs_diff / torch.max(torch.abs(ref_res), torch.abs(res))
                rtol = 0.01
                atol = 0.0001
                outliers = abs_diff > atol + rtol * torch.abs(ref_res)
                abs_diff = torch.sum(abs_diff) / abs_diff.numel()
                rel_diff = torch.sum(rel_diff) / rel_diff.numel()
                outliers = torch.sum(outliers) / outliers.shape[0]
                print(
                    f"abs_diff {abs_diff:4f}, "
                    f"rel_diff {rel_diff:4f}, "
                    f"outliers {outliers:4f}"
                )
                self.assertLessEqual(abs_diff, 0.05)
                self.assertLessEqual(rel_diff, 0.01)


if __name__ == "__main__":
    unittest.main()
