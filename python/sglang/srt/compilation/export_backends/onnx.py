"""ONNX Runtime export backend for ``@sgl_compile`` callsites."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.compilation.export_backends.base import ExportRuntime


class OnnxExportRuntime(ExportRuntime):
    """Convert exported programs to ONNX and run them with ONNX Runtime."""

    export_format = "onnx"

    def __init__(
        self,
        providers: list[str] | None = None,
        prefer_cuda_iobinding: bool = True,
    ):
        self.providers = providers
        self.prefer_cuda_iobinding = prefer_cuda_iobinding

    def prepare_runtime(self, exported_program, artifact, compile_config):
        onnx_path = artifact.runtime_artifact_path
        if onnx_path is None:
            raise RuntimeError("SGLANG_EXPORT_DIR is required for ONNX export.")

        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX export requires optional packages: onnx, onnxscript, "
                "onnxruntime-gpu or onnxruntime, and numpy."
            ) from exc

        artifact.validate_metadata()
        if not onnx_path.exists():
            if artifact.mode == "load_only":
                raise FileNotFoundError(
                    f"ONNX artifact {onnx_path} is required by load_only mode."
                )
            artifact.ensure_export_dir()
            torch.onnx.export(exported_program, args=(), f=str(onnx_path), dynamo=True)
        artifact.write_metadata()

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        selected_providers = self._select_providers(ort)
        session = ort.InferenceSession(str(onnx_path), providers=selected_providers)
        input_names = [input_info.name for input_info in session.get_inputs()]
        output_infos = session.get_outputs()

        def run_onnx(*args, **kwargs):
            if kwargs:
                raise TypeError("ONNX runtime only supports positional arguments.")

            tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
            if len(tensor_args) != len(input_names):
                raise TypeError(
                    f"ONNX runtime expected {len(input_names)} tensor inputs, "
                    f"got {len(tensor_args)}."
                )

            if (
                self.prefer_cuda_iobinding
                and compile_config.copy_output_to_arg_index is not None
                and "CUDAExecutionProvider" in session.get_providers()
            ):
                result = _try_run_cuda_iobinding(
                    session,
                    input_names,
                    output_infos,
                    tensor_args,
                    args,
                    compile_config.copy_output_to_arg_index,
                )
                if result is not _IOBINDING_UNSUPPORTED:
                    run_onnx.execution_mode = "cuda_iobinding"
                    return result

            run_onnx.execution_mode = "copied"
            return _run_copied(session, input_names, tensor_args, args, compile_config, np)

        run_onnx.onnx_path = onnx_path
        run_onnx.providers = session.get_providers()
        run_onnx.execution_mode = "uninitialized"
        return run_onnx

    def _select_providers(self, ort) -> list[str]:
        if self.providers is not None:
            return self.providers
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


_IOBINDING_UNSUPPORTED = object()


def _run_copied(session, input_names, tensor_args, args, compile_config, np):
    device = tensor_args[0].device if tensor_args else torch.device("cpu")
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, tensor_args)
    }
    outputs = session.run(None, ort_inputs)
    tensors = [
        torch.as_tensor(np.asarray(output), device=device) for output in outputs
    ]
    if compile_config.copy_output_to_arg_index is not None:
        dst = args[compile_config.copy_output_to_arg_index]
        dst.copy_(tensors[0])
        return None
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def _try_run_cuda_iobinding(
    session,
    input_names: list[str],
    output_infos: list[Any],
    tensor_args: list[torch.Tensor],
    original_args: tuple[Any, ...],
    copy_output_to_arg_index: int | None,
):
    if not tensor_args or not all(tensor.device.type == "cuda" for tensor in tensor_args):
        return _IOBINDING_UNSUPPORTED
    if not all(tensor.is_contiguous() for tensor in tensor_args):
        return _IOBINDING_UNSUPPORTED

    try:
        io_binding = session.io_binding()
        for name, tensor in zip(input_names, tensor_args):
            element_type = _torch_dtype_to_numpy_dtype(tensor.dtype)
            io_binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=tensor.device.index or 0,
                element_type=element_type,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        if copy_output_to_arg_index is not None:
            dst = original_args[copy_output_to_arg_index]
            if not isinstance(dst, torch.Tensor) or not dst.is_contiguous():
                return _IOBINDING_UNSUPPORTED
            if dst.device.type != "cuda":
                return _IOBINDING_UNSUPPORTED
            io_binding.bind_output(
                name=output_infos[0].name,
                device_type="cuda",
                device_id=dst.device.index or 0,
                element_type=_torch_dtype_to_numpy_dtype(dst.dtype),
                shape=tuple(dst.shape),
                buffer_ptr=dst.data_ptr(),
            )
        else:
            return _IOBINDING_UNSUPPORTED

        session.run_with_iobinding(io_binding)
        return None
    except Exception:
        return _IOBINDING_UNSUPPORTED


def _torch_dtype_to_numpy_dtype(dtype: torch.dtype):
    import numpy as np

    dtype_map = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    if dtype not in dtype_map:
        raise TypeError(f"Unsupported dtype for ONNX Runtime I/O binding: {dtype}")
    return dtype_map[dtype]


__all__ = ["OnnxExportRuntime"]
