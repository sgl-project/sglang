import ctypes
import threading
from functools import wraps

import torch
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided_into,
)
from sglang.srt.speculative.spec_utils import traverse_tree


class GrammarHostCallback:
    def __init__(self, vocab_mask, stride):
        self.vocab_mask = vocab_mask
        self.stride = stride
        self.tokens_cpu = torch.zeros(
            vocab_mask.shape[0], dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.mask_cpu = torch.full_like(vocab_mask, -1, device="cpu", pin_memory=True)
        self.tokens_device = torch.zeros_like(self.tokens_cpu, device=vocab_mask.device)
        self.next_token = torch.arange(1, stride + 1, device="cpu")
        self.next_token[-1] = -1
        self.next_sibling = torch.full((stride,), -1, device="cpu")
        self.stream = torch.cuda.Stream(device=vocab_mask.device)
        self.done = threading.Event()
        self.grammars = None
        self.error = None
        self.in_flight = False
        # cuda.bindings' trampoline frees its userdata after the first replay.
        self.launch = ctypes.CDLL("libcuda.so.1").cuLaunchHostFunc
        self.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.launch.restype = ctypes.c_int
        self.callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self._build_mask)

    def prepare(self, grammar, barrier):
        if self.error is not None:
            raise RuntimeError("Grammar host callback failed") from self.error
        if self.in_flight:
            raise RuntimeError("Grammar host callback still in flight")
        grammars = tuple(grammar) if isinstance(grammar, (list, tuple)) else (grammar,)
        if len(grammars) * self.stride > self.tokens_cpu.numel():
            raise ValueError("Grammar batch exceeds the captured mask capacity")
        if barrier is not None:
            barrier()
        self.done.clear()
        self.grammars = grammars
        self.in_flight = True

    def finish(self):
        # Bounds callback completion, not the target GPU forward.
        if not self.done.wait(timeout=30):
            raise RuntimeError("Grammar host callback did not complete within 30s")
        self.in_flight = False
        if self.error is not None:
            raise RuntimeError("Grammar host callback failed") from self.error

    def _build_mask(self, num_rows):
        grammars, self.grammars = self.grammars, None
        try:
            self.mask_cpu[:num_rows].fill_(-1)
            if grammars is not None:
                for i, grammar in enumerate(grammars):
                    if grammar is not None:
                        rows = slice(i * self.stride, (i + 1) * self.stride)
                        traverse_tree(
                            self.next_token,
                            self.next_sibling,
                            self.tokens_cpu[rows],
                            grammar,
                            self.mask_cpu[rows],
                            vocab_size=grammar.vocab_size,
                        )
        except BaseException as exc:
            self.error = exc
            self.mask_cpu.zero_()
        finally:
            if grammars is not None:
                self.done.set()

    def bind(self, model):
        original = model.forward

        @wraps(original)
        def forward(input_ids, positions, forward_batch, *args, **kwargs):
            if (
                not forward_batch.forward_mode.is_target_verify()
                or input_ids.numel() > self.tokens_cpu.numel()
            ):
                return original(input_ids, positions, forward_batch, *args, **kwargs)
            current = torch.cuda.current_stream()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                layout = getattr(
                    getattr(forward_batch, "spec_info", None),
                    "ragged_verify_layout",
                    None,
                )
                tokens = input_ids
                if layout is not None:
                    rows = forward_batch.batch_size * self.stride
                    tokens = scatter_compact_to_strided_into(
                        compact=input_ids.view(-1, 1),
                        verify_lens=layout.verify_lens,
                        out=self.tokens_device[:rows].view(-1, 1),
                        stride=self.stride,
                        fill_value=0,
                    ).view(-1)
                self.tokens_cpu[: tokens.numel()].copy_(tokens, non_blocking=True)
                # CUDA retains the row count as opaque userdata for each graph shape.
                error = self.launch(
                    self.stream.cuda_stream,
                    self.callback,
                    max(tokens.numel(), self.stride),
                )
                if error:
                    raise RuntimeError(f"cuLaunchHostFunc failed: {error}")
                self.vocab_mask[: tokens.numel()].copy_(
                    self.mask_cpu[: tokens.numel()], non_blocking=True
                )
            out = original(input_ids, positions, forward_batch, *args, **kwargs)
            # Only acceptance needs the mask; target compute overlaps its CPU build.
            current.wait_stream(self.stream)
            return out

        model.forward = forward
