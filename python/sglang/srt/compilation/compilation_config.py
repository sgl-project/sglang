# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/compilation_config.py

import json
from typing import List


# TODO(Yuwei): support better compile config support
class CompilationConfig:
    def __init__(
        self,
        capture_sizes: List[int] = [],
        compiler: str = "eager",
        enable_debug_mode: bool = False,
        splitting_ops: List[str] = [],
    ):
        self.traced_files = set()
        self.capture_sizes = capture_sizes
        self.compiler = compiler
        self.enable_debug_mode = enable_debug_mode
        self.splitting_ops = splitting_ops

    def add_traced_file(self, file_path: str):
        self.traced_files.add(file_path)

    def get_traced_files(self):
        return self.traced_files

    def get_capture_sizes(self):
        return self.capture_sizes

    @classmethod
    def from_cli(cls, args) -> "CompilationConfig":
        args_dict = json.loads(args)
        return CompilationConfig(**args_dict)

    def get_enable_debug_mode(self):
        return self.enable_debug_mode

    def get_splitting_ops(self):
        return self.splitting_ops
