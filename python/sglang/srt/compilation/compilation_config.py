# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/compilation_config.py

from typing import List


# TODO(Yuwei): support better compile config support
class CompilationConfig:
    def __init__(self, capture_sizes: List[int]):
        self.traced_files = set()
        self.capture_sizes = capture_sizes

    def add_traced_file(self, file_path: str):
        self.traced_files.add(file_path)

    def get_traced_files(self):
        return self.traced_files

    def get_capture_sizes(self):
        return self.capture_sizes
