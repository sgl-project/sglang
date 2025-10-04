# TODO(Yuwei): support better compile config support
class CompilationConfig:
    def __init__(self):
        self.traced_files = set()

    def add_traced_file(self, file_path: str):
        self.traced_files.add(file_path)

    def get_traced_files(self):
        return self.traced_files

    def get_capture_sizes(self):
        return [
            1,
            2,
            4,
            8,
            16,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            176,
            192,
            208,
            224,
            240,
            256,
            288,
            320,
            352,
            384,
            416,
            448,
            480,
            512,
        ]
