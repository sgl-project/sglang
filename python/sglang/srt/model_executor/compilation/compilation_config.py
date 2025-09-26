class CompilationConfig:
    def __init__(self):
        self.traced_files = set()

    def add_traced_file(self, file_path: str):
        self.traced_files.add(file_path)

    def get_traced_files(self):
        return self.traced_files
