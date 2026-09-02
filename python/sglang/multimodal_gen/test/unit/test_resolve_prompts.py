import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


def _make_generator(prompt_file_path=None):
    """Return a DiffGenerator-shaped object with only server_args populated."""
    obj = object.__new__(DiffGenerator)
    obj.server_args = SimpleNamespace(prompt_file_path=prompt_file_path)
    return obj


class TestResolvePrompts(unittest.TestCase):
    # ---- inline prompt ----
    def test_none_prompt_returns_space(self):
        gen = _make_generator()
        self.assertEqual(gen._resolve_prompts(None), [" "])

    def test_string_prompt(self):
        gen = _make_generator()
        self.assertEqual(gen._resolve_prompts("hello"), ["hello"])

    def test_list_prompt(self):
        gen = _make_generator()
        self.assertEqual(gen._resolve_prompts(["a", "b"]), ["a", "b"])

    # ---- prompt_path (SamplingParams) ----
    def test_prompt_path_single_line(self):
        gen = _make_generator()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("sunset over the ocean\n")
            path = f.name
        try:
            result = gen._resolve_prompts(None, prompt_path=path)
            self.assertEqual(result, ["sunset over the ocean"])
        finally:
            os.unlink(path)

    def test_prompt_path_multi_line(self):
        gen = _make_generator()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("line one\n\nline two\n")
            path = f.name
        try:
            result = gen._resolve_prompts(None, prompt_path=path)
            self.assertEqual(result, ["line one", "line two"])
        finally:
            os.unlink(path)

    def test_prompt_path_takes_priority_over_server_args(self):
        with (
            tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f1,
            tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f2,
        ):
            f1.write("from prompt_path\n")
            f2.write("from server_args\n")
            path1, path2 = f1.name, f2.name
        try:
            gen = _make_generator(prompt_file_path=path2)
            result = gen._resolve_prompts(None, prompt_path=path1)
            self.assertEqual(result, ["from prompt_path"])
        finally:
            os.unlink(path1)
            os.unlink(path2)

    # ---- prompt_file_path (ServerArgs) ----
    def test_server_args_prompt_file_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("from server args\n")
            path = f.name
        try:
            gen = _make_generator(prompt_file_path=path)
            result = gen._resolve_prompts(None)
            self.assertEqual(result, ["from server args"])
        finally:
            os.unlink(path)

    # ---- error cases ----
    def test_missing_file_raises(self):
        gen = _make_generator()
        with self.assertRaises(FileNotFoundError):
            gen._resolve_prompts(None, prompt_path="/nonexistent/file.txt")

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("   \n\n  \n")
            path = f.name
        try:
            gen = _make_generator()
            with self.assertRaises(ValueError):
                gen._resolve_prompts(None, prompt_path=path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
