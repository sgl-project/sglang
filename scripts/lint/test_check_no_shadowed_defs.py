import pathlib
import tempfile
import unittest

from check_no_shadowed_defs import find_shadowed_defs


class TestFindShadowedDefs(unittest.TestCase):
    def check_source(self, source: str):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "example.py"
            path.write_text(source, encoding="utf-8")
            return find_shadowed_defs(path)

    def test_reports_shadowed_method(self):
        source = """
class Backend:
    def forward(self):
        return "dead"

    def forward(self):
        return "live"
"""
        self.assertEqual(self.check_source(source), [("Backend", "forward", 3, 6)])

    def test_accepts_single_definition(self):
        source = """
class Backend:
    def forward(self):
        return 1
"""
        self.assertEqual(self.check_source(source), [])

    def test_accepts_same_name_in_different_classes(self):
        source = """
class A:
    def forward(self):
        return 1


class B:
    def forward(self):
        return 2
"""
        self.assertEqual(self.check_source(source), [])

    def test_accepts_same_name_in_a_nested_class(self):
        source = """
class Outer:
    def forward(self):
        return 1

    class Inner:
        def forward(self):
            return 2
"""
        self.assertEqual(self.check_source(source), [])

    def test_accepts_property_accessors(self):
        source = """
class Config:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
"""
        self.assertEqual(self.check_source(source), [])

    def test_accepts_typing_overloads(self):
        source = """
class Loader:
    @overload
    def load(self, name: str) -> str: ...

    @typing.overload
    def load(self, name: int) -> int: ...

    def load(self, name):
        return name
"""
        self.assertEqual(self.check_source(source), [])

    def test_reports_async_method(self):
        source = """
class Client:
    async def send(self):
        return "dead"

    async def send(self):
        return "live"
"""
        self.assertEqual(self.check_source(source), [("Client", "send", 3, 6)])

    def test_reports_every_extra_definition(self):
        source = """
class Backend:
    def forward(self):
        return 1

    def forward(self):
        return 2

    def forward(self):
        return 3
"""
        self.assertEqual(
            self.check_source(source),
            [("Backend", "forward", 3, 6), ("Backend", "forward", 6, 9)],
        )

    def test_ignores_module_level_redefinition(self):
        source = """
def helper():
    return 1


def helper():
    return 2
"""
        self.assertEqual(self.check_source(source), [])

    def test_ignores_unparsable_file(self):
        self.assertEqual(self.check_source("class Backend\n"), [])


if __name__ == "__main__":
    unittest.main()
