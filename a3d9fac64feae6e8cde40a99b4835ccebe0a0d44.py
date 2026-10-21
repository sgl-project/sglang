from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mechanical_refactor_reproduction_utils import verify_mechanical_refactor


def transform(root):
    cp = root / "python/sglang/srt/layers/cp"
    package = cp / "bcg"
    package.mkdir()
    for name in ("bcg.py", "interleave_bcg.py"):
        (cp / name).rename(package / name)
    (package / "__init__.py").touch()
    for directory in ("python", "test"):
        for path in (root / directory).rglob("*.py"):
            before = path.read_text()
            after = before.replace(
                "from sglang.srt.layers.cp.bcg import ",
                "from sglang.srt.layers.cp.bcg.bcg import ",
            ).replace(
                "from sglang.srt.layers.cp.interleave_bcg import ",
                "from sglang.srt.layers.cp.bcg.interleave_bcg import ",
            )
            if before != after:
                path.write_text(after)


if __name__ == "__main__":
    verify_mechanical_refactor('68c42ba6e389c1d2b6387615f227c5720ff635cb', 'a3d9fac64feae6e8cde40a99b4835ccebe0a0d44', transform)
