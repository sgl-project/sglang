import os
import subprocess
from pathlib import Path

from cuda.rust_build_inputs import compute_build_input_digest


def _initialize_checkout(root: Path, rust_source: str) -> None:
    (root / "rust/src").mkdir(parents=True)
    (root / "proto").mkdir()
    (root / "rust/src/lib.rs").write_text(rust_source)
    (root / "rust/Cargo.lock").write_text("version = 4\n")
    (root / "proto/service.proto").write_text('syntax = "proto3";\n')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "rust", "proto"], check=True)


def test_digest_uses_content_instead_of_checkout_timestamps(tmp_path: Path) -> None:
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    _initialize_checkout(checkout_a, 'pub const VALUE: &str = "a";\n')
    _initialize_checkout(checkout_b, 'pub const VALUE: &str = "b";\n')

    source_a = checkout_a / "rust/src/lib.rs"
    source_b = checkout_b / "rust/src/lib.rs"
    newer_time = source_a.stat().st_mtime + 60
    os.utime(source_a, (newer_time, newer_time))
    os.utime(source_b, (newer_time - 120, newer_time - 120))

    assert source_b.stat().st_mtime < source_a.stat().st_mtime
    assert compute_build_input_digest(checkout_a) != compute_build_input_digest(
        checkout_b
    )


def test_digest_is_stable_when_only_timestamps_change(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _initialize_checkout(checkout, 'pub const VALUE: &str = "same";\n')
    before = compute_build_input_digest(checkout)
    source = checkout / "rust/src/lib.rs"
    os.utime(source, (1, 1))
    assert compute_build_input_digest(checkout) == before


def test_digest_includes_generated_code_inputs(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _initialize_checkout(checkout, 'pub const VALUE: &str = "same";\n')
    before = compute_build_input_digest(checkout)
    (checkout / "proto/service.proto").write_text(
        'syntax = "proto3"; message Changed {}\n'
    )
    assert compute_build_input_digest(checkout) != before
