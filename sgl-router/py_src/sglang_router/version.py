"""Version information for sglang-router."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _read_field_from_pyproject(field: str) -> Optional[str]:
    """Read a field from pyproject.toml using TOML parser."""
    # Try to find pyproject.toml relative to this file
    current_file = Path(__file__).resolve()
    # Go up from py_src/sglang_router/version.py to project root
    project_root = current_file.parent.parent.parent

    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return None

    try:
        # Try Python 3.11+ built-in tomllib
        if sys.version_info >= (3, 11):
            import tomllib

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                project = data.get("project", {})
                value = project.get(field)
                if value is not None:
                    return str(value)
        else:
            # Fallback to tomli for Python 3.8-3.10
            try:
                import tomli

                with open(pyproject_path, "rb") as f:
                    data = tomli.load(f)
                    project = data.get("project", {})
                    value = project.get(field)
                    if value is not None:
                        return str(value)
            except ImportError:
                # If tomli is not available, try toml (pip install toml)
                try:
                    import toml

                    with open(pyproject_path, "r", encoding="utf-8") as f:
                        data = toml.load(f)
                        project = data.get("project", {})
                        value = project.get(field)
                        if value is not None:
                            return str(value)
                except ImportError:
                    pass
    except Exception:
        pass

    return None


def _get_project_name() -> str:
    """Read project name from pyproject.toml."""
    name = _read_field_from_pyproject("name")
    if name:
        return name

    # Fallback: try importlib.metadata if package is installed
    try:
        if sys.version_info >= (3, 8):
            from importlib.metadata import metadata

            meta = metadata("sglang-router")
            if meta:
                return meta.get("Name", "sglang-router")
    except Exception:
        pass

    # Final fallback
    return "sglang-router"


def _get_version_from_pyproject() -> str:
    """Read version from pyproject.toml."""
    version = _read_field_from_pyproject("version")
    if version:
        return version

    # Fallback: try importlib.metadata if package is installed
    try:
        if sys.version_info >= (3, 8):
            from importlib.metadata import version as get_pkg_version

            return get_pkg_version("sglang-router")
    except Exception:
        pass

    # Final fallback
    return "0.2.2"


def _get_git_branch() -> Optional[str]:
    """Get current Git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_git_commit() -> Optional[str]:
    """Get current Git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_build_time() -> str:
    """Get build time or current time if not available."""
    # Try to get build time from Rust extension if available
    try:
        # If the Rust extension is available, it might have build-time info
        # For now, we'll use the current time as a proxy
        # In a real scenario, this could be embedded during build
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _get_git_status() -> Optional[str]:
    """Get Git repository status (clean/dirty)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            if result.stdout.strip():
                return "dirty"
            else:
                return "clean"
    except Exception:
        pass
    return None


def _get_python_version() -> str:
    """Get Python version."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    return f"{major}.{minor}.{micro}"


def _get_build_mode() -> str:
    """Get build mode (development or production)."""
    # For Python, we can't easily determine debug/release mode
    # Check if we're running from source or installed package
    try:
        import sglang_router

        if hasattr(sglang_router, "__file__"):
            # Check if running from development source
            return "development"
        else:
            return "production"
    except Exception:
        return "unknown"


def _get_platform() -> str:
    """Get platform information."""
    import platform
    return platform.platform()


def get_version_string() -> str:
    """Get formatted version information string with structured format."""
    project_name = _get_project_name()
    version = _get_version_from_pyproject()
    build_time = _get_build_time()
    git_branch = _get_git_branch() or "unknown"
    git_commit = _get_git_commit() or "unknown"
    git_status = _get_git_status() or "unknown"
    python_version = _get_python_version()
    build_mode = _get_build_mode()
    platform_info = _get_platform()

    # Try to get Rust extension version info if available
    rustc_version = "N/A (Rust extension not available)"
    cargo_version = "N/A"
    try:
        # If Rust extension is available, we could access version info
        # For now, just mark it as available
        import sglang_router  # noqa: F401, F811

        rustc_version = "Available (version info from extension)"
    except ImportError:
        pass

    return (
        f"{project_name} {version}\n\n"
        f"Build Information:\n"
        f"  Build Time: {build_time}\n"
        f"  Build Mode: {build_mode}\n"
        f"  Platform: {platform_info}\n\n"
        f"Version Control:\n"
        f"  Git Branch: {git_branch}\n"
        f"  Git Commit: {git_commit}\n"
        f"  Git Status: {git_status}\n\n"
        f"Runtime:\n"
        f"  Python: {python_version}\n"
        f"  Rustc: {rustc_version}\n"
        f"  Cargo: {cargo_version}"
    )


def get_version() -> str:
    """Get version number only."""
    return _get_version_from_pyproject()


def get_short_version_string() -> str:
    """Get short version information string."""
    project_name = _get_project_name()
    version = _get_version_from_pyproject()
    git_commit = _get_git_commit() or "unknown"
    return f"{project_name} version {version}, build {git_commit}"


__version__ = get_version()
