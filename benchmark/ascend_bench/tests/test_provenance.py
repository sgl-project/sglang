"""Provenance capture: env override and CANN layout tolerance."""

from asc_bench import provenance


def test_git_sha_env_override(monkeypatch):
    monkeypatch.setenv(provenance.GIT_SHA_ENV, "abc1234")
    assert provenance._git_sha() == "abc1234"


def test_cann_version_reads_ascend_toolkit_home(tmp_path, monkeypatch):
    toolkit = tmp_path / "cann-9.0.0"
    (toolkit / "aarch64-linux").mkdir(parents=True)
    (toolkit / "aarch64-linux" / "ascend_toolkit_install.info").write_text(
        "version=9.0.0\n", encoding="utf-8"
    )
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    assert provenance._cann_version() == "9.0.0"


def test_cann_version_without_toolkit_does_not_raise(monkeypatch):
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    # CANN-less host: may return None or a real version, but must not raise
    provenance._cann_version()
