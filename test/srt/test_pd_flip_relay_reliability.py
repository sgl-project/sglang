import ast
from pathlib import Path


SOURCE = Path("python/sglang/srt/managers/tokenizer_manager.py")


def _method(name: str) -> ast.AST:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_relay_post_is_fail_closed_and_sends_admin_bearer():
    source = ast.unparse(_method("_pd_flip_post_relay_output"))
    assert "getattr(self.server_args, 'admin_api_key', None)" in source
    assert "if not admin_api_key:" in source
    assert "'Authorization': f'Bearer {admin_api_key}'" in source


def test_relay_output_enters_outbox_and_retries_instead_of_local_fallback():
    source = ast.unparse(_method("_pd_flip_maybe_relay_output"))
    assert "pd_flip_output_relay_outbox" in source
    assert "await self._pd_flip_flush_relay_key(relay_key)" in source
    assert "self._pd_flip_ensure_relay_retry(relay_key)" in source
    assert source.rstrip().endswith("return True")


def test_relay_ack_advances_baseline_before_terminal_route_cleanup():
    source = ast.unparse(_method("_pd_flip_flush_relay_key_locked"))
    post = source.index("self._pd_flip_post_relay_output")
    success_gate = source.index("if not result.get('success')")
    baseline = source.index("self.pd_flip_output_relay_baseline[relay_key] = output_seq")
    terminal_cleanup = source.index("self.pd_flip_output_relay_targets.pop")
    assert post < success_gate < baseline < terminal_cleanup
    assert "for output_seq in sorted(outbox)" in source


def test_abort_cleanup_cancels_retry_tasks_and_drops_outbox():
    source = ast.unparse(_method("_pd_flip_clear_session_relay_state"))
    assert "pd_flip_output_relay_outbox" in source
    assert "pd_flip_output_relay_retry_tasks" in source
    assert "task.cancel()" in source
