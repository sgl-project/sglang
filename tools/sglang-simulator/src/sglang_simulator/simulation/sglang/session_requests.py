"""Classification of the SGLang request structs the session timeline cares about.

The io_struct types are resolved once, on construction, rather than per request:
the hook layer replaces `builtins.__build_class__` to intercept SGLang classes as
they are defined, so this module must not import `sglang.srt.*` at import time.
"""


class SessionRequestClassifier:
    """Answers what a raw IPC request is, and which session it belongs to."""

    def __init__(self):
        from sglang.srt.managers.io_struct import (
            BatchTokenizedGenerateReqInput,
            CloseSessionReqInput,
            TokenizedGenerateReqInput,
        )

        self.close_type = CloseSessionReqInput
        self.generate_type = TokenizedGenerateReqInput
        self.batch_type = BatchTokenizedGenerateReqInput

    def is_close(self, req) -> bool:
        return isinstance(req, self.close_type)

    def iter_generate_reqs(self, req):
        """Yield the single generate requests carried by `req`, if any.

        `BatchTokenizedGenerateReqInput` does not subclass the single-request
        type and holds none of its fields; it is a container of them.
        """
        if isinstance(req, self.generate_type):
            yield req
        elif isinstance(req, self.batch_type):
            yield from req.batch

    def carries_generate_req(self, req) -> bool:
        return isinstance(req, (self.generate_type, self.batch_type))

    def session_id_of(self, req) -> str | None:
        """Session a generate request belongs to, or None when unsessioned.

        Radix-native sessions carry the top-level `session_id`; the older
        session-controller path carries `session_params.id` (scheduler.py routes
        on both). A request never sets both -- `GenerateReqInput` rejects that.
        """
        if not isinstance(req, self.generate_type):
            return None
        if req.session_id is not None:
            return req.session_id
        if req.session_params is None:
            return None
        return req.session_params.id

    def session_ids_of(self, req) -> set[str]:
        """Sessions represented by `req`, unpacking a batch."""
        ids = {self.session_id_of(inner) for inner in self.iter_generate_reqs(req)}
        ids.discard(None)
        return ids
