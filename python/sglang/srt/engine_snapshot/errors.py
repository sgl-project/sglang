# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Error hierarchy for initialized engine snapshots.

Every snapshot error derives from :class:`SnapshotError`. Validation,
artifact-integrity and path-security failures additionally derive from
``ValueError``; failures raised while a create/restore operation is running
derive from ``RuntimeError``. Callers may therefore catch the specific
snapshot type, or the broad builtin, interchangeably.
"""


class SnapshotError(Exception):
    """Base class for all engine snapshot errors."""


class SnapshotUsageError(SnapshotError, ValueError):
    """The snapshot request or configuration is not supported."""


class SnapshotSecurityError(SnapshotUsageError):
    """The artifact path violates the private-path security contract."""


class SnapshotCompatibilityError(SnapshotUsageError):
    """The artifact's identity or format does not match this runtime."""


class SnapshotRuntimeFailure(SnapshotError, RuntimeError):
    """A snapshot operation failed while running."""


def error_detail(error: BaseException) -> str:
    """Render an exception as ``TypeName: message``.

    Used where several failures are merged into one message, so a bare
    ``RuntimeError`` with no text still identifies itself.
    """
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
