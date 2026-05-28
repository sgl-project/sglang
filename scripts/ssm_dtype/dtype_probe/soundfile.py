"""Stub that masks a missing libsndfile / soundfile install in the smoke/eval
environment. Always present on the probe PYTHONPATH; calls raise immediately
so importing modules that *only conditionally* need soundfile keep working.
"""


class LibsndfileError(Exception):
    pass


def read(*args, **kwargs):
    raise LibsndfileError("soundfile is not installed in this smoke/eval environment")


def info(*args, **kwargs):
    raise LibsndfileError("soundfile is not installed in this smoke/eval environment")
