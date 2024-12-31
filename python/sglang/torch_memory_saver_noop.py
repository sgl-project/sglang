from contextlib import contextmanager


@contextmanager
def configure_subprocess():
    yield


class TorchMemorySaver:
    @contextmanager
    def region(self):
        yield

    def pause(self):
        pass

    def resume(self):
        pass
