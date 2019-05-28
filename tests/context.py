import os
import sys
import mock
import estraces  # noqa E402
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@contextmanager
def patch_open(func):

    def _open(*args, **kwargs):
        func(*args, *kwargs)
        mocked_file.stop()
        o = open(*args, **kwargs)
        mocked_file.start()
        return o

    mocked_file = mock.patch("builtins.open", _open)
    try:
        mocked_file.start()
        yield func
        mocked_file.stop()
    except Exception as e:
        mocked_file.stop()
        raise e


class Counter:
    def __init__(self):
        self.counts = {}

    def __call__(self, *args, **kwargs):
        if self.counts.get(args[0]):
            self.counts[args[0]] += 1
        else:
            self.counts[args[0]] = 1
