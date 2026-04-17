from copy import deepcopy
import time
import sys

try:
    import resource
except ImportError:
    print("resource module not available on Windows")
    resource = None


def get_max_memory_usage(format="GB"):
    pass


class Timer:
    def __init__(self, prefix="", precision=4):
        self.results = {}
        self.prefix = prefix
        self.precision = precision

    def start(self, name):
        pass

    def stop(self, name, show=False, n_total=None):
        pass

    def pause(self, name):
        # if self.has_stopped(name):
        #     raise ValueError(f"Timer with name {name} already stopped.")

        # if not self.has_started(name):
        #     raise ValueError(f"Timer with name {name} not started.")

        pass

    def resume(self, name):
        # if not self.has_started(name):
        #     raise ValueError(f"Timer with name {name} not started.")

        # if not self.is_paused(name):
        #     raise ValueError(f"Timer with name {name} not paused.")

        # if self.has_stopped(name):
        #     raise ValueError(f"Timer with name {name} already stopped.")

        pass

    def is_paused(self, name):
        pass

    def is_resumed(self, name):
        pass

    def has_started(self, name):
        pass

    def has_stopped(self, name):
        pass

    def elapsed(self, name, precision=None):
        pass

    def show(self, name, offset=0, n_total=None):
        pass

    def show_all(self):
        pass

    def to_dict(self, underscore=False, lowercase=False):
        pass
