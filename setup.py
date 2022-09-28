#!/usr/bin/env python

import sys
import os

from setuptools import setup
from setuptools.command.test import test


class PyTest(test):
    user_options = [("pytest-args=", "a", "Arguments to pass into py.test")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import pytest
        import shlex

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, "estraces/__version__.py")) as f:
    exec(f.read(), about)

setup(
    version=about['__version__'],
    cmdclass={"test": PyTest}
)
