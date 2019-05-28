# -*- coding: utf-8 -*-
from ..formats import ets_format as _ets_format
import warnings as _warnings


def _ets_reader(filename):
    _warnings.warn(
        'ETSReader is deprecated. Use read_ths_from_ets functions from estraces package.',
        DeprecationWarning
    )

    return _ets_format.read_ths_from_ets_file(filename=filename)


ETSReader = _ets_reader
