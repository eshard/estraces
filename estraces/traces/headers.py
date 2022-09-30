import collections
from . import abstract_reader
import numpy as _np


class Headers(collections.abc.Mapping):
    """Provide a dict-like object of trace header set global metadata values."""

    def __init__(self, reader: abstract_reader.AbstractReader):
        if not isinstance(reader, abstract_reader.AbstractReader):
            raise TypeError(
                "reader must be a subclass of {f}".format(
                    f=abstract_reader.AbstractReader.__name__
                )
            )
        self._reader = reader
        self._keys = reader.headers_keys

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f'key must be a str, not {type(key)}.')
        if key not in self._keys:
            raise KeyError(f'Header with key {key} missing')
        return self._reader.fetch_header(key=key)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __eq__(self, other):
        if not isinstance(other, collections.abc.Mapping):
            raise NotImplementedError(f'Cant compare Headers type with {type(other)} type.')
        if sorted(list(other.keys())) != sorted(list(self.keys())):
            return False
        for k, v in self.items():
            if isinstance(v, _np.ndarray):
                if not _np.array_equal(v, other[k]):
                    return False
            elif v != other[k]:
                return False
        return True

    def __str__(self):
        return str(list(self._keys))

    def __repr__(self):
        return str(self)
