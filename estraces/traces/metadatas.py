import collections
from . import abstract_reader


class Metadatas(collections.abc.Mapping):
    """Provides a dict-like object of traces metadatas.

    Each metadata value is either a value, if the `Metadata` instance wraps one trace,
    or an array of values, if the :class:`Metadata` wraps a trace set metadatas.

    """

    def __init__(self, reader: abstract_reader.AbstractReader, trace_id=None):
        if not isinstance(reader, abstract_reader.AbstractReader):
            raise TypeError(
                "reader must be a subclass of {f}".format(
                    f=abstract_reader.AbstractReader.__name__
                )
            )
        self._keys = reader.metadatas_keys
        self._trace_id = trace_id
        self._reader = reader
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError("Metadata with key {k} missing".format(k=key))
        if key not in self._cache:
            self._cache[key] = self._reader.fetch_metadatas(key=key, trace_id=self._trace_id)
        return self._cache[key]

    def __setitem__(self, key, value):
        self._cache[key] = value
        self._keys = list(self._keys) + [key]

    def is_trace(self):
        return self._trace_id is not None

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __repr__(self):
        return f'{self._keys}-{self._reader}'

    def __str__(self):
        return repr(self)

    def _copy_with_cache(self, key, reader=None):
        if not reader:
            m = Metadatas(reader=self._reader, trace_id=key)
        else:
            m = Metadatas(reader=reader, trace_id=None)
        m._keys = self._keys
        m._cache = {
            k: v[key] for k, v in self._cache.items()
        }
        return m
