# -*- coding: utf-8 -*-
from . import metadatas
from . import samples
from . import headers


class Trace:
    """Provide a consistent API to manipulate a trace samples data and metadatas.

    Attributes:
        samples (:class:`Samples`): 1 dimension samples data
        metadatas (:class:`Metadatas`): trace metadatas
        headers (:class:`Headers`): headers value of the trace set
    Note:
        All metadatas are available through the `metadatas` attributes and through a corresponding named property.

    Examples:
        Samples are available as 1 dimensional array-like :class:`Samples` with shape (size of trace,).
        It supports a subset of numpy.ndarray-like slicing - including advanced list-based slicing::

            trace.samples[2000:3000]
            trace.samples[[1, 100, 1000]] # get samples at 1, 100 and 1000 indexes

        Metadatas are available through a dict-like :class:`Metadatas` instance::

            metadatas = trace.metadatas
            metadatas['plaintext']

        Each metadata can be reached with its own property::

            trace.plaintext # is equivalent to trace.metadatas['plaintext']

        Headers are metadata value which are the same for all the traces of one given trace set.
        It is provided at the trace level through a dict-like object:

            trace.headers['key'] # equivalent to ths.headers['key'] where ths is the trace header set of the trace.

    """

    _is_initialized = False

    def __init__(self, trace_id, reader):
        if trace_id is None:
            raise AttributeError("trace_id can't be None.")
        self._id = trace_id
        self._reader = reader
        self._samples = None
        self._metadatas = None
        self._headers = None
        try:
            self.name = f'Trace n°{self._id}'
        except AttributeError:
            # Name is already a metadata of the ths, so we don't bother with it.
            pass
        try:
            self.id = self._id
        except AttributeError:
            # id is already a metadata of the ths, so we don't bother with it.
            pass
        self._is_initialized = True

    def __len__(self):
        return len(self.samples)

    @property
    def samples(self):
        if self._samples is None:
            self._samples = samples.Samples(reader=self._reader, trace_id=self._id)
        return self._samples

    @property
    def metadatas(self):
        if not self._metadatas:
            self._metadatas = metadatas.Metadatas(reader=self._reader, trace_id=self._id)
        return self._metadatas

    @property
    def headers(self):
        if not self._headers:
            self._headers = headers.Headers(reader=self._reader)
        return self._headers

    def __repr__(self):
        return str(self)

    def __str__(self):
        r = 'Trace:\n'
        r += f'{"Reader instance":.<17}: {self._reader}\n'
        r += f'{"Index in set":.<17}: {self._id}\n'
        r += f'{"Samples size":.<17}: {len(self)}\n'
        for k in self.metadatas.keys():
            r += f'{k:.<17}: {self.metadatas.get(k)}\n'
        return r

    def __getattr__(self, name):
        try:
            return self.metadatas[name]
        except KeyError:
            raise AttributeError(f'No attribute {name}.')

    def __setattr__(self, name, value):
        if name not in self.__dir__() and self._is_initialized:
            self.metadatas  # Metadata needs to be instantiated before adding a in-memory metadata.
            self._metadatas[name] = value
        super().__setattr__(name, value)
