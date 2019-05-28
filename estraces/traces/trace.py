# -*- coding: utf-8 -*-
from . import metadatas
from . import samples


class Trace:
    """Provides a consistent API to manipulate a trace samples data and metadatas.

    Attributes:
        id (int): identifier of the trace
        name (str): name of the trace
        samples (:class:`Samples`): 1 dimension samples data
        metadatas (:class:`Metadatas`): trace metadatas

    Note:
        All metadatas are available through the `metadatas` attributes and through a corresponding named property.

    Examples:
        Samples are available as 1 dimensionnal array-like :class:`Samples` with shape (size of trace,).
        It supports a subset of numpy.ndarray-like slicing - including advanced list-based slicing::

            trace.samples[2000:3000]
            trace.samples[[1, 100, 1000]] # get samples at 1, 100 and 1000 indexes

        Metadatas are available through a dict-like :class:`Metadatas` instance::

            metadatas = trace.metadatas
            metadatas['plaintext']

        Each metadata can be reached with its own property::

            trace.plaintext # is equivalent to trace.metadatas['plaintext']

    """

    def __init__(self, trace_id, reader):
        if trace_id is None:
            raise AttributeError("trace_id can't be None.")
        self.id = trace_id
        self._reader = reader
        self._samples = None
        self._metadatas = None
        self.name = f'Trace n°{self.id}'

    def __len__(self):
        return len(self.samples)

    @property
    def samples(self):
        if self._samples is None:
            self._samples = samples.Samples(reader=self._reader, trace_id=self.id)
        return self._samples

    @property
    def metadatas(self):
        if not self._metadatas:
            self._metadatas = metadatas.Metadatas(reader=self._reader, trace_id=self.id)
        return self._metadatas

    def __repr__(self):
        return f'{self.__class__.__name__}(trace_id={self.id}, reader={self._reader})'

    def __str__(self):
        r = 'Trace\n'
        r += f'{"Id":.<17}: {self.id}\n'
        r += f'{"Samples size":.<17}: {len(self)}\n'
        for k in self.metadatas.keys():
            r += f'{k:.<17}: {self.metadatas.get(k)}\n'
        return r
