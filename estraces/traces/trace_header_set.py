# -*- coding: utf-8 -*-
from . import trace, samples, metadatas, _metaclass, headers
from . import abstract_reader
from .._legacy_formats.trace import TraceDeprecated
from .._legacy_formats.trace_header_set import TraceHeaderSetDeprecated
from collections import abc
import numpy as _np


class TracesContainer(abc.Collection):

    def __init__(self, trace_klass, reader, sub_slice=None):
        self._trace_klass = trace_klass
        self._reader = reader
        self._traces = {}
        self._sub_slice = sub_slice if sub_slice else slice(0, len(self._reader), 1)
        self._keys_list = range(len(self._reader))[self._sub_slice]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return TracesContainer(trace_klass=self._trace_klass, reader=self._reader, sub_slice=key)
        key = self._keys_list[key]
        if key not in self._traces:
            self._traces[key] = self._trace_klass(trace_id=key, reader=self._reader)
        return self._traces[key]

    def __len__(self):
        return len(self._keys_list)

    def __contains__(self, trace):
        return trace._id in self._keys_list

    def __iter__(self):
        for trc in range(len(self)):
            yield self[trc]


class TraceHeaderSet:
    """Provide a consistent API for manipulating samples data and metadatas from any kind of trace files.

    Attributes:
        name (str): name of traces set
        metadatas (:class:`Metadatas`): dict-like metadatas object
        headers (:class:`Headers`): mapping of headers for this trace set, ie one off value metadata.
        samples (:class:`Samples`): 2 dimensions samples object
        traces (list): list of :class:`Trace` instances

    Notes:
        Each metadata available in `metadatas` attribute can be reached through a property.

    Examples:
        With a :class:`TraceHeaderSet` instance you can manipulate underlying :class:`Traces`, samples data,
        metadatas, headers, slice, iterate or filter on your set.

        Iterate on :class:`TraceHeaderSet` instance::

            for trace in ths:
                trace # each item is a :class:`Trace` instance

        Slice is a new :class:`TraceHeaderSet` instance limited to the slice::

            new_ths = ths[:200]
            new_ths = ths[[1, 2, 100]] # list-based slice supported

        Get a :class:`Trace` instance with an integer::

            trace = ths[200] # get item returns a :class:`Trace` instance

        You can also get the traces iterable::

            ths.traces

        Samples are available as 2 dimensional array-like :class:`Samples` with shape (number of traces, size of traces).
        It supports a subset of numpy.ndarray-like slicing - including advanced list-based slicing::

            ths.samples[2000:3000] # slice on traces, keeping all samples
            ths.samples[:, 1000:2000] # slice on samples data, for all traces
            ths.samples[[1, 100, 1000], :] # get samples for traces with indexes 1, 100 and 100

        Metadatas are available through a dict-like :class:`Metadatas` instance::

            metadatas = ths.metadatas
            metadatas['plaintext']

        Each metadata can be reached with its own attribute::

            ths.plaintext # is equivalent to ths.metadatas['plaintext']

        Headers are metadata which have only one value for all the trace set. It is hold on a mapping, dict like object:

            ths.headers['key']

        Filter traces in your ths with any arbitrary filtering function.
        As an example, get a new TraceHeaderSet with traces which first samples value is even::

            filtered_ths = ths.filter(lambda trace: trace.samples[0] % 2 == 0)

    Warning:
        TraceHeaderSet instance must be obtained through the use of a factory function suited to the use of your files format.

    """

    _is_valid_trace_class = False
    _is_initialized = False

    def __init__(self, reader, name=""):
        """Must not be instantiated or initialized directly, only through :func:`build_trace_header_set`."""
        if not self.__class__._is_valid_trace_class:
            raise TypeError(
                "{c} cant be instantiate directly. Use {f} function.".format(
                    c=self.__class__.__name__, f=build_trace_header_set.__name__
                )
            )
        try:
            self.name = name
        except AttributeError:
            # Name is already a metadata of the ths, so we don't bother with it.
            pass

        self._reader = reader
        self._trace_klass = _metaclass.ClassWithMetadatas.__new__(
            cls=_metaclass.ClassWithMetadatas,
            name=f"Trace{reader.__class__.__name__}",
            bases=(trace.Trace, TraceDeprecated),
            namespace={},
            metadatas_keys=reader.metadatas_keys,
        )
        self._traces = None
        self._metadatas = None
        self._headers = None
        self._samples = None
        self._is_initialized = True

    def __str__(self):
        self_part = self[:3]
        r = f'Trace Header Set:\n{"Name":.<17}: {self.name}\n{"Reader":.<17}: {self._reader}\n'
        for k in self.metadatas.keys():
            r += f'{k:.<17}: {self_part.metadatas.get(k).dtype}\n'
        return r

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.traces)

    def __getattr__(self, name):
        try:
            return self.metadatas[name]
        except KeyError:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name not in self.__dir__() and self._is_initialized:
            self.metadatas  # Metadata needs to be instantiated before adding a in-memory metadata.
            self._metadatas[name] = value
        return super().__setattr__(name, value)

    def _transpose_slice(self, key):
        start = key.start
        stop = key.stop
        if start and start < 0:
            start = 0 if start < -len(self) else len(self) + start
        if stop and stop < 0:
            stop = 0 if stop < -len(self) else len(self) + stop
        elif stop and stop > len(self):
            stop = None
        return slice(start, stop, key.step)

    def __getitem__(self, key):
        """Implement slicing of :class:`TraceHeaderSet` instance.

        Args:
            key (int, slice, list): slice or item to return

        Returns:
            (:obj:`Trace`) if ``key`` is integer
            (:obj:`TraceHeaderSet`) new instance otherwise

        """
        if isinstance(key, (int, _np.integer)):
            if isinstance(key, _np.integer):
                key = key.item()
            trace = self.traces[key]
            if self._metadatas:
                trace._metadatas = self._metadatas._copy_with_cache(key=key)
            return trace
        elif isinstance(key, (slice, list, _np.ndarray)):
            if isinstance(key, _np.ndarray):
                if key.ndim > 1 or key.dtype.kind not in ['i', 'u']:
                    raise IndexError(f'Cannot slice TraceHeaderSet with numpy array of dim {key.ndim} and kind {key.dtype.kind}.')
            if not isinstance(key, slice) and max(key) > len(self):
                raise IndexError(f'Cannot index TraceHeaderSet of length {len(self)} with index {max(key)}.')

            if isinstance(key, slice):
                key = self._transpose_slice(key)

            new_ths = type(self).__new__(type(self))
            reader = self._reader[key]
            new_ths.__init__(name=self.name, reader=reader)
            if self._metadatas:
                new_ths._metadatas = self._metadatas._copy_with_cache(key=key, reader=reader)
            return new_ths
        else:
            if isinstance(key, tuple):
                raise IndexError('Too many indices for TraceHeaderSet.')
            raise IndexError(
                "Only integers, slices (':'), 1 dimension numpy array and lists ([1, 10, 5]) are valid indices for TraceHeaderSet."
            )

    def __iter__(self):
        for trc in self.traces:
            yield trc

    @property
    def traces(self):
        """Provide a list of :class:`Trace` instances from this trace header set."""
        if not self._traces:
            self._traces = TracesContainer(trace_klass=self._trace_klass, reader=self._reader)
        return self._traces

    @property
    def metadatas(self):
        """Provide a :class:`Metadatas` mapping-like object. Provides metadatas values for this trace header set."""
        if not self._metadatas:
            self._metadatas = metadatas.Metadatas(reader=self._reader)
        return self._metadatas

    @property
    def headers(self):
        if not self._headers:
            self._headers = headers.Headers(reader=self._reader)
        return self._headers

    @property
    def samples(self):
        """Provide a 2d :class:`Samples` instance ndarray-like of shape (number of traces, size of traces)."""
        if self._samples is None:
            self._samples = samples.Samples(reader=self._reader)
        return self._samples

    def split(self, part_size):
        """Return an iterable of :class:`TraceHeaderSet` slices of at least length ``part_size``.

        Args:
            part_size (int): number of traces per slice

        Returns:
            (:obj:`SplittedTraceHeaderSetIterable`) An iterable :class:`TraceHeaderSet` objects.

        """
        class SplittedTraceHeaderSetIterable:
            """Provide iterable around parts of equal size of a given :class:`TraceHeaderSet`."""

            def __init__(self, ths, part_size):
                self._ths = ths
                self._slices = [
                    slice(start * part_size, (start + 1) * part_size, 1)
                    for start in range(len(ths) // part_size)
                ]
                if len(ths) % part_size != 0:
                    self._slices.append(
                        slice(len(ths) // part_size * part_size, None, 1)
                    )

            def __iter__(self):
                for sl in self._slices:
                    yield self._ths[sl]

            def __getitem__(self, key):
                return self._ths[self._slices[key]]

            def __len__(self):
                return len(self._slices)

        if not isinstance(part_size, int) or part_size <= 0:
            raise TypeError("part_size must be a positive integer.")
        return SplittedTraceHeaderSetIterable(self, part_size)

    def filter(self, filter_function):
        """Build a new :class:`TraceHeaderSet` instance based on traces passing the ``filter_function`` condition.

        Args:
            filter_function (callable): function or lambda condition.
                Must take a trace as argument and return True or False.
        Returns:
            (:obj:`TraceHeaderSet`) a new instance with only selected traces.

        """
        keys_to_keep = [idx for idx, trc in enumerate(self) if filter_function(trc)]
        return self[keys_to_keep]

    def __reduce__(self):
        return build_trace_header_set, (self._reader, self.name)


def build_trace_header_set(reader: abstract_reader.AbstractReader, name: str) -> TraceHeaderSet:
    """Factory function which provides easy instantiation of :class:`TraceHeaderSet` instance.

    The function first creates a class, inheriting from :class:`TraceHeaderSet`,
    suited to the format instance provided, instantiate a trace header set and returns it.

    Warning:
        End user should not use directly this function, but factory functions suited to their concrete traces file format, provided by format developers.

    Args:
        reader: instance of a format reader class inheriting from :class:`AbstractReader`
        name: trace header set name

    Raises:
        TypeError: if ``reader`` does not implement :class:`AbstractReader`.

    Returns:
        :obj:`TraceHeaderSet`: trace header set instance.

    """
    if not isinstance(reader, abstract_reader.AbstractReader):
        raise TypeError(
            "reader must be a subclass of {f}".format(
                f=abstract_reader.AbstractReader.__name__
            )
        )
    ths_cls = _metaclass.ClassWithMetadatas.__new__(
        cls=_metaclass.ClassWithMetadatas,
        name=f"TraceHeaderSetFor{reader.__class__.__name__}",
        bases=(TraceHeaderSet, TraceHeaderSetDeprecated),
        namespace={},
        metadatas_keys=reader.metadatas_keys,
    )
    return ths_cls(name=name, reader=reader)
