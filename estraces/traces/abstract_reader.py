import abc
import typing
import numpy as _np

"""Abstract base class for concrete trace formats.


"""

FrameType = typing.Union[....__class__, int, slice, list, _np.ndarray]
MetaType = typing.TypeVar("M")
MetadataTypes = typing.Union[MetaType, typing.Container[MetaType]]
TraceId = typing.TypeVar("I")


class AbstractReader(abc.ABC):
    """Provides the basic interface any format reader for trace set must provides.

    The :class:`TraceHeaderSet` and :class:`Trace` abstraction rely on the use of concrete implementation of `AbstractReader`.
    Format implementation must provides fetching methods for samples and metadatas, and must implement `__getitem__`.
    All performances strategies (lazy loading, caching, memory or CPU usage) are thus delegated to the implementer of the concrete format.

    Methods:
        fetch_samples: given traces and samples frame, returns samples data
        fetch_metadatas: given a trace id and a metadata key, returns metadata values
        __getitem__: handles slicing of reader
        metadatas_keys: property - Returns a view or container of the metadatas available keys

    """

    _size = None

    def __len__(self):
        return self._size

    @abc.abstractmethod
    def fetch_samples(self, traces: list, frame=None) -> _np.ndarray:
        """Fetch samples for the given traces id and given samples data frame.

        Args:
            traces: Lists of traces id to fetch.
            frame: Samples data to fetch. Must support `Ellipsis`, `slice`, `list`, `ndarray` or `int` types.

        Returns:
            (:class:`numpy.ndarray`) array of shape (number of traces, size of samples)

        """
        pass

    @abc.abstractmethod
    def fetch_metadatas(self, key: typing.Hashable, trace_id: int = None) -> MetadataTypes:
        """Fetch metadata value for the given metadata key and trace id.

        Args:
            key (typing.Hashable): Key of the metadata to fetch. Must be hashable.
            trace_id (int): Trace id for which to fetch the metadata.

        Returns:
            A container of all the values of the trace set for the given metadata if trace_id is None.
            Else, the value of the metadata for the given trace id.

        """
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        """Returns a new format instance limited to traces[key] subset.

        Args:
            key: slice or list of traces indexes to slice on.

        """
        if not isinstance(key, (slice, list, _np.ndarray)):
            raise TypeError('Only slice, 1 dimension numpy array and lists are valid indices for types implementing AbstractReader')

    @abc.abstractmethod
    def fetch_header(self, key: typing.Hashable):
        """Fetch header value for the given key.

        Args:
            key (typing.Hashable): key of the header to fetch.

        Returns:
            the header value.

        """
        pass

    @property
    @abc.abstractmethod
    def metadatas_keys(self):
        """Provides a list or views of the metadatas keys available."""
        pass

    @abc.abstractmethod
    def get_trace_size(self, trace_id):
        """Provides the size of trace trace_id."""
        pass

    @property
    @abc.abstractmethod
    def headers_keys(self):
        """Provides a list or view of the headers keys available."""
        pass
