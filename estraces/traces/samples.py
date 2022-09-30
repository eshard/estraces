import numpy as _np
from . import abstract_reader


class Samples:
    """Provide a wrapper object around Numpy ndarray for traces samples data manipulation.

    A Samples instance can have 1 or 2 dimensions. It supports a reduced slicing and indexing API.
    ndarray API methods calls are proxied to the wrapped ndarray.

    Attributes:
        array (:class:`ndarray`): wrapped Numpy array.

    """

    SUPPORTED_INDICES_TYPES = (Ellipsis.__class__, slice, int, list, _np.ndarray, range)

    def __init__(self, reader: abstract_reader.AbstractReader, trace_id=None):
        if not isinstance(reader, abstract_reader.AbstractReader):
            raise TypeError(
                "reader must be a subclass of {f}".format(
                    f=abstract_reader.AbstractReader.__name__
                )
            )
        self._trace_id = trace_id
        self._reader = reader
        self._set_ndim()
        self._array = None

    def __getattr__(self, name):
        arr = self.array
        if hasattr(arr, name):
            return getattr(arr, name)
        else:
            raise AttributeError(f'Attribute {name} does not exists on Samples.')

    def __repr__(self):
        return repr(self.array)

    def __str__(self):
        return repr(self)

    def __len__(self):
        if self.ndim == 1:
            return self._reader.get_trace_size(self._trace_id)
        else:
            return len(self._reader)

    def _set_ndim(self):
        self.ndim = 1 if self._trace_id is not None else 2 * (len(self._reader) > 0)

    def __getitem__(self, key):
        self._check_indices_type(key)
        traces, frame = self._get_traces_and_frame_from_key(key)
        self._check_traces_dimension(traces_key=traces)
        self._check_frame_dimension(frame=frame)
        if isinstance(traces, int):
            return Samples(reader=self._reader, trace_id=traces)[frame]
        res = self._reader.fetch_samples(traces=traces, frame=frame)
        return self._shaped_result(result=res, traces=traces, frame=frame)

    def _shaped_result(self, result, traces, frame):
        if self.ndim == 1:
            result = result[0]
        if isinstance(frame, int):
            if result.shape[0] == 1:
                return result[0]
            else:
                return result.squeeze()
        else:
            return result

    def _check_frame_dimension(self, frame):
        if isinstance(frame, int):
            frame = [frame]
        if isinstance(frame, range) and len(frame) == 0:
            raise IndexError('indexing with length 0 range not allowed.')
        if isinstance(frame, list):
            for f in frame:
                if f >= self._samples_size:
                    raise IndexError(f'index {f} is out of bounds for axis {self._samples_axis} with size {self._samples_size}')

    @property
    def _samples_axis(self):
        return 0 if self.ndim < 2 else 1

    def _check_indices_type(self, indices):
        inds = indices if isinstance(indices, tuple) else (indices, )
        if isinstance(indices, tuple) and len(indices) > self.ndim:
            raise IndexError(f'Samples support at most {self.ndim} dimensions. Perhaps try using samples this way: trace.samples[:]')
        for index in inds:
            self._check_index_type(index=index)

    def _check_index_type(self, index):
        if not isinstance(index, self.SUPPORTED_INDICES_TYPES):
            raise IndexError(
                "only {types} are valid indices for Samples".format(
                    types=", ".join([str(t) for t in self.SUPPORTED_INDICES_TYPES]),
                )
            )

    def _get_traces_and_frame_from_key(self, key):
        if self.ndim == 1:
            if isinstance(key, _np.ndarray) and key.dtype == 'bool':
                key = _np.argwhere(key).squeeze().tolist()
            return [self._trace_id], key
        if not isinstance(key, tuple):
            return self._convert_traces_key_to_iterable(key), ...
        else:
            if isinstance(key[1], _np.ndarray) and key[1].dtype == 'bool':
                key = (key[0], _np.argwhere(key[1]).squeeze())
            return self._convert_traces_key_to_iterable(key[0]), key[1]

    def _convert_traces_key_to_iterable(self, traces_key):
        if isinstance(traces_key, (None.__class__, Ellipsis.__class__)):
            return range(len(self._reader))
        elif isinstance(traces_key, (slice, range)):
            stop = traces_key.stop if traces_key.stop is not None and traces_key.stop < len(self._reader) else len(self._reader)
            return range(
                traces_key.start if traces_key.start else 0,
                stop,
                traces_key.step if traces_key.step else 1,
            )
        elif isinstance(traces_key, _np.ndarray) and traces_key.dtype == 'bool':
            return _np.argwhere(traces_key).squeeze().tolist()
        return traces_key

    def _check_traces_dimension(self, traces_key):
        if isinstance(traces_key, int):
            traces_key = [traces_key]
        if isinstance(traces_key, list):
            for t in traces_key:
                if t > len(self._reader) - 1:
                    raise IndexError(f'index {t} is out of bounds for axis 0 with size {len(self._reader)}')

    @property
    def _samples_size(self):
        if self.ndim == 0:
            return None
        elif self.ndim == 1:
            return len(self)
        else:
            return len(self[0])

    @property
    def shape(self):
        return (
            (0,),
            (self._samples_size, ),
            (len(self), self._samples_size)
        )[self.ndim]

    @property
    def dtype(self):
        if self._array is not None:
            return self._array.dtype
        return self[0].dtype

    @property
    def size(self):
        return _np.prod(self.shape)

    @property
    def array(self):
        # See issue https://gitlab.eshard.int/esdynamic/estraces/issues/35
        # Having these attributes ensure that numpy.array constructor correctly instantiate array given a Samples instance.
        if self._array is None:
            self._array = self[:]
        return self._array

    @property
    def __array_struct__(self):
        return self.self.array.__array_struct__

    @property
    def __array_interface__(self):
        return self.array.__array_interface__
