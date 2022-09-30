import numpy as _np

from ..traces.abstract_reader import AbstractReader
from ..traces.trace_header_set import build_trace_header_set


def read_ths_from_ram(samples, **kwargs):
    """Build and return a :class:`TraceHeaderSet` instance from arrays in RAM memory.

    Args:
        samples (ndarray): traces samples
        **kwargs (ndarray): metadata as ndarray

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    return build_trace_header_set(
        reader=RAMReader(samples=samples, **kwargs),
        name='RAM Format THS'
    )


class RAMReader(AbstractReader):

    def __init__(self, samples, headers={}, **kwargs):
        self._test_nd_arrays(samples, kwargs)
        self._test_shapes_compatibility(samples, kwargs)

        self._samples = samples
        self._kwargs_dict = kwargs
        self._headers = {k: v for k, v in headers.items()}
        self._size, self._trace_size = samples.shape

    @staticmethod
    def _test_nd_arrays(samples, kwargs_dict):
        if not isinstance(samples, _np.ndarray):
            raise TypeError(f'`samples` argument must be a {_np.ndarray}')
        if samples.ndim != 2:
            raise TypeError('`samples` argument must be a 2-dimensions ndarray')
        for key in kwargs_dict.keys():
            if not isinstance(kwargs_dict[key], _np.ndarray):
                raise TypeError(f'`{key}` argument must be a {_np.ndarray}')

    @staticmethod
    def _test_shapes_compatibility(samples, kwargs_dict):
        samples_shape = samples.shape
        for key in kwargs_dict.keys():
            if kwargs_dict[key].shape[0] != samples_shape[0]:
                raise ValueError(f'Incompatible shapes: samples {samples_shape} and {key} {kwargs_dict[key].shape}')

    def fetch_metadatas(self, key, trace_id):
        if trace_id is not None:
            return self._kwargs_dict[key][trace_id]
        else:
            return self._kwargs_dict[key]

    def fetch_header(self, key):
        return self._headers[key]

    def fetch_samples(self, traces, frame):
        if isinstance(traces, int):
            traces = [traces]
        if isinstance(frame, int):
            frame = [frame]
        if frame is not None:
            return self._samples[traces][:, frame]
        else:
            return self._samples[traces]

    def __getitem__(self, key):
        super().__getitem__(key)
        if isinstance(key, int):
            key = [key]
        new_samples = self._samples[key]
        new_kwargs = {item[0]: item[1][key] for item in self._kwargs_dict.items()}
        new_reader = RAMReader(new_samples, headers=self._headers, **new_kwargs)
        return new_reader

    @property
    def metadatas_keys(self):
        """Provide a list or views of the metadatas keys available."""
        return self._kwargs_dict.keys()

    @property
    def headers_keys(self):
        """Provide a list or views of the headers keys available."""
        return self._headers.keys()

    def get_trace_size(self, trace_id):
        """Provide the size of trace trace_id."""
        return self._trace_size

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'RAM reader with {len(self)} traces. Samples shape: {self._samples.shape} and metadatas: {list(self._kwargs_dict.keys())}'
