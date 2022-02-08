import numpy as _np

from estraces.traces.abstract_reader import AbstractReader as _AbstractReader
from estraces.traces.trace_header_set import build_trace_header_set as _build_trace_header_set


def read_ths_from_random(nb_traces: int,
                         trace_length: int,
                         interval=(-128, 128),
                         dtype='int8',
                         seed=None,
                         headers={},
                         **kwargs):
    """Allow user to a TraceHeaderFile for random trace reader.

    Attributes:
        nb_trace (int): Number of trace to randomly generate
        trace_length (int): Length of trace to generate
        interval : default value are (-128, 128)
        dtype (str): default value is 'int8'
        seed : seed to use for `_np.random.seed(seed)`
        **kwargs (ndarray): metadata as ndarray, for example, plaintext=(16, 'uint8')

    """
    return _build_trace_header_set(reader=RandomReader(nb_traces=nb_traces,
                                                       trace_length=trace_length,
                                                       interval=interval,
                                                       dtype=dtype,
                                                       seed=seed,
                                                       headers=headers,
                                                       **kwargs),
                                   name='Random reader')


"""RandomReader generate random trace with specific length.
"""


class RandomReader(_AbstractReader):
    """Allow user to generate randomized traces with specific length.

    Attributes:
        nb_trace (int): Number of trace to randomly generate
        trace_length (int): Length of trace to generate
        interval : default value are (-128, 128)
        dtype (str): default value is 'int8'
        seed : seed to use for `_np.random.seed(seed)`
        **kwargs (ndarray): metadata as ndarray, for example plaintext=(16, 'uint8')

    """

    def __init__(self,
                 nb_traces: int,
                 trace_length: int,
                 interval=(-128, 128),
                 dtype='int8',
                 seed=None,
                 headers={},
                 **kwargs
                 ):
        self.nb_traces = nb_traces
        self.trace_length = trace_length
        self.interval = interval
        self.dtype = _np.dtype(dtype)
        self._check_dtype_and_interval()
        _np.random.seed(seed)
        self._headers = headers
        self._metadatas = kwargs

    def _generate_data(self, shape, dtype):
        if dtype == _np.dtype('float64'):
            x, y = _np.random.rand(shape[0]) * 2 - 1, _np.random.rand(shape[1]) * 2 - 1
        else:
            x, y = _np.random.rand(shape[0]).astype('float32') * 2 - 1, _np.random.rand(shape[1]).astype('float32') * 2 - 1
        data = _np.outer(x, y)
        a, b = self.interval
        data = data * ((b - a) / 2) + (b - (b - a) / 2)
        return data.astype(dtype)

    def _check_dtype_and_interval(self):
        if self.interval[0] >= self.interval[1]:
            raise ValueError(f'Minimum must be lower than Maximum, but {self.interval[0]} >= {self.interval[1]}.')
        try:
            infos = _np.iinfo(self.dtype)
        except ValueError:
            infos = _np.finfo(self.dtype)
        if self.interval[0] < infos.min:
            raise ValueError(f'Given minimum value ({self.interval[0]}) is lower than the minimum of the type {self.dtype}: {infos.min}.')
        if self.interval[1] - 1 > infos.max:
            raise ValueError(f'Given maximum value ({self.interval[1]}) is greater than the maximum of the type {self.dtype}: {infos.max}.')

    def _slice_length(self, frame, reference_length):
        start = frame.start if frame.start is not None else 0
        step = frame.step if frame.step is not None else 1
        stop = frame.stop
        if stop is None or stop > reference_length:
            stop = reference_length
        return abs(stop - start) // abs(step)

    def _key_length(self, key, reference_length):
        if isinstance(key, (slice, range)):
            return self._slice_length(key, reference_length)
        if key is Ellipsis:
            return reference_length
        return len(key)

    def fetch_samples(self, traces=slice(None), frame=slice(None)):
        nb_traces = self._key_length(traces, self.nb_traces)
        trace_length = self._key_length(frame, self.trace_length)
        return self._generate_data((nb_traces, trace_length), self.dtype)

    def fetch_metadatas(self, key, trace_id=None):
        length, dtype = self._metadatas[key]
        nb_data = self.nb_traces if trace_id is None else 1
        data = self._generate_data((nb_data, length), dtype)
        if trace_id is not None:
            return data[:, 0]
        return data

    @property
    def metadatas_keys(self):
        return self._metadatas.keys()

    def __len__(self):
        return self.nb_traces

    def __getitem__(self, key):
        nb_traces = self._key_length(key, self.nb_traces)
        return RandomReader(nb_traces, self.trace_length, self.interval, self.dtype, None, self._headers, **self._metadatas)

    def get_trace_size(self, trace_id=None):
        return self.trace_length

    def fetch_header(self, key):
        return self._headers[key]

    @property
    def headers_keys(self):
        return self._headers.keys()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'Random reader with {len(self)} traces of {self.trace_length} length. Interval {self.interval}. Metadatas: {list(self.metadatas_keys)}'
