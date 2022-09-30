import h5py as _h5
import pathlib
import numpy as _np

from ..traces.abstract_reader import AbstractReader
from ..traces.trace_header_set import build_trace_header_set
from . import frames


def read_ths_from_ets_file(filename):
    """Build and return a :class:`TraceHeaderSet` instance from an ETS file.

    Args:
        filename (str or Path): filename of the ETS file

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    return build_trace_header_set(
        reader=ETSFormatReader(filename=filename),
        name='ETS Format THS'
    )


_TRACES_KEY = 'traces'
_METADATAS_KEY = 'metadata'


def _cast_to_str(val):
    # h5py doesn't manage nicely strings values. We must cast these explicitly before returning to the user.
    if isinstance(val, _np.ndarray):
        if 'S' in str(val.dtype) or 'object' == str(val.dtype):
            try:
                return _np.array([r.decode() for r in val])
            except AttributeError:
                return val
    if isinstance(val, bytes):
        return val.decode()
    return val


class ETSFormatReader(AbstractReader):

    def __init__(self, filename):
        self._filename = str(filename)
        if not pathlib.Path(filename).exists():
            raise AttributeError(f'File {filename} does not exist.')
        if not _h5.h5f.is_hdf5(bytes(self._filename, 'ascii')):
            raise TypeError(f'File {self._filename} is not a valid ETS file.')

        self._file = None
        self._sub_traceset_indices = None
        self._size, self._trace_size = self._h5_file[_TRACES_KEY].shape

    @property
    def _metadatas(self):
        return self._h5_file[_METADATAS_KEY]

    def fetch_header(self, key):
        return self._h5_file[_METADATAS_KEY].attrs[key]

    def fetch_metadatas(self, key, trace_id):
        if trace_id is not None:
            trace_index = self._convert_trace_index_to_file_index(trace_id)
            res = self._metadatas[key][trace_index]
        else:
            if isinstance(self._traceset_indices, slice):
                res = self._metadatas[key][self._traceset_indices]
            else:
                res = _np.array([self._metadatas[key][i] for i in self._traceset_indices])
        return _cast_to_str(res)

    @property
    def _traceset_indices(self):
        if self._sub_traceset_indices is not None:
            if frames.is_array_equivalent_to_a_slice(self._sub_traceset_indices):
                return frames.build_equivalent_slice(array=self._sub_traceset_indices)
            return self._sub_traceset_indices
        return slice(None, None, 1)

    def _convert_trace_index_to_file_index(self, trace_id):
        return self._convert_traces_indices_to_file_indices_array([trace_id])[0]

    def _convert_traces_indices_to_file_indices_key(self, traces):
        indices = self._convert_traces_indices_to_file_indices_array(traces=traces)
        if frames.is_array_equivalent_to_a_slice(indices):
            return frames.build_equivalent_slice(array=indices)
        return indices

    def _convert_traces_indices_to_file_indices_array(self, traces):
        if self._sub_traceset_indices is not None:
            sub_max = len(self._sub_traceset_indices)
            traces_index = _np.array([t for t in traces if t < sub_max])
            return self._sub_traceset_indices[traces_index]
        return _np.array(traces)

    def fetch_samples(self, traces, frame):
        traces = self._convert_traces_indices_to_file_indices_key(traces)

        if isinstance(frame, int):
            frame = _np.array([frame])
        elif isinstance(frame, list):
            frame = _np.array(frame)
        if isinstance(frame, _np.ndarray) and frames.is_array_equivalent_to_a_slice(frame):
            frame = frames.build_equivalent_slice(frame)

        if isinstance(frame, (Ellipsis.__class__, slice)):
            return self._slice_traces(keys_tuple=(traces, frame))
        else:
            if len(frame) == 1 and frame[0] < 0:
                start = frame[0]
                stop = None
            else:
                start = _np.min(frame)
                stop = _np.max(frame) + 1
            frame = frame - start
            return self._slice_traces(keys_tuple=(traces, slice(start, stop)), final_frame=frame)
            # TODO: how to write a test case for the memory overhead optimization strategy below ?
            # ratio = (stop - start) / frame.size
            # if ratio > 100:
            #   sorted_deduped_frame, inversion_frame = np.unique(frame, return_inverse=True)
            #   return self._slice_traces(keys_tuple=(traces, sorted_deduped_frame), final_frame=inversion_frame)

    def _slice_traces(self, keys_tuple, final_frame=None):
        traces, frame_slice = keys_tuple
        if final_frame is not None:
            if isinstance(traces, slice):
                return self._h5_file[_TRACES_KEY][keys_tuple][:, final_frame]
            if len(traces) > 0:
                return _np.vstack([self._h5_file[_TRACES_KEY][t, frame_slice][final_frame] for t in traces])
        else:
            if isinstance(traces, slice):
                return self._h5_file[_TRACES_KEY][keys_tuple]
            if len(traces) > 0:
                return _np.vstack([self._h5_file[_TRACES_KEY][t, frame_slice] for t in traces])
        return _np.array([])

    def __getitem__(self, key):
        super().__getitem__(key)

        if isinstance(key, int):
            key = [key]
        elif isinstance(key, slice):
            key = range(
                key.start if key.start is not None else 0,
                key.stop if key.stop is not None else len(self),
                key.step if key.step is not None else 1
            )
        sub_traceset_indices = self._convert_traces_indices_to_file_indices_array(traces=key)
        traces_number = len(sub_traceset_indices)
        new_reader = ETSFormatReader(filename=self._filename)
        new_reader._file = self._file
        new_reader._sub_traceset_indices = sub_traceset_indices
        new_reader._size = traces_number
        return new_reader

    @property
    def metadatas_keys(self):
        # A keys only dict value is instantiated here because h5py doesn't manage properly all keys' types.
        return {k: None for k in self._h5_file[_METADATAS_KEY].keys()}.keys()

    @property
    def headers_keys(self):
        return {k: None for k in self._h5_file[_METADATAS_KEY].attrs.keys()}.keys()

    def get_trace_size(self, trace_id):
        return self._trace_size

    @property
    def _h5_file(self):
        if self._file is None:
            self._file = _h5.File(name=self._filename, mode='r', libver='latest', swmr=True)
        return self._file

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(filename={self._filename})'
        )

    def __str__(self):
        return f'ETS format reader of file {self._filename} with {len(self)} traces.'

    def __reduce__(self):
        def deserialize(state):
            new_obj = ETSFormatReader(state['_filename'])
            new_obj.__dict__.update(state)
            return new_obj

        state = self.__dict__.copy()
        del state['_file']
        return deserialize, (state, )
