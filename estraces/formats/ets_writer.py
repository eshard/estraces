import psutil as _ps
import os as _os
import h5py as _h5py
import numpy as _np
from .ets_format import read_ths_from_ets_file
from ..traces import samples as _samples, metadatas as _metadata, trace as _trace, trace_header_set as _ths
import logging
import warnings

logger = logging.getLogger(__name__)

METADATA_GROUP_KEY = 'metadata'
TRACES_DATASET_KEY = 'traces'


class ETSWriterError(Exception):
    pass


class ETSWriter:
    """Provide API to create an ETS (Eshard Trace Set) file."""

    def __init__(self, filename, overwrite=False):
        """Create an ETS file writer instance.

        Args:
            filename (str): path and filename to write the ETS.
            overwrite (bool, default=False): if True, any existing file with filename will be erased before writing data.

        """
        self._overwrite = overwrite
        self._filename = filename
        self._h5_file = None
        self._initialized_datasets = None
        if _os.path.isfile(self._filename):
            if self._overwrite:
                logger.warning(f'File {self._filename} already exists. It will be reset on the first write operation.')
            else:
                logger.info(f'File {self._filename} already exists. If no indexes are specified, new data will be append on existing datasets.')
        self._is_init = False

    def _close_all(self):
        try:
            f = _h5py.File(self._filename, 'r')
            to_close = _h5py.h5f.get_obj_ids(f.id)
            for fid in to_close:
                tmp = _h5py.File(fid, 'r')
                tmp.flush()
                tmp.close()
            f.close()
        except OSError as e:
            logger.error(f'Exception raised during init of h5f file: {e}.')

    def _init_file(self):
        self._close_all()
        if _os.path.isfile(self._filename) and self._overwrite:
            _os.remove(self._filename)
            self._h5_file = _h5py.File(self._filename, mode='w', libver='latest')
        else:
            self._h5_file = _h5py.File(self._filename, mode='a', libver='latest')

        self._h5_file.swmr_mode = True

        if METADATA_GROUP_KEY not in list(self._h5_file.keys()):
            self._h5_file.create_group(METADATA_GROUP_KEY)
        self._initialized_datasets = list(self._h5_file[METADATA_GROUP_KEY].keys())
        self._is_init_traces = TRACES_DATASET_KEY in list(self._h5_file.keys())
        self._is_init = True

    def __del__(self):
        self.close()

    @staticmethod
    def _np_to_str(data):
        s = data.shape
        data = str(data)
        if s != ():
            data = data[1:-1]
        return _np.array([data])

    def add_trace_header_set(self, trace_header_set: _ths.TraceHeaderSet):
        """Append all traces samples and metadata hold by trace_header_set at the end of the ETS.

        Args:
            trace_header_set (`TraceHeaderSet`): a trace header set instance.

        """
        if not isinstance(trace_header_set, _ths.TraceHeaderSet):
            raise TypeError(f'trace header set is not a TraceHeaderSet instance but {type(trace_header_set)}.')
        if self._initialized_datasets and self._initialized_datasets != list(trace_header_set.metadatas.keys()):
            raise ETSWriterError(
                f'trace header set instance has different metadata {trace_header_set.metadatas.keys()} than existing in file {self._initialized_datasets}.'
            )

        batch_size = _get_optimal_batch_size(trace_header_set)
        self.add_samples(trace_header_set.samples, batch_size=batch_size)
        self.add_metadata(trace_header_set.metadatas, batch_size=batch_size)
        self.write_headers(trace_header_set.headers)

    def add_trace(self, trace: _trace.Trace):
        """Append trace samples and metadata provided to the end of the ETS.

        Args:
            trace (`Trace`): a Trace instance.

        """
        if not isinstance(trace, _trace.Trace):
            raise TypeError(f'trace is not a Trace instance but {type(trace)}.')
        if self._initialized_datasets and self._initialized_datasets != list(trace.metadatas.keys()):
            raise ETSWriterError(f'trace instance has different metadata {trace.metadatas.keys()} than existing in file {self._initialized_datasets}.')
        self.add_samples(trace.samples)
        self.add_metadata(trace.metadatas)

    def add_samples(self, samples: _samples.Samples, batch_size=1):
        """Append provided samples to the end of ETS.

        Warning: when using directly this method, it is up to you to ensure consistency of indices between samples and metadata in
        your ETS file. Use of `add_trace` or `add_trace_header_set` methods is recommended.

        Args:
            samples (`Samples`): a samples instance.

        """
        if not isinstance(samples, _samples.Samples):
            raise TypeError(f'samples is not a Samples instance but {type(samples)}.')
        if samples.ndim > 1:
            for i in range(0, len(samples), batch_size):
                self.write_samples(samples[i: i + batch_size])
        else:
            self.write_samples(samples[:])

    def add_metadata(self, metadata: _metadata.Metadatas, batch_size=1):
        """Append provided metadata to the end of ETS.

        Warning: when using directly this method, it is up to you to ensure consistency of indices between samples and metadata in
        your ETS file. Use of `add_trace` or `add_trace_header_set` methods is recommended.

        Args:
            metadata (`Metadatas`): a metadata instance.

        """
        if not isinstance(metadata, _metadata.Metadatas):
            raise TypeError(f'metadata is not a Metadatas instance, not {type(metadata)}.')
        for key, val in metadata.items():
            if metadata.is_trace():
                self.write_metadata(key, val)
            else:
                for i in range(0, len(val), batch_size):
                    self.write_metadata(key, val[i: i + batch_size], scalar=val.ndim != 2)

    def write_trace_object_and_points(self, trace_object, points, index=None):
        """Write provided trace samples and metadata at specified index of the ETS.

        If index is None, data will be appended to the end of the ETS.

        Args:
            trace_object (`Trace`): a Trace instance.
            points (numpy.ndarray): samples to write.
            index (int, default=None): index at which to write.

        """
        for tag, value in trace_object.metadatas.items():
            self.write_metadata(tag, value, index=index)
        self.write_samples(points, index=index)

    def write_samples(self, samples_array, index=None):
        """Write provided trace samples at specified index of the ETS.

        If index is None, data will be appended to the end of the ETS.

        Warning: when using directly this method, it is up to you to ensure consistency of indices between samples and metadata in
        your ETS file.

        Args:
            samples_array (numpy.ndarray): samples to write.
            index (int, default=None): index at which to write.

        """
        if not self._is_init:
            self._init_file()
        if not self._is_init_traces:
            self._init_traces(samples_array)
        self._add_to_dataset(self._h5_file[TRACES_DATASET_KEY], samples_array, index)

    def write_points(self, points, index=None):
        warnings.warn('This method is deprecated and will be removed in a future version. Use write_samples instead.', DeprecationWarning)
        return self.write_samples(points, index=index)

    def write_headers(self, headers):
        if not self._is_init:
            self._init_file()
        self._h5_file[METADATA_GROUP_KEY].attrs.update(headers)

    def write_metadata(self, key, value, index=None, scalar=False):
        """Write provided metadata at specified index of the ETS.

        If index is None, data will be appended to the end of the ETS.

        Warning: when using directly this method, it is up to you to ensure consistency of indices between samples and metadata in
        your ETS file.

        Args:
            key (str): metadata key
            value (numpy.ndarray): metadata value to write.
            index (int, default=None): index at which to write.

        """
        if not self._is_init:
            self._init_file()
        if not isinstance(value, _np.ndarray):
            value = _np.array(value)
        if key not in self._initialized_datasets:
            self._init_metadata(key, value, scalar=scalar)
        self._add_to_dataset(self._h5_file[METADATA_GROUP_KEY][key], value, index, scalar=scalar)

    def write_meta(self, tag, metadata, index=None):
        warnings.warn('This method is deprecated and will be removed in a future version. Use write_samples instead.', DeprecationWarning)
        return self.write_metadata(tag, metadata, index=index)

    @staticmethod
    def _compute_shapes_and_htype(array):
        if array.ndim == 1:
            shape = (0, len(array))
        elif array.ndim == 2:
            shape = (0, array.shape[1])
        else:
            shape = (0, 1)
        return shape, (None, shape[1]), array.dtype

    def _init_traces(self, data):
        shape, maxshape, dtype = self._compute_shapes_and_htype(data)
        self._h5_file.create_dataset(TRACES_DATASET_KEY, shape, maxshape=maxshape, dtype=dtype)
        self._is_init_traces = True

    def _init_metadata(self, metadata_key, metadata, scalar=False):
        shape, maxshape, h5type = self._compute_shapes_and_htype(metadata)

        if metadata.dtype.kind not in ['b', 'i', 'u', 'f', 'c']:
            h5type = _h5py.special_dtype(vlen=str)
            shape = (0, )
            maxshape = (None,)
        elif scalar:
            shape = (0, )
            maxshape = (None, )

        self._h5_file[METADATA_GROUP_KEY].create_dataset(metadata_key, shape, maxshape=maxshape, dtype=h5type)
        self._initialized_datasets.append(metadata_key)

    def _add_to_dataset(self, data_set, data, index, scalar=False):
        if data.ndim == 1 and not scalar:
            _data = _np.empty((1, len(data)), dtype=data.dtype)
            _data[0, :] = data
        elif (scalar and data.ndim == 1) or data.ndim == 2:
            _data = data
        else:
            _data = _np.array([[data]])
        return self._write_to_data_set(data_set, _data, index)

    def _write_to_data_set(self, data_set, data, index):
        if index is None:
            index = data_set.shape[0]
        if index < data_set.shape[0] and not self._overwrite:
            raise ETSWriterError(
                f'An element already exists in {str(data_set.name)} dataset at index {index} and overwriting is disabled.')
        if index >= data_set.shape[0]:
            data_set.resize(index + data.shape[0], axis=0)

        if data_set.ndim == 2 and data.ndim == 2 and data.shape[1] != data_set.shape[1]:
            shape = (data.shape[0], data_set.shape[1])
            tmp_data = _np.zeros(shape, dtype=data.dtype)
            data = data[:, :data_set.shape[1]]
            tmp_data[:, :data.shape[1]] = data
            tmp_data[:, data.shape[1]:] = 0
            data = tmp_data

        data_set[index: index + data.shape[0]] = data
        self._h5_file.flush()

    def get_reader(self):
        """Returns a `TraceHeaderSet` instance from the current writer, and closes it."""
        self.close()
        return read_ths_from_ets_file(self._filename)

    def close(self):
        """Close the current instance ETS file."""
        if self._h5_file:
            self._h5_file.close()


def compress_ets(filename, out_filename):
    """Create a compressed version of an ETS file.

    Creates a new ETS file using compression options provided by h5.
    The gain of compression can highly vary, depending on the nature of the data, from 10% to 50%.
    The trade-off is a somewhat slower reading from the resulting file.

    Args:
        filename (str): filename of original uncompressed ETS file.
        out_filename (str): filename of output compressed ETS file.

    Raises:
        ValueError if a file named out_filename already exists.
        AttributeError if the original file filename doesn't exists.

    """
    if not isinstance(filename, str) or not isinstance(out_filename, str):
        raise TypeError(f'filename and out_filename must be str, not {type(filename)}, {type(out_filename)}.')

    if _os.path.exists(out_filename):
        raise ValueError(f'A file named {out_filename} already exists.')
    ths = read_ths_from_ets_file(filename)

    ets_compressed = _h5py.File(out_filename, mode='w', libver='latest')
    ets_compressed.create_dataset(
        TRACES_DATASET_KEY,
        shape=(len(ths), len(ths.samples[0])),
        dtype=ths.samples[0].dtype,
        compression='gzip',
        compression_opts=9
    )
    ets_compressed.create_group(METADATA_GROUP_KEY)

    batch_size = _get_optimal_batch_size(ths)

    for i in range(0, len(ths), batch_size):
        ets_compressed[TRACES_DATASET_KEY][i: i + batch_size] = ths[i: i + batch_size].samples[:]
        for k, v in ths[i: i + batch_size].metadatas.items():
            if v.dtype.kind not in ['b', 'i', 'u', 'f', 'c']:
                h5type = _h5py.special_dtype(vlen=str)
                shape = (len(ths), )
            else:
                h5type = v.dtype
                shape = (len(ths), v.shape[-1])

            if k not in ets_compressed[METADATA_GROUP_KEY]:
                ets_compressed[METADATA_GROUP_KEY].create_dataset(k, shape=shape, compression='gzip', compression_opts=9, dtype=h5type)
            ets_compressed[METADATA_GROUP_KEY][k][i: i + batch_size] = v

    ets_compressed.close()


def _get_optimal_batch_size(ths, available_memory_ratio=0.5):
    base_trace = ths[0]
    unit_size = base_trace.samples[:].nbytes
    for k, v in base_trace.metadatas.items():
        try:
            unit_size += v.nbytes
        except AttributeError:
            logger.info(f'Scalar metadata {k} not taken into account for optimal batch size computation.')
    available_mem = _ps.virtual_memory()[1]

    if unit_size > available_mem:
        raise MemoryError(f"Not enough memory to load one trace (Available memory : {available_mem} bytes)")
    return int(available_memory_ratio * available_mem / unit_size)
