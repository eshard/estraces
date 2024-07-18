import numpy as _np
from estraces import ETSWriter as _ETSWriter


_SAMPLES_KEY = 'samples'


class _Dataset():
    def __init__(self, length):
        self.length = length
        self.data = None
        self.current_index = None

    def write(self, data):
        """Write data at current dataset index.

        Args:
            data (ndarray): Must be 1D.

        """
        if not isinstance(data, _np.ndarray):
            data = _np.array(data)
        if data.ndim > 1:
            raise ValueError(f'Data must be at most 1D, but {data.ndim}D found')
        if self.data is None:
            self._init(data)

        if self.current_index >= self.length:
            raise ValueError('Dataset is full.')

        if self.data.shape[1] != 1:
            self.data[self.current_index, :len(data)] = data
        else:
            self.data[self.current_index] = data
        self.current_index += 1

    def _init(self, data):
        if data.ndim:
            self.data = _np.zeros_like(data, shape=(self.length, len(data)))
        else:
            self.data = _np.zeros_like(data, shape=(self.length, 1))
        self.current_index = 0

    def _reset(self):
        self.data = None
        self.current_index = None

    def get_data(self):
        """Return data[:current_index + 1] and reset."""
        if self.current_index is None or self.current_index == 0:
            to_return = None
        else:
            to_return = self.data[:self.current_index]
        self._reset()
        return to_return


class BufferedETSWriter():
    """Extend ETSWriter capabilities by adding a bufferization feature."""

    def __init__(self, filename, overwrite=False, buffer_length=None):
        """Create a Buffered ETS file writer instance.

        Args:
            filename (str): Path and filename to write the ETS.
            overwrite (bool, default=False): If True, any existing file with filename will be erased before writing data.
            buffer_length (int): Length of the buffer, in number of traces. If None, the buffer size is computed accordingly
                to the trace size to be equal to 100MB.

        Once the buffer is full, all the data are write back into the ETS file and the buffer is reseted.

        Note that this class provide only "atomic" operations, where samples and metadata are write together, in order to garanty consistency
             between samples and metadata.

        """
        self.ets_writer = _ETSWriter(filename, overwrite=overwrite)
        self.buffer_length = buffer_length
        self.datasets = {}
        self.cnt = 0

    def write_headers(self, headers):
        self.ets_writer.write_headers(headers)

    def write_trace_object_and_points(self, trace_object, points, index=None):
        """Add provided trace samples and metadata to BufferedETS.

        Args:
            trace_object (`Trace`): A `Trace` instance.
            points (numpy.ndarray): Samples to write.
            index (int, default=None): Dummy argument, not taken into account, for compatibility with simple `ETSWriter` format.

        """
        if self.buffer_length is None:
            self.buffer_length = int(100e6 / _np.array(points).nbytes)
        self._write_to_dataset(_SAMPLES_KEY, points)
        for tag, value in trace_object.metadatas.items():
            self._write_to_dataset(tag, value)
        self.cnt += 1
        if self.cnt == self.buffer_length:
            self.flush()

    def flush(self):
        """Write the temporary data into the ETS file."""
        for name1, dataset1 in self.datasets.items():
            for name2, dataset2 in self.datasets.items():
                if dataset1.current_index != dataset2.current_index:
                    raise ValueError('All Datasets must have the same length, '
                                     f'but len({name1})={dataset1.current_index} and len({name2})={dataset2.current_index}')
        for name, dataset in self.datasets.items():
            data = dataset.get_data()
            if data is None:
                return
            if name == _SAMPLES_KEY:
                self.ets_writer.write_samples(data)
            else:
                if data.dtype.kind in {'U', 'S'}:
                    for d in data.squeeze():
                        self.ets_writer.write_metadata(name, str(d))
                else:
                    self.ets_writer.write_metadata(name, data)
        self.cnt = 0

    def _write_to_dataset(self, dataset_name, data):
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = _Dataset(self.buffer_length)
        self.datasets[dataset_name].write(data)

    def close(self):
        self.flush()
        self.ets_writer.close()

    def get_reader(self):
        self.flush()
        return self.ets_writer.get_reader()
