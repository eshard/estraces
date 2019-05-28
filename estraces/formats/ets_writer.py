import os as _os
import warnings as _warn
import h5py as _h5py
import numpy as _np
from .ets_format import read_ths_from_ets_file


class ETSWriter:
    """Class used to write a eshard custom format trace set."""

    def __init__(self, filename, overwrite=False):
        self._overwrite = overwrite
        self._filename = filename
        if _os.path.isfile(self._filename):
            if self._overwrite:
                _warn.warn("File '%s' already exists. It will be reseted on the first write operation" % self._filename)
            else:
                _warn.warn("File '%s' already exists. If no indexes are specified, new data will be append on existing datasets" % self._filename)
        self._isinit = False

    def __close_all__(self):
        try:
            f = _h5py.File(self._filename)
            to_close = _h5py.h5f.get_obj_ids(f.id)
        except Exception:
            to_close = []
        for fid in to_close:
            tmp = _h5py.File(fid)
            tmp.flush()
            tmp.close()
        try:
            f.close()
        except Exception:
            pass

    def __init_file__(self):
        self.__close_all__()
        if self._overwrite:
            _os.remove(self._filename)
            self._h5_file = _h5py.File(self._filename, mode='w', libver='latest')
        else:
            self._h5_file = _h5py.File(self._filename, mode='a', libver='latest')
        try:
            self._h5_file.swmr_mode = True
        except Exception:
            pass
        if 'metadata' not in list(self._h5_file.keys()):
            self._h5_file.create_group('metadata')
        self._initialized_datasets = list(self._h5_file['metadata'].keys())
        self._is_init_traces = 'traces' in list(self._h5_file.keys())
        self._isinit = True

    def __del__(self):
        self.close()

    @staticmethod
    def __np_to_str__(data):
        if data.shape == ():
            data = str(data)
        else:
            data = str(data)
            data = data[1:-1]
        return data

    def write_trace_object_and_points(self, trace_object, points, index=None):
        try:
            writable_attributes = trace_object.__get_writable_attributes__()
        except Exception:
            writable_attributes = trace_object.__dict__.keys()
        for tag in writable_attributes:
            tmp = getattr(trace_object, tag)
            self.write_meta(tag=tag, metadata=tmp, index=index)
        self.write_points(points=points, index=index)

    def write_points(self, points, index=None):
        if not self._isinit:
            self.__init_file__()
        if not isinstance(points, _np.ndarray) or points.ndim != 1:
            raise Exception("'points' should be a 1D ndarray")
        if not self._is_init_traces:
            self.__init_traces__(points)
        self.__add_to_dataset__(self._h5_file['traces'], points, index)

    def write_meta(self, tag, metadata, index=None):
        if not self._isinit:
            self.__init_file__()
        if not isinstance(metadata, _np.ndarray):
            metadata = _np.array(metadata)
        metadata = metadata.squeeze()

        if tag not in self._initialized_datasets:
            self.__init_meta__(tag, metadata)
        self.__add_to_dataset__(self._h5_file['metadata'][tag], metadata, index)

    def __init_traces__(self, points):
        points = points.squeeze()
        try:
            trace_size = len(points)
        except TypeError:
            trace_size = 1
        dtype = points.dtype
        self._h5_file.create_dataset('traces', (0, trace_size), maxshape=(None, trace_size), dtype=dtype)
        self._is_init_traces = True

    def __init_meta__(self, tag, first_meta):
        if first_meta.dtype.kind not in ['b', 'i', 'u', 'f', 'c']:
            h5type = _h5py.special_dtype(vlen=str)
            self._h5_file['metadata'].create_dataset(tag, (0,), maxshape=(None, ), dtype=h5type)
            self._initialized_datasets.append(tag)
        else:
            try:
                meta_size = len(first_meta)
            except TypeError:
                meta_size = 1
            dtype = first_meta.dtype
            self._h5_file['metadata'].create_dataset(tag, (0, meta_size), maxshape=(None, meta_size), dtype=dtype)
            self._initialized_datasets.append(tag)

    def __add_to_dataset__(self, data_set, data, index):
        try:
            if data_set.dtype.kind != 'O':
                expected_size = data_set.shape[-1]
                try:
                    data_size = len(data)
                except TypeError:
                    data_size = 1
                if data_size == expected_size:
                    tmp_data = data
                else:
                    tmp_data = _np.empty(expected_size, dtype=data.dtype)
                    data = data[:expected_size]
                    tmp_data[:len(data)] = data
                    tmp_data[len(data):] = 0
            else:
                tmp_data = self.__np_to_str__(data)
            nb_data = data_set.shape[0]

            if index is None:
                index = nb_data
            if index < nb_data and not self._overwrite:
                raise Exception("An element already exists in '%s' dataset at index %d and overwriting is disabled" % (str(data_set.name), index))

            if index >= data_set.shape[0]:
                data_set.resize(index + 1, axis=0)
            data_set[index] = tmp_data
        except Exception as e:
            raise e
        finally:
            self._h5_file.flush()

    def get_reader(self):
        self.close()
        return read_ths_from_ets_file(self._filename)

    def close(self):
        try:
            self._h5_file.close()
        except Exception:
            pass
