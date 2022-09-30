from functools import lru_cache

from ..traces.abstract_reader import AbstractReader
from ..traces.trace_header_set import build_trace_header_set
import estraces

import numpy as _np


def read_ths_from_multiple_ths(*args, **kwargs):
    """Build and return a :class:`TraceHeaderSet` instance as the concatenation of all input THSs.

    Args:
        args: some `TraceHeaderSet` objects.
        kwargs: some `TraceHeaderSet` objects.

    You can pass a headers dict argument to override inconsistent headers between trace header sets or add custom values.

    Note:
        shape and metadata must be consistent between each ths.

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    return build_trace_header_set(
        reader=ConcatFormatReader(*args, **kwargs),
        name='Concat Format THS'
    )


class ConcatFormatReader(AbstractReader):

    def __init__(self, *args, **kwargs):

        headers = kwargs.pop('headers', {})
        check = kwargs.pop('check', True)
        all_args = dict(**{f'ths_{i}': arg for i, arg in enumerate(args)}, **kwargs)

        if check:
            self._check_are_trace_header_sets(all_args)
            self._check_metadata_consistency(all_args)
            self._check_sample_shapes_and_dtype_consistency(all_args)
            self._check_metadata_shapes_and_dtype_consistency(all_args)
            self._headers = self._check_headers(all_args, headers)
        else:
            self._headers = headers

        self.ths_list = list(all_args.values())
        self._ths_dict = all_args
        self._trace_size = self.ths_list[0]._reader.get_trace_size(0)
        self._trace_dtype = self.ths_list[0].samples[0].dtype
        self._metas_infos = self._get_metadatas_infos()

        self._sizes = [len(ths) for ths in self.ths_list]
        self._size = _np.sum(self._sizes)

        self._sub_traceset_indices = _cached_arange(self._size)

    @staticmethod
    def _check_are_trace_header_sets(thss_dict):
        for key in thss_dict.keys():
            if not isinstance(thss_dict[key], estraces.TraceHeaderSet):
                raise TypeError(f'{key} argument must be a `TraceHeaderSet` object, not `{type(thss_dict[key])}`.')

    @staticmethod
    def _check_metadata_consistency(thss_dict):
        key0 = list(thss_dict.keys())[0]
        set0 = set(thss_dict[key0]._reader.metadatas_keys)
        for key in thss_dict.keys():
            set1 = set(thss_dict[key]._reader.metadatas_keys)
            if set0 != set1:
                raise ValueError(f'Inconsistent metadata keys between: {key0}, {set0} and {key}, {set1}')

    @staticmethod
    def _check_headers(thss_dict, headers={}):
        ths_0 = list(thss_dict.values())[0]
        set_0 = set(ths_0.headers.keys())
        for key, ths in thss_dict.items():
            if ths == ths_0:
                continue
            set_1 = set(ths.headers.keys())
            if set_0 != set_1:
                diff = set_0.difference(set_1)
                for key in diff:
                    if key not in headers:
                        raise ValueError(
                            f'Inconsistent headers, {key} missing from one ths.\
                            You should override headers values by passing a headers dict to the constructor.'
                        )
            for k in set_0:
                if k not in headers:
                    val_1 = ths.headers[k]
                    val_2 = ths_0.headers[k]
                    equal = _np.array_equal(val_1, val_2) if isinstance(val_1, _np.ndarray) else val_1 == val_2
                    if not equal:
                        raise ValueError(
                            f'Inconsistent headers values between {ths_0} and {ths} for header {k}.\
                            You should override headers values by passing a headers dict to the constructor.'
                        )
                    headers[k] = val_1
        return headers

    @staticmethod
    def _check_sample_shapes_and_dtype_consistency(thss_dict):
        key0 = list(thss_dict.keys())[0]
        ths0 = thss_dict[key0]
        for key in thss_dict.keys():
            ths1 = thss_dict[key]
            if ths0._reader.get_trace_size(0) != ths1._reader.get_trace_size(0):
                raise ValueError(f'Inconsistent samples shapes: samples length is {ths0._reader.get_trace_size(0)} for {key0} and samples length '
                                 f'is {ths1._reader.get_trace_size(0)} for {key}')
            if ths0[0].samples[0].dtype != ths1[0].samples[0].dtype:
                raise ValueError(f'Inconsistent samples dtypes: samples dtype is {ths0[0].samples[0].dtype} for {key0} and samples dtype '
                                 f'is {ths1[0].samples[0].dtype} for {key}')

    @staticmethod
    def _check_metadata_shapes_and_dtype_consistency(thss_dict):
        key0 = list(thss_dict.keys())[0]
        ths0 = thss_dict[key0]
        metas0 = list(ths0.metadatas)
        for key in thss_dict.keys():
            ths1 = thss_dict[key]
            for meta in metas0:
                meta0 = ths0[0].metadatas[meta]
                meta1 = ths1[0].metadatas[meta]
                dtype0 = ConcatFormatReader._get_meta_dtype(meta0)
                dtype1 = ConcatFormatReader._get_meta_dtype(meta1)
                if dtype0.kind != dtype1.kind:
                    raise ValueError(f'Inconsistent {meta} dtypes: {meta} dtype is {dtype0} for {key0} and {meta} dtype '
                                     f'is {dtype1} for {key}')
                if dtype0.kind != 'U':
                    if len(meta0) != len(meta1):
                        raise ValueError(f'Inconsistent {meta} shapes: {meta} length is {len(meta0)} for {key0} and {meta} length '
                                         f'is {len(meta1)} for {key}')

    @staticmethod
    def _get_meta_dtype(meta):
        try:
            dtype = meta.dtype
            if dtype.kind == 'O':
                meta = meta.flat[0]
                raise AttributeError
            return meta.dtype
        except AttributeError:
            if not isinstance(meta, str):
                raise TypeError(f'Metadata type not supported: {type(meta)}')
            return _np.dtype(f'<U{len(meta)}')

    def _get_metadatas_infos(self):
        ths = self.ths_list[0]
        metadatas_infos = dict()
        for meta in ths._reader.metadatas_keys:
            meta0 = ths[0].metadatas[meta]
            dtype = self._get_meta_dtype(meta0)
            length = 0 if dtype.kind == 'U' else len(meta0)
            metadatas_infos.update({f'{meta}': dict(length=length, dtype=dtype)})
        return metadatas_infos

    def _convert_traces_indices_to_list_of_raw_traces_indices_per_set(self, traces_indices):
        original_traces_indices = self._sub_traceset_indices[traces_indices]
        cumsizes = _np.cumsum(self._sizes)
        sets = _np.searchsorted(cumsizes, original_traces_indices, side='right')
        indices_per_set = [_np.where(sets == i)[0] for i in range(len(self._sizes))]
        original_traces_indices_per_set = [original_traces_indices[indices] - offset for indices, offset in zip(indices_per_set, _np.concatenate(([0], cumsizes)))] # noqa
        return original_traces_indices_per_set, indices_per_set, len(original_traces_indices)

    def fetch_metadatas(self, key, trace_id):
        if trace_id is None:
            trace_id = slice(len(self))
        is_int_trace_id = False
        if isinstance(trace_id, int):
            trace_id = [trace_id]
            is_int_trace_id = True

        original_traces_indices_per_set, indices_per_set, length = self._convert_traces_indices_to_list_of_raw_traces_indices_per_set(trace_id)
        shape = (length, self._metas_infos[key]['length'])
        shape = (shape[0], ) if shape[1] == 0 else shape
        rez = _np.empty(shape=shape, dtype=self._metas_infos[key]['dtype'])
        for indices, result_indices, ths in zip(original_traces_indices_per_set, indices_per_set, self.ths_list):
            if len(indices):
                data = ths.metadatas[key][indices]
                data_dtype = self._get_meta_dtype(data)
                if data_dtype.itemsize > rez.dtype.itemsize:
                    rez = rez.astype(data_dtype)
                rez[result_indices] = data
        if is_int_trace_id:
            rez = rez[0]
        return rez

    def fetch_samples(self, traces, frame):
        if frame is None:
            frame = slice(None)
        if traces is None:
            traces = slice(len(self))
        if isinstance(traces, int):
            traces = slice(traces, traces + 1)
        if isinstance(frame, int):
            if frame < 0:
                frame = slice(frame, None)
            else:
                frame = slice(frame, frame + 1)

        original_traces_indices_per_set, indices_per_set, length = self._convert_traces_indices_to_list_of_raw_traces_indices_per_set(traces)
        rez = _np.empty(shape=(length, self._trace_size), dtype=self._trace_dtype)
        rez = rez[:, frame]
        for indices, result_indices, ths in zip(original_traces_indices_per_set, indices_per_set, self.ths_list):
            if len(indices):
                data = ths.samples[indices, frame]
                rez[result_indices] = data
        return rez

    def __getitem__(self, key):
        super().__getitem__(key)

        if isinstance(key, int):
            key = [key]

        new_sub_traceset_indices = self._sub_traceset_indices[key]
        new_reader = ConcatFormatReader(check=False, headers=self._headers, **self._ths_dict)
        new_reader._sub_traceset_indices = new_sub_traceset_indices
        new_reader._size = len(new_sub_traceset_indices)

        return new_reader

    @property
    def metadatas_keys(self):
        return self.ths_list[0]._reader.metadatas_keys

    @property
    def headers_keys(self):
        return self._headers.keys()

    def fetch_header(self, key):
        return self._headers[key]

    def get_trace_size(self, trace_id):
        return self._trace_size

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Concat format reader of {self.ths_list} with {len(self)} traces.'


# See https://gitlab.com/eshard/estraces/-/issues/18
@lru_cache(maxsize=8)
def _cached_arange(s):
    return _np.arange(s)
