# -*- coding: utf-8 -*-
import os as _os
import glob as _glob
import re as _re
import numpy as _np
import enum as _enum
from . import frames
from . import bin_extractor
from ..traces.trace_header_set import build_trace_header_set
from ..traces.abstract_reader import AbstractReader

__all__ = [
    'get_sorted_filenames',
    'read_ths_from_bin_filenames_list',
    'read_ths_from_bin_filenames_pattern',
    'PaddingMode',
    'BinFormat'
]


def get_sorted_filenames(pattern):
    """Retrieve a filename list conforming to a given glob string pattern, sorted by digit chunk found in filename."""
    files = _glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No file found with pattern: '" + pattern + "'")
    files = sorted(
        files,
        key=lambda ll: [
            int(chunk) if chunk.isdigit() else 0 for chunk in _re.split("([0-9]+)", ll)
        ],
    )
    return files


class PaddingMode(_enum.Enum):
    """Define the padding mode used when traces of binary format have different length.

    Possible modes are:
        - NONE: no padding is applied, an exception is thrown when instantiating the trace set.
        - PAD: samples will be padded with zero to the length of the longer trace of the set.
        - TRUNCATE: samples will be truncated to the length of the smaller trace of the set.
    """

    NONE = 0
    PAD = 1
    TRUNCATE = 2


def read_ths_from_bin_filenames_pattern(filename_pattern, metadatas_parsers, dtype, headers={}, offset=0, padding_mode=PaddingMode.NONE):
    """Build and return a :class:`TraceHeaderSet` instance from binaries files following a filename pattern.

    Each file must contains samples data for one trace.

    Args:
        filename_pattern (str): binary files filename pattern, for example ``"samples/*.bin"``
        metadatas_parsers (dict): dict with a key for each metadata of the trace set, with value a :class:`bin_extractor.HeaderExtractor`,
            :class:`bin_extractor.PatternExtractor`, :class:`bin_extractor.FilePatternExtractor` or :class:`bin_extractor.DirectValue` instance.
        dtype (dtype): the Numpy samples data dtype
        headers (dict, default={}): dictionary containing headers values for this trace set.
        offset (int, default:0): use as an offset when reading samples in files.
        padding_mode (:class:`PaddingMode`, default: `PaddingMode.NONE`): choose how to handle different traces size in your trace list.
            Possible modes are NONE, PAD and TRUNCATE (see :class:`bin_format.PaddingMode`).

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    files_list = get_sorted_filenames(pattern=filename_pattern)
    return read_ths_from_bin_filenames_list(
        filenames_list=files_list,
        headers=headers,
        metadatas_parsers=metadatas_parsers,
        dtype=dtype,
        offset=offset,
        padding_mode=padding_mode
    )


def read_ths_from_bin_filenames_list(filenames_list, metadatas_parsers, dtype, headers={}, offset=0, padding_mode=PaddingMode.NONE):
    """Build and return a :class:`TraceHeaderSet` instance from binaries files listed.

    Each file must contains samples data for one trace.

    Args:
        filenames_list (list): binary files filenames list
        metadatas_parsers (dict): dict with a key for each metadata of the trace set, with value a :class:`bin_extractor.HeaderExtractor`,
            :class:`bin_extractor.PatternExtractor`, :class:`bin_extractor.FilePatternExtractor` or :class:`bin_extractor.DirectValue` instance.
        dtype (dtype): the Numpy samples data dtype
        headers (dict, default={}): dictionary containing headers values for this trace set.
        offset (int, default: 0): use as an offset when reading samples in files.
        padding_mode (:class:`PaddingMode`, default: `PaddingMode.NONE`): choose how to handle different traces size in your trace list.
            Possible values are NONE, PAD and TRUNCATE (see :class:`bin_format.PaddingMode`).

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    return build_trace_header_set(
        reader=BinFormat(
            filenames=filenames_list,
            headers=headers,
            offset=offset,
            dtype=dtype,
            metadatas_parsers=metadatas_parsers,
            padding_mode=padding_mode
        ),
        name="BinFormat trace header set",
    )


class _FExtractor:

    def __init__(self, filename, index_conversion):
        self._content = None
        self._filename = filename
        self._index_conversion = index_conversion

    def __call__(self, index):
        if self._content is None:
            with open(self._filename, 'r') as f:
                self._content = f.readlines()
        if len(self._content) == 1:
            return self._content[0]
        else:
            return self._content[self._index_conversion(index)]


class BinFormat(AbstractReader):

    def __init__(
        self, filenames, dtype, metadatas_parsers, headers={}, offset=0, padding_mode=PaddingMode.NONE, metadatas_indices_conversion_function=lambda i: i
    ):
        if not isinstance(filenames, list):
            raise TypeError("filenames must be a list of filenames strings.")

        self._metadatas_indices_conversion_function = metadatas_indices_conversion_function
        self._filenames = _np.array(filenames)
        self._raw_metas = metadatas_parsers
        self._dtype = _np.dtype(dtype)
        self._offset = offset
        self._set_padding_mode(padding_mode)
        self._headers = {k: v for k, v in headers.items()}
        self._initialize_metadatas(metadatas_parsers)

    def _set_padding_mode(self, padding_mode):
        if not isinstance(padding_mode, PaddingMode):
            raise AttributeError(f'padding_mode must be a PaddingMode enum instance, not {type(padding_mode)}.')
        self._padding_mode = padding_mode

        lens = _np.array([self._trace_file_size(i) for i in range(len(self))])
        if self._padding_mode == PaddingMode.NONE:
            try:
                self._trace_size = lens[0]
            except IndexError:
                self._trace_size = 0
            if _np.any(_np.diff(lens)):
                raise ValueError(
                    "Not all traces are of the same length, and you are using no padding mode of BinFormat. "
                    "You should either fix you trace set files or use TRUNCATE or PAD padding mode.")
        elif self._padding_mode == PaddingMode.TRUNCATE:
            self._trace_size = lens.min()
        elif self._padding_mode == PaddingMode.PAD:
            self._trace_size = lens.max()

    def _initialize_metadatas(self, raw_metas):
        self._metadatas_parsers = {}

        def _2(index):
            return self._filenames[index]

        for k, extractor in raw_metas.items():
            if isinstance(extractor, bin_extractor.FilePatternExtractor):
                p = bin_extractor.PatternExtractor(
                    pattern=extractor.pattern,
                    replace=extractor.replace,
                    num=extractor.num,
                    unhexlify=extractor.unhexlify
                )

                ex = _FExtractor(filename=extractor.filename, index_conversion=self._metadatas_indices_conversion_function)
                self._metadatas_parsers[k] = (p, ex)
            else:
                self._metadatas_parsers[k] = (extractor, _2)

    @property
    def headers_keys(self):
        return self._headers.keys()

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, key):
        super().__getitem__(key)
        if isinstance(key, (list, _np.ndarray)):
            filenames = [self._filenames[i] for i in key]
        elif isinstance(key, slice):
            filenames = self._filenames[key].tolist()

        indices_map = {
            new_index: self._filenames.tolist().index(filenames[new_index]) for new_index in range(len(filenames))
        }

        fmt = BinFormat(
            filenames=filenames,
            offset=self._offset,
            headers=self._headers,
            dtype=self._dtype,
            metadatas_parsers=self._raw_metas,
            metadatas_indices_conversion_function=lambda i: self._metadatas_indices_conversion_function(indices_map[i]),
            padding_mode=self._padding_mode
        )
        return fmt

    def _read_samples(self, filename, frame):
        if isinstance(frame, int):
            frame = _np.array([frame])
        elif isinstance(frame, list):
            frame = _np.array(frame)

        if isinstance(frame, _np.ndarray) and frames.is_array_equivalent_to_a_slice(frame):
            frame = frames.build_equivalent_slice(frame)

        readsize = self._trace_size * self._dtype.itemsize
        with open(filename, "rb") as trace_file:
            trace_file.seek(self._offset)
            samples = _np.frombuffer(trace_file.read(readsize), dtype=self._dtype)

        if self._padding_mode == PaddingMode.PAD and samples.shape[0] < self._trace_size:
            t_samples = _np.zeros((self._trace_size), dtype=self._dtype)
            t_samples[:samples.shape[0]] = samples
            return t_samples[frame].squeeze()

        return samples[frame].squeeze()

    def fetch_samples(self, traces, frame):
        if len(traces) == 0 or len(self._filenames) == 0:
            return _np.array([], dtype=self._dtype)
        return _np.vstack(
            [
                self._read_samples(filename=self._filenames[i], frame=frame)
                for i in traces
            ]
        )

    def fetch_metadatas(self, key, trace_id=None):
        if trace_id is not None:
            return _apply_extractor(
                extractor=self._metadatas_parsers[key][0],
                value_function=self._metadatas_parsers[key][1],
                index=trace_id
            )
        return _np.array(
            [
                _apply_extractor(
                    extractor=self._metadatas_parsers[key][0],
                    value_function=self._metadatas_parsers[key][1],
                    index=idx
                )
                for idx in range(len(self._filenames))
            ]
        )

    def fetch_header(self, key):
        return self._headers[key]

    def get_trace_size(self, trace_id):
        return self._trace_size

    def _trace_file_size(self, trace_id):
        file_size = _os.stat(self._filenames[trace_id]).st_size
        return int((file_size - self._offset) / self._dtype.itemsize)

    @property
    def metadatas_keys(self):
        return self._metadatas_parsers.keys()

    @property
    def _filename(self):
        return self._filenames

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(filenames={self._filenames}, dtype={self._dtype}, offset={self._offset}, metadatas_parsers={self._metadatas_parsers})'
        )

    def __str__(self):
        return f'Bin format reader with {len(self._filenames)} files, dtype {self._dtype}'


# TODO: refactor pattern extractor
def _apply_extractor(extractor, value_function, index):
    try:
        if extractor.unhexlify is True:
            return _np.frombuffer(extractor.get_text(value_function(index)), dtype=_np.uint8)
        return extractor.get_text(value_function(index))
    except AttributeError:
        return extractor
