from ..traces import abstract_reader
from ..traces.trace_header_set import build_trace_header_set
import numpy as _np
import trsfile as _trsfile
try:
    from trsfile.traceparameter import ParameterType as _ParameterType
    TRS_V2 = True
except ModuleNotFoundError:
    TRS_V2 = False


def read_ths_from_trs_file(filename, metadatas_parsers, dtype=None):
    """Build and return a :class:`TraceHeaderSet` instance from TraceRiscureSet file.

    Args:
        filename (str): TRS filename
        metadatas_parsers (dict): dict with a key for each metadata of the trace set,
            with value a function or lambda taking `data` as input and returning the metadata value parsed from it.
            For TRS V2 files, the `metadata_parsers` argument can be None. In this case, it is built with the TRACE_PARAMETER_DEFINITIONS header.
        dtype (dtype, optional): a Numpy dtype that will override the TRS samples data type

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    return build_trace_header_set(
        reader=TRSFormatReader(
            filename=filename,
            metadatas_parsers=metadatas_parsers,
            dtype=dtype
        ),
        name="TrsFormat trace header set"
    )


class TRSFormatReader(abstract_reader.AbstractReader):

    def __init__(self, filename, metadatas_parsers, custom_headers={}, dtype=None):
        self._filename = filename
        self._sub_traceset_indices = None
        self._headers = {k: v for k, v in custom_headers.items()}
        self._custom_headers = self._headers
        try:
            with _trsfile.open(filename, 'r') as trs_file:
                self._internal_headers = trs_file.get_headers()
                self._headers.update(
                    {
                        f'{str(k).replace("Header.", "").lower()}':
                            v.value if isinstance(v, _trsfile.SampleCoding) else v for k, v in self._internal_headers.items()
                    }
                )
                self._size = self._internal_headers[_trsfile.Header.NUMBER_TRACES]
        except (FileNotFoundError, ValueError, NotImplementedError) as e:
            raise AttributeError(f"{filename} is not a valid TRS file or filename. Original exception was:{e}.")
        metadatas_parsers = metadatas_parsers if metadatas_parsers is not None else self._metadatas_parsers_from_trace_parameter_definitions()
        self._set_metadatas_parsers(metadatas_parsers)
        self.dtype = _np.dtype(dtype) if dtype else self._internal_headers[_trsfile.Header.SAMPLE_CODING].format

    def _metadatas_parsers_from_trace_parameter_definitions(self):
        if not TRS_V2:
            raise TypeError('Setting a TRS reader with `metadatas_parsers=None` allowed only for `python > 3.6` and `trsfile > 0.3.2`.')
        try:
            param_definitions = self._headers['trace_parameter_definitions']
        except KeyError:
            raise TypeError('The TRS file does not have a TRACE_PARAMETER_DEFINITIONS field. You must set the `metadatas_parsers` argument.')
        metadatas_parsers = {}
        for key, param_def in param_definitions.items():
            metadatas_parsers[key] = self._generate_lambda_from_parameter_definition(param_def)
        return metadatas_parsers

    @staticmethod
    def _generate_lambda_from_parameter_definition(param_def):
        offset = param_def.offset
        length = param_def.length
        if param_def.param_type == _ParameterType.STRING:
            return lambda x: _np.array([x[offset:offset + length].decode()])
        dtype = {_ParameterType.BOOL: _np.dtype('bool'), _ParameterType.BYTE: _np.uint8,
                 _ParameterType.DOUBLE: _np.float64, _ParameterType.FLOAT: _np.float32,
                 _ParameterType.INT: _np.int32, _ParameterType.LONG: _np.int64, _ParameterType.SHORT: _np.int16}[param_def.param_type]
        return lambda x: _np.frombuffer(x, dtype=dtype, offset=0, count=16)

    def _set_metadatas_parsers(self, metadatas_parsers):
        self._metadatas_parsers = metadatas_parsers
        try:
            for k in self.metadatas_keys:
                self.fetch_metadatas(key=k, trace_id=0)
        except TypeError as e:
            raise AttributeError(f'Metadatas parsers {metadatas_parsers} are not valid unary functions. Original exception was:{e}.')

    def fetch_samples(self, traces, frame=None):
        traces = self._convert_traces_indices_to_file_indices_array(traces)

        if isinstance(frame, int):
            frame = [frame]
        with _trsfile.open(self._filename, 'r') as trs_file:
            raw_traces = [trs_file[trace] for trace in traces]
            samples = _np.array([trace[frame] for trace in raw_traces], dtype=self.dtype)
        return samples

    def fetch_header(self, key):
        return self._headers[key]

    def fetch_metadatas(self, key, trace_id):
        if trace_id is not None:
            trace_index = self._convert_traces_indices_to_file_indices_array(trace_id).squeeze()
            with _trsfile.open(self._filename, 'r') as trs_file:
                raw_trace = trs_file[trace_index]
                # .data in <=0.3.2 version is not available in TRS V2 pacakge
                # We need to use .parameters.serialize() now
                try:
                    res = self._metadatas_parsers[key](raw_trace.parameters.serialize())
                except AttributeError:
                    res = self._metadatas_parsers[key](raw_trace.data)
        else:
            if self._sub_traceset_indices is not None:
                indices = [i for i in self._sub_traceset_indices]
            else:
                indices = [i for i in range(len(self))]

            with _trsfile.open(self._filename, 'r') as trs_file:
                try:
                    res = _np.array([self._metadatas_parsers[key](trs_file[i].parameters.serialize()) for i in indices])
                except AttributeError:
                    res = _np.array([self._metadatas_parsers[key](trs_file[i].data) for i in indices])
        return res

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
        new_reader = TRSFormatReader(
            filename=self._filename,
            metadatas_parsers=self._metadatas_parsers,
            custom_headers=self._custom_headers
        )
        new_reader._sub_traceset_indices = sub_traceset_indices
        new_reader._size = traces_number
        return new_reader

    @property
    def metadatas_keys(self):
        return self._metadatas_parsers.keys()

    @property
    def headers_keys(self):
        return self._headers.keys()

    def get_trace_size(self, trace_id):
        return self._internal_headers[_trsfile.Header.NUMBER_SAMPLES]

    def _convert_traces_indices_to_file_indices_array(self, traces):
        if isinstance(traces, int):
            traces = [traces]
        if self._sub_traceset_indices is not None:
            sub_max = len(self._sub_traceset_indices)
            traces_index = [t for t in traces if t < sub_max]
            return _np.array(self._sub_traceset_indices[traces_index])
        return _np.array(traces)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(filename={self._filename}, metadatas_parsers={self._metadatas_parsers}, dtype={self.dtype})'
        )

    def __str__(self):
        return f'TRS format reader with {self._filename}, contains {len(self)} traces.'
