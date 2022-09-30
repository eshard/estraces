from ..traces.abstract_reader import AbstractReader
from ..traces.trace_header_set import build_trace_header_set
import os as _os
import numpy as _np
import typing
import sqlite3
from typing import NamedTuple, Dict


class SQLiteError(Exception):
    pass


class SamplesConfig(NamedTuple):
    table_name: str = 'traces'
    pk_column_name: str = 'trace_id'
    samples_column_name: str = 'samples'
    dtype: str = 'uint8'


class MetadataConfig(NamedTuple):
    metadata_defs: Dict = {}
    table_name: str = 'traces'
    pk_column_name: str = 'trace_id'


class HeadersConfig(NamedTuple):
    table_name: str = 'headers'
    pk_column_name: str = 'id'
    tag_column_name: str = 'tag'
    value_column_name: str = 'value'
    dtype_column_name: str = 'dtype'


EmptyMetadata = MetadataConfig(table_name=None)
EmptyHeaders = HeadersConfig(table_name=None)


def read_ths_from_sqlite(
    filename,
    samples_config: SamplesConfig,
    metadata_config: MetadataConfig = EmptyMetadata,
    headers_config: HeadersConfig = EmptyHeaders
):
    """Build and return a :class:`TraceHeaderSet` instance read from a SQLite3 database file.

    Args:
        filename (str or Path): filename of the SQLite3 database.
        samples_config (SamplesConfig): instance of `SamplesConfig` with parameters to read samples - table name, pk column name, samples column name and dtype
            format of samples data.
        metadata_config (MetadataConfig): instance of `MetadataConfig` with parameters to read trace metadata - table name, pk column name and dict where each
            entry is the column name of a metadata and the corresponding value the dtype of the metadata.
        headers_config (HeadersConfig): instance of `HeadersConfig` with parameters to read headers from a specific table - table name, tag column name,
            value column name and dtype column name.

    Returns:
        (:obj:`TraceHeaderSet`)

    """
    return build_trace_header_set(
        reader=SQLiteFormatReader(
            filename,
            samples_config=samples_config,
            metadata_config=metadata_config,
            headers_config=headers_config
        ),
        name='SQLite reader'
    )


class SQLiteFormatReader(AbstractReader):

    def __init__(self, filename: str, samples_config: SamplesConfig, metadata_config: MetadataConfig, headers_config: HeadersConfig):
        if not _os.path.exists(filename):
            raise ValueError(f'SQLite database file {filename} does not exists.')
        self._connection = sqlite3.connect(filename)
        try:
            self._connection.cursor().execute('select name from sqlite_master')
        except sqlite3.DatabaseError:
            raise ValueError(f'{filename} is not a SQLite3 database.')
        self._filename = filename
        self._samples_config = samples_config
        self._metadata_config = metadata_config
        self._headers_config = headers_config
        self.dtype = samples_config.dtype
        self._sub_traceset_indices = None
        self._headers_keys = None

    def __len__(self):
        if self._size is None:
            query = f'SELECT {self._samples_config.pk_column_name} FROM {self._samples_config.table_name}'
            data = _execute_query(self._connection, query, (), error_message='Problem with samples configuration: ')
            self._size = len(data)
        return self._size

    def fetch_samples(self, traces: list, frame=None) -> _np.ndarray:
        """Fetch samples for the given traces id and given samples data frame.

        Args:
            traces: Lists of traces id to fetch.
            frame: Samples data to fetch. Must support `Ellipsis`, `slice`, `list`, `ndarray` or `int` types.

        Returns:
            (:class:`numpy.ndarray`) array of shape (number of traces, size of samples)

        """
        if isinstance(traces, range) and (traces.start < 0 or traces.stop < 0):
            traces = range(self._size + traces.start, self._size + traces.stop, traces.step)

        traces = [i if i >= 0 else self._size + i for i in traces]
        traces = self._convert_traces_indices_to_db_indices(traces)

        ordered_unique_ids = sorted(set(traces))
        mask_needed = ordered_unique_ids != traces.tolist()

        ids = ','.join([str(t) for t in traces])

        query = f'''
        SELECT {self._samples_config.pk_column_name}, {self._samples_config.samples_column_name}
        FROM {self._samples_config.table_name} WHERE {self._samples_config.pk_column_name} in ({ids})
        '''
        data = _execute_query(self._connection, query, (), error_message='Error with samples configuration: ')
        res = _np.array([_np.frombuffer(d[1], dtype=self.dtype) for d in data])

        if mask_needed:
            mask = _np.zeros((len(traces), len(res)), dtype='uint8')
            for i in range(len(traces)):
                mask[i, ordered_unique_ids.index(traces[i])] = 1
            res = mask.dot(res)

        if isinstance(frame, int):
            frame = slice(frame, frame + 1) if frame >= 0 else slice(res[0].shape[0] + frame, res[0].shape[0] + 1 + frame)
        if res.ndim > 1:
            res = res[:, frame]
        return res

    def fetch_metadatas(self, key: typing.Hashable, trace_id: int = None):
        """Fetch metadata value for the given metadata key and trace id.

        Args:
            key (typing.Hashable): Key of the metadata to fetch. Must be hashable.
            trace_id (int): Trace id for which to fetch the metadata.

        Returns:
            A container of all the values of the trace set for the given metadata if trace_id is None.
            Else, the value of the metadata for the given trace id.

        """
        if self._metadata_config is not EmptyMetadata:
            if trace_id is not None:
                trace_id = self._convert_trace_index_to_db_index(trace_id)
                query = f'SELECT {key} from {self._metadata_config.table_name} where {self._metadata_config.pk_column_name}={trace_id}'
                data = _execute_query(self._connection, query, (), error_message=f'Error with metadata {key}: ')[0][0]
                if isinstance(data, str):
                    return data
                return _np.frombuffer(data, dtype=self._metadata_config.metadata_defs[key])
            else:
                query = f'SELECT {key} FROM {self._metadata_config.table_name}'
                data = _execute_query(self._connection, query, (), error_message=f'Error with metadata {key}: ')
                res = []
                for t in data:
                    if isinstance(t[0], str):
                        res.append(t[0])
                    else:
                        res.append(_np.frombuffer(t[0], dtype=self._metadata_config.metadata_defs[key]))
                return _np.array(res)[self._traceset_indices]

    @property
    def _traceset_indices(self):
        return self._sub_traceset_indices if self._sub_traceset_indices is not None else slice(None, None, 1)

    def _convert_trace_index_to_db_index(self, trace_id):
        return self._convert_traces_indices_to_db_indices([trace_id])[0]

    def _convert_traces_indices_to_db_indices(self, traces):
        if self._sub_traceset_indices is not None:
            sub_max = len(self._sub_traceset_indices)
            traces_index = [t for t in traces if t < sub_max]
            return self._sub_traceset_indices[traces_index]
        return _np.array(traces)

    def __getitem__(self, key):
        """Return a new format instance limited to traces[key] subset.

        Args:
            key: slice or list of traces indexes to slice on.

        """
        super().__getitem__(key)
        if isinstance(key, int):
            key = [key]
        elif isinstance(key, slice):
            key = range(
                key.start if key.start is not None else 0,
                key.stop if key.stop is not None else len(self),
                key.step if key.step is not None else 1
            )
        new_reader = SQLiteFormatReader(
            filename=self._filename,
            metadata_config=self._metadata_config,
            samples_config=self._samples_config,
            headers_config=self._headers_config
        )
        sub_traceset_indices = self._convert_traces_indices_to_db_indices(traces=key)
        traces_number = len(sub_traceset_indices)
        new_reader._sub_traceset_indices = sub_traceset_indices
        new_reader._size = traces_number
        return new_reader

    def fetch_header(self, key: typing.Hashable):
        """Fetch header value for the given key.

        Args:
            key (typing.Hashable): key of the header to fetch.

        Returns:
            the header value.

        """
        if self._headers_config is not EmptyHeaders:
            query = f'''
            SELECT {self._headers_config.value_column_name}, {self._headers_config.dtype_column_name}
            FROM {self._headers_config.table_name} where {self._headers_config.tag_column_name}="{key}"
            '''
            data = _execute_query(self._connection, query, (), error_message='Problem with headers config: ')
            if isinstance(data[0][0], str):
                return data[0][0]
            else:
                return _np.frombuffer(data[0][0], dtype=data[0][1])

    @property
    def metadatas_keys(self):
        """Provide a list or views of the metadatas keys available."""
        if self._metadata_config is not EmptyMetadata:
            return self._metadata_config.metadata_defs.keys()
        return []

    def get_trace_size(self, trace_id):
        """Provide the size of trace trace_id."""
        query = f'SELECT {self._samples_config.samples_column_name} FROM {self._samples_config.table_name} WHERE {self._samples_config.pk_column_name}=?'
        data = _execute_query(self._connection, query, (trace_id, ), error_message='Problem with samples configuration: ')
        data = _np.frombuffer(data[0][0], dtype=self.dtype)
        return data.shape[0]

    @property
    def headers_keys(self):
        """Provide a list or view of the headers keys available."""
        if self._headers_keys is None and self._headers_config is not EmptyHeaders:
            query = f'SELECT {self._headers_config.tag_column_name} FROM {self._headers_config.table_name}'
            data = _execute_query(self._connection, query, (), 'Problem with headers config: ')
            self._headers_keys = [t[0] for t in data]
        elif self._headers_keys is None:
            self._headers_keys = []
        return self._headers_keys


def _execute_query(connection, query, query_args, error_message='Problem executing query:'):
    try:
        return connection.cursor().execute(query, query_args).fetchall()
    except sqlite3.OperationalError as e:
        raise SQLiteError(f'{error_message} {e}.')
