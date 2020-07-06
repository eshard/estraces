from .context import estraces # noqa
import pytest
import sqlite3
import numpy as np
from estraces.formats import sqlite_format
from .conftest import PLAINS, CIPHERS, DATAS, _HEADERS
import os
import timeit


test_path = 'tests/samples/test.db'


@pytest.fixture
def filename():
    try:
        conn = sqlite3.connect(test_path)
        create_samples = '''
        CREATE TABLE traces (
            trace_id INTEGER PRIMARY KEY NOT NULL,
            samples BLOB,
            plaintext BLOB,
            ciphertext BLOB
        )
        '''
        cur = conn.cursor()
        cur.execute(create_samples)
        create_headers = '''
        CREATE TABLE headers (
            id INTEGER PRIMARY KEY NOT NULL,
            tag VARCHAR,
            dtype VARCHAR,
            value BLOB
        )
        '''
        cur = conn.cursor()
        cur.execute(create_headers)

        insert = '''
        INSERT INTO traces(trace_id, samples, plaintext, ciphertext) VALUES (?, ?, ?, ?)
        '''
        for i in range(len(DATAS)):
            cur = conn.cursor()
            cur.execute(insert, (i, DATAS[i], PLAINS[i], CIPHERS[i]))

        insert = 'INSERT INTO headers(id, tag, value, dtype) VALUES (?, ?, ?, ?)'
        i = 1
        for k, v in _HEADERS.items():
            cur = conn.cursor()
            cur.execute(insert, (i, k, v, str(v.dtype) if isinstance(v, np.ndarray) else 'str'))
            i += 1
        conn.commit()
        yield test_path
    finally:
        if conn:
            conn.close()
        os.remove(test_path)


def test_ths_without_headers(filename):
    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=sqlite_format.SamplesConfig(),
        metadata_config=sqlite_format.MetadataConfig(metadata_defs={'plaintext': 'uint8'})
    )
    assert isinstance(ths, estraces.TraceHeaderSet)
    assert len(ths) == 10
    assert [] == list(ths.headers.keys())
    assert not ths._reader.fetch_header('tag')


def test_ths_without_metadata(filename):
    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=sqlite_format.SamplesConfig()
    )
    assert isinstance(ths, estraces.TraceHeaderSet)
    assert len(ths) == 10
    assert [] == list(ths.metadatas.keys())
    assert not ths._reader.fetch_metadatas('plain')


def test_raises_exception_if_file_does_not_exist():
    with pytest.raises(ValueError):
        estraces.read_ths_from_sqlite('ouch', samples_config=sqlite_format.SamplesConfig())


def test_raises_exception_if_file_is_not_a_sqlite_db():
    with pytest.raises(ValueError):
        estraces.read_ths_from_sqlite('tests/samples/test.ets', samples_config=sqlite_format.SamplesConfig())


def test_raises_exception_fetching_non_existing_header(filename):
    samples_config = sqlite_format.SamplesConfig(table_name='traces', pk_column_name='trace_id', dtype='uint8', samples_column_name='samples')
    metadata_config = sqlite_format.MetadataConfig(
        table_name='traces',
        pk_column_name='trace_id',
        metadata_defs={
            'plaintext': 'uint8',
            'ciphertext': 'uint8',
        }
    )
    headers_config = sqlite_format.HeadersConfig(table_name='headers', tag_column_name='tag', value_column_name='value', dtype_column_name='dtype')

    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )
    with pytest.raises(KeyError):
        ths.headers['woop']


def test_raises_exception_with_wrong_header_config(filename):
    samples_config = sqlite_format.SamplesConfig(table_name='traces', pk_column_name='trace_id', dtype='uint8', samples_column_name='samples')
    metadata_config = sqlite_format.MetadataConfig()
    headers_config = sqlite_format.HeadersConfig(table_name='headers', tag_column_name='tags', value_column_name='value', dtype_column_name='dtype')

    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )
    with pytest.raises(sqlite_format.SQLiteError):
        ths.headers['key']

    with pytest.raises(sqlite_format.SQLiteError):
        ths._reader.fetch_header('key')


def test_raises_exception_fetching_non_existing_metadata(filename):
    samples_config = sqlite_format.SamplesConfig(table_name='traces', pk_column_name='trace_id', dtype='uint8', samples_column_name='samples')
    metadata_config = sqlite_format.MetadataConfig(
        table_name='traces',
        pk_column_name='trace_id',
        metadata_defs={
            'plaintext': 'uint8',
            'ciphertext': 'uint8',
        }
    )
    headers_config = sqlite_format.HeadersConfig()

    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )
    with pytest.raises(KeyError):
        ths.metadatas['woop']


def test_raises_exception_configuring_non_existing_metadata(filename):
    samples_config = sqlite_format.SamplesConfig(table_name='traces', pk_column_name='trace_id', dtype='uint8', samples_column_name='samples')
    metadata_config = sqlite_format.MetadataConfig(
        table_name='traces',
        pk_column_name='trace_id',
        metadata_defs={
            'plaintext': 'uint8',
            'ciphertext': 'uint8',
            'woop': 'uint8'
        }
    )
    headers_config = sqlite_format.HeadersConfig()

    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )
    with pytest.raises(sqlite_format.SQLiteError):
        ths[0].metadatas['woop']
    with pytest.raises(sqlite_format.SQLiteError):
        ths.metadatas['woop']


def test_raises_exception_configuring_non_existing_samples_col(filename):
    samples_config = sqlite_format.SamplesConfig(
        table_name='traces',
        pk_column_name='trace_id',
        dtype='uint8',
        samples_column_name='sampless'
    )
    metadata_config = sqlite_format.MetadataConfig()
    headers_config = sqlite_format.HeadersConfig()

    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )
    with pytest.raises(sqlite_format.SQLiteError):
        ths[0].samples[:]
    with pytest.raises(sqlite_format.SQLiteError):
        ths.samples[:]
    with pytest.raises(sqlite_format.SQLiteError):
        ths._reader.get_trace_size(0)

    samples_config = sqlite_format.SamplesConfig(
        table_name='tracess',
        pk_column_name='trace_id',
        dtype='uint8',
        samples_column_name='samples'
    )

    ths = estraces.read_ths_from_sqlite(
        filename,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )
    with pytest.raises(sqlite_format.SQLiteError):
        ths[0].samples[:]
    with pytest.raises(sqlite_format.SQLiteError):
        ths.samples[:]
    with pytest.raises(sqlite_format.SQLiteError):
        ths._reader.get_trace_size(0)


@pytest.fixture
def lots_of_traces():
    try:
        n_traces = 10000
        l_traces = 200
        conn = sqlite3.connect('lots_of_traces.db')
        create_samples = '''
        CREATE TABLE traces (
            trace_id INTEGER PRIMARY KEY NOT NULL,
            samples BLOB,
            plaintext BLOB
        )
        '''
        cur = conn.cursor()
        cur.execute(create_samples)
        create_headers = '''
        CREATE TABLE headers (
            id INTEGER PRIMARY KEY NOT NULL,
            tag VARCHAR,
            dtype VARCHAR,
            value BLOB
        )
        '''
        cur = conn.cursor()
        cur.execute(create_headers)

        insert = '''
        INSERT INTO traces(trace_id, samples, plaintext) VALUES (?, ?, ?)
        '''
        datas = np.random.randint(0, 255, (n_traces, l_traces), dtype='uint8')
        plains = np.random.randint(0, 255, (n_traces, 16), dtype='uint8')

        for i in range(len(datas)):
            cur = conn.cursor()
            cur.execute(insert, (i, datas[i], plains[i]))
        conn.commit()
        yield 'lots_of_traces.db'
    finally:
        if conn:
            conn.close()
        os.remove('lots_of_traces.db')


def test_samples_shape_is_not_too_slow(lots_of_traces):

    ths = estraces.read_ths_from_sqlite(
        lots_of_traces,
        samples_config=sqlite_format.SamplesConfig(),
        metadata_config=sqlite_format.MetadataConfig(metadata_defs={'plaintext': 'uint8'})
    )
    assert isinstance(ths, estraces.TraceHeaderSet)
    stm = 's_ths.samples.shape'
    s_ths = ths[:100]
    t1 = timeit.timeit(stm, globals={'s_ths': s_ths}, number=10)
    s_ths = ths[:10]
    t2 = timeit.timeit(stm, globals={'s_ths': s_ths}, number=10)
    assert t1 / t2 < 10
