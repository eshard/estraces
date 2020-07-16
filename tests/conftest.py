import pytest
from .context import estraces  # noqa
from estraces.formats import ets_format, bin_format, bin_extractor, trs_format, ram_format, concat_format, sqlite_format
from .format_test_implementation import Format as DumbFormat
import numpy as np
import h5py
import uuid
import os
import glob
import trsfile
import sqlite3

np.random.seed(987654321)

CIPHERS = np.array([np.random.randint(0, 256, 16, dtype="uint8") for i in range(10)])
PLAINS = np.array([np.random.randint(0, 256, 16, dtype="uint8") for i in range(10)])
INDICES = np.array(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'])
_IND = np.array(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'], dtype='S2')
DATAS = np.vstack([np.random.randint(0, 256, 1000, dtype="uint8") for i in range(10)])
KEY = np.random.randint(0, 255, (16,), dtype='uint8')
TIME = 'this is a time'
FOO = 'this is foo'

_HEADERS = {
    'key': KEY,
    'time': TIME,
    'foo': FOO
}


def ram_reader(request):
    return ram_format.RAMReader(samples=DATAS, headers=_HEADERS, ciphertext=CIPHERS, plaintext=PLAINS, plain_t=PLAINS, indices=INDICES)


def concat_reader(request):
    ets = estraces.traces.trace_header_set.build_trace_header_set(reader=ets_reader(request), name="None")
    binr = estraces.traces.trace_header_set.build_trace_header_set(reader=bin_format_reader(request), name="None")
    return concat_format.ConcatFormatReader(ets=ets[:6], binr=binr[6:])


def dumb_format(request):
    return DumbFormat(
        datas=DATAS,
        metadatas={"ciphertext": CIPHERS, "plaintext": PLAINS, "indices": INDICES, "plain_t": PLAINS},
        headers=_HEADERS
    )


PATH = 'tests/samples/'
FN = uuid.uuid4()
CIPHERS_FILE = f'{PATH}ciphers.txt'
PLAINS_FILE = f'{PATH}plains.txt'


def write_bin_file():
    for i, t in enumerate(DATAS):
        with open(f'{PATH}{INDICES[i]}{FN}.bin', mode='bw') as f:
            f.write(t.tobytes())
    ciphers = [bytes([i for i in c]).hex() for c in CIPHERS]
    plains = [bytes([i for i in c]).hex() for c in PLAINS]
    with open(CIPHERS_FILE, mode='w') as cf:
        cf.write("\n".join(ciphers))
    with open(PLAINS_FILE, mode='w') as pf:
        pf.write("\n".join(plains))


TRS_FILENAME = f'{PATH}{FN}.trs'


def write_trs_file():
    with trsfile.open(TRS_FILENAME, 'w', live_update=1, padding_mode=trsfile.TracePadding.AUTO) as trs:
        trs.extend([
            trsfile.Trace(
                trsfile.SampleCoding.SHORT,
                DATAS[i],
                data=bytes(CIPHERS[i].tolist() + PLAINS[i].tolist()) + INDICES[i].encode('utf8'),
                title=f'The {i}th trace'
            ) for i in range(len(DATAS))]
        )


def bin_format_reader(request):
    write_bin_file()
    files = sorted(glob.glob(f'{PATH}*{FN}.bin'))

    def _():
        for f in files:
            os.remove(f)
        os.remove(CIPHERS_FILE)
        os.remove(PLAINS_FILE)

    request.addfinalizer(_)

    cipherextract = bin_extractor.FilePatternExtractor(CIPHERS_FILE, r"([A-Fa-f0-9]{32})", num=0, unhexlify=True)
    plainextract = bin_extractor.FilePatternExtractor(PLAINS_FILE, r"([A-Fa-f0-9]{32})", num=0, unhexlify=True)
    indicesextract = bin_extractor.PatternExtractor(r"([0-9]{2})", num=0, unhexlify=False)

    return bin_format.BinFormat(
        filenames=files,
        headers=_HEADERS,
        offset=0,
        dtype='uint8',
        metadatas_parsers={
            'ciphertext': cipherextract,
            'plaintext': plainextract,
            'indices': indicesextract,
            'plain_t': plainextract,
        })


ETS_FILENAME = f'{PATH}{FN}.ets'


def write_ets_file():
    file = h5py.File(name=ETS_FILENAME, mode='w')
    file.create_group('metadata')
    file.create_dataset('traces', dtype='uint8', shape=(10, 1000))
    file['metadata'].create_dataset('ciphertext', dtype='uint8', shape=(10, 16))
    file['metadata'].create_dataset('plaintext', dtype='uint8', shape=(10, 16))
    file['metadata'].create_dataset('plain_t', dtype='uint8', shape=(10, 16))
    file['metadata'].create_dataset('indices', dtype='S2', shape=(10,))
    for k, v in _HEADERS.items():
        file['metadata'].attrs[k] = v
    for i in range(10):
        file['traces'][i] = DATAS[i]
        file['metadata']['ciphertext'][i] = CIPHERS[i]
        file['metadata']['plaintext'][i] = PLAINS[i]
        file['metadata']['plain_t'][i] = PLAINS[i]
        file['metadata']['indices'][i] = _IND[i]
    file.flush()
    file.close()


def trs_reader(request):
    write_trs_file()

    def _():
        os.remove(TRS_FILENAME)
    request.addfinalizer(_)
    return trs_format.TRSFormatReader(
        filename=TRS_FILENAME,
        custom_headers=_HEADERS,
        metadatas_parsers={
            'plaintext': lambda d: np.array(bytearray(d[16:32])),
            'ciphertext': lambda d: np.array(bytearray(d[0:16])),
            'plain_t': lambda d: np.array(bytearray(d[16:32])),
            'indices': lambda d: d[32:34].decode('utf8')
        },
        dtype='uint8'
    )


def ets_reader(request):
    write_ets_file()

    def _():
        os.remove(ETS_FILENAME)
    request.addfinalizer(_)
    return ets_format.ETSFormatReader(filename=ETS_FILENAME)


SQLITE_FILENAME = 'tests/samples/test.db'


def write_sqlite3():

    try:
        conn = sqlite3.connect(SQLITE_FILENAME)
        create_samples = '''
        CREATE TABLE traces (
            trace_id INTEGER PRIMARY KEY NOT NULL,
            samples BLOB,
            plaintext BLOB,
            plain_t BLOB,
            ciphertext BLOB,
            indices BLOB
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
        INSERT INTO traces(trace_id, samples, plaintext, ciphertext, indices, plain_t) VALUES (?, ?, ?, ?, ?, ?)
        '''
        for i in range(len(DATAS)):
            cur = conn.cursor()
            cur.execute(insert, (i, DATAS[i], PLAINS[i], CIPHERS[i], INDICES[i], PLAINS[i]))

        insert = 'INSERT INTO headers(id, tag, value, dtype) VALUES (?, ?, ?, ?)'
        i = 1
        for k, v in _HEADERS.items():
            cur = conn.cursor()
            cur.execute(insert, (i, k, v, str(v.dtype) if isinstance(v, np.ndarray) else 'str'))
            i += 1
        conn.commit()
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()


def sqlite_reader(request):
    write_sqlite3()

    def _():
        os.remove(SQLITE_FILENAME)
    request.addfinalizer(_)

    samples_config = sqlite_format.SamplesConfig(table_name='traces', pk_column_name='trace_id', dtype='uint8', samples_column_name='samples')
    metadata_config = sqlite_format.MetadataConfig(
        table_name='traces',
        pk_column_name='trace_id',
        metadata_defs={
            'plaintext': 'uint8',
            'plain_t': 'uint8',
            'ciphertext': 'uint8',
            'indices': 'U2',
        }
    )
    headers_config = sqlite_format.HeadersConfig(table_name='headers', tag_column_name='tag', value_column_name='value', dtype_column_name='dtype')
    return sqlite_format.SQLiteFormatReader(
        filename=SQLITE_FILENAME,
        samples_config=samples_config,
        metadata_config=metadata_config,
        headers_config=headers_config
    )


fixtures_formats = [dumb_format, ets_reader, bin_format_reader, ram_reader, concat_reader, trs_reader, sqlite_reader]


@pytest.fixture(params=fixtures_formats)
def fmt(request):
    return request.param(request)
