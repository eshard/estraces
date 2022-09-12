from .context import estraces  # noqa: F401
from estraces import read_ths_from_trs_file
import pytest
import numpy as np
import cloudpickle
import trsfile


TRS_FILENAME = 'tests/samples/aes128_sb_ciph_0fec9ca47fb2f2fd4df14dcb93aa4967.trs'  # TRS V1 file

PLAINTEXTS_10 = np.array([[111, 111, 0, 202, 89, 18, 30, 159, 41, 95, 134, 24, 243, 193, 171, 27],
                          [212, 44, 106, 221, 40, 124, 241, 26, 4, 129, 240, 57, 201, 59, 120, 84],
                          [225, 198, 49, 234, 172, 146, 40, 214, 209, 83, 29, 53, 226, 63, 44, 201],
                          [31, 20, 136, 25, 124, 69, 70, 19, 253, 3, 130, 45, 143, 134, 119, 153],
                          [102, 189, 40, 64, 172, 191, 65, 96, 107, 114, 53, 0, 68, 122, 195, 106],
                          [38, 184, 241, 197, 204, 242, 97, 144, 177, 186, 87, 227, 10, 135, 74, 128],
                          [80, 42, 250, 44, 133, 32, 174, 44, 185, 160, 95, 20, 58, 186, 216, 105],
                          [186, 88, 211, 113, 164, 119, 106, 12, 116, 102, 82, 27, 32, 90, 6, 44],
                          [141, 194, 214, 51, 137, 21, 44, 34, 190, 116, 10, 153, 179, 135, 195, 191],
                          [105, 120, 11, 202, 136, 103, 43, 18, 215, 112, 151, 64, 92, 234, 159, 113]], dtype='uint8')


@pytest.fixture
def trs():
    return read_ths_from_trs_file(TRS_FILENAME, dtype='float32',
                                  metadatas_parsers=dict(plaintext=lambda x: np.frombuffer(x, dtype='uint8', offset=0, count=16),
                                                         ciphertext=lambda x: np.frombuffer(x, dtype='uint8', offset=16)))


def test_trs_file_raise_exception_if_filename_incorrect():
    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename='fakefilename', metadatas_parsers={})

    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename=12334, metadatas_parsers={})

    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename='tests/samples/test.ets', metadatas_parsers={})


def test_trs_reader_raise_exception_if_metadatas_parser_incorrect():
    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers=[1, 2])

    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers="ffgfgfgf")

    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={"meta1": "Yipi"})

    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={"meta1": 1})

    with pytest.raises(AttributeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={"meta1": lambda x, y: x + y})

    with pytest.raises(AttributeError):
        def _(x, y):
            x + y
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={"meta1": _})

    with pytest.raises(AttributeError):
        def _(x, y):
            x + y
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={"meta1": lambda x: "constant", "meta2": _})


def test_trs_file_raises_exception_if_dtype_is_incorrect():
    with pytest.raises(TypeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={}, dtype='toto')

    with pytest.raises(TypeError):
        read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={}, dtype='uuint8')


def test_trs_reader_optional_dtype_overrides_default():
    ths = read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={})
    assert ths.samples[0, :].dtype == 'int8'
    ths = read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={}, dtype='float32')
    assert ths.samples[0, :].dtype == 'float32'
    ths = read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={}, dtype=np.uint16)
    assert ths.samples[0, :].dtype == 'uint16'
    ths = read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={}, dtype='uint8')
    assert ths.samples[0, :].dtype == 'uint8'


def test_trs_headers_provides_native_file_format_headers():
    ths = read_ths_from_trs_file(filename=TRS_FILENAME, metadatas_parsers={})
    assert dict(ths.headers) == {'title_space': 0, 'sample_coding': 1, 'length_data': 32, 'number_samples': 1920, 'number_traces': 500, 'trace_block': None}


def test_serialization(trs):
    dumped = cloudpickle.dumps(trs)
    loaded_ths = cloudpickle.loads(dumped)
    assert np.array_equal(loaded_ths.samples[:], trs.samples[:])
    assert np.array_equal(loaded_ths.plaintext[:], trs.plaintext[:])
    assert np.array_equal(loaded_ths.ciphertext[:], trs.ciphertext[:])


def test_simple_indexing(trs):
    assert np.array_equal(trs.samples[:10], trs[:10].samples[:])
    assert np.array_equal(trs[:10].plaintext, PLAINTEXTS_10)
    assert np.array_equal(trs[[1, 3, 5]].plaintext, PLAINTEXTS_10[[1, 3, 5]])


def test_simple_indexing_by_integer(trs):
    trs.samples[3]
    assert np.array_equal(trs[3].plaintext, PLAINTEXTS_10[3])


def test_double_indexing_1(trs):
    sub_trs = trs[:20]
    assert np.array_equal(sub_trs.samples[:10], sub_trs[:10].samples[:])
    assert np.array_equal(sub_trs[:10].plaintext, PLAINTEXTS_10)
    assert np.array_equal(sub_trs[[1, 3, 5]].plaintext, PLAINTEXTS_10[[1, 3, 5]])


def test_double_indexing_2(trs):
    sub_trs = trs[[1, 3, 5, 7, 9, 11]]
    assert np.array_equal(sub_trs[:4].plaintext, PLAINTEXTS_10[[1, 3, 5, 7]])
    assert np.array_equal(sub_trs[[1, 3]].plaintext, PLAINTEXTS_10[[3, 7]])


def test_double_indexing_by_integer(trs):
    # requires an independent test to avoid caching effect
    sub_trs = trs[:20]
    sub_trs.samples[3]
    assert np.array_equal(sub_trs[3].plaintext, PLAINTEXTS_10[3])
    sub_trs = trs[[1, 3, 5, 7, 9, 11]]
    assert np.array_equal(sub_trs[3].plaintext, PLAINTEXTS_10[7])


@pytest.mark.skipif(trsfile.__version__ == '0.3.2', reason="Test only for trsfile v2.")
def test_read_trs_v2_file():
    filename = 'tests/samples/test_without_parser.trs'
    trs = read_ths_from_trs_file(filename, dtype='float32',
                                 metadatas_parsers=dict(plaintext=lambda x: np.frombuffer(x, dtype='uint8', offset=0, count=16),
                                                        ciphertext=lambda x: np.frombuffer(x, dtype='uint8', offset=16, count=16)))
    ref = {'title_space': 255, 'sample_coding': 2, 'length_data': 34, 'number_samples': 33, 'number_traces': 10, 'trace_block': None}
    for key, value in ref.items():
        print(key)
        assert trs.headers[key] == value
    assert trs[:3].samples[:].shape == (3, 33)
    assert trs[:3].ciphertext[:].shape == (3, 16)
    assert trs[:3].plaintext[:].shape == (3, 16)


@pytest.mark.skipif(trsfile.__version__ == '0.3.2', reason="Test only for trsfile v2.")
def test_trs_without_metadata_parser():
    filename = 'tests/samples/test_without_parser.trs'
    trs = read_ths_from_trs_file(filename, dtype='float32', metadatas_parsers=None)
    assert trs[:3].samples[:].shape == (3, 33)
    assert trs[:3].ciphertext[:].shape == (3, 16)
    assert trs[:3].plaintext[:].shape == (3, 16)
    assert trs[:3].indice[:].shape == (3, 1)


@pytest.mark.skipif(trsfile.__version__ == '0.3.2', reason="Test only for trsfile v2.")
def test_trsv1_without_metadata_parser_raises():
    with pytest.raises(TypeError, match='The TRS file does not have a TRACE_PARAMETER_DEFINITIONS field. You must set the'):
        read_ths_from_trs_file(TRS_FILENAME, dtype='float32', metadatas_parsers=None)


@pytest.mark.skipif(trsfile.__version__ != '0.3.2', reason="Test only for trsfile v1.")
def test_old_trsfile_version_without_metadata_parser_raises():
    with pytest.raises(TypeError, match='Setting a TRS reader with `metadatas_parsers=None` allowed only '):
        read_ths_from_trs_file(TRS_FILENAME, dtype='float32', metadatas_parsers=None)
