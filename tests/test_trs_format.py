from .context import estraces  # noqa: F401
from estraces import read_ths_from_trs_file
import pytest
import numpy as np


TRS_FILENAME = 'tests/samples/aes128_sb_ciph_0fec9ca47fb2f2fd4df14dcb93aa4967.trs'


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


def test_trs_reader_optionnal_dtype_overrides_default():
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
