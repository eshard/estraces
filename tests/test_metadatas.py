from .context import estraces  # noqa
import pytest
import numpy as np
from .conftest import PLAINS, CIPHERS, INDICES


@pytest.fixture(params=['1 trace', 'all traces'])
def metadatas(request, fmt):
    return estraces.traces.metadatas.Metadatas(reader=fmt, trace_id=0 if request.param == '1 trace' else None)


@pytest.fixture
def metas_1_trace(fmt):
    return estraces.traces.metadatas.Metadatas(reader=fmt, trace_id=0)


@pytest.fixture
def metas_1_ths(fmt):
    return estraces.traces.metadatas.Metadatas(reader=fmt)


def test_metadatas_init_with_improper_reader_type_raises_exception():
    with pytest.raises(TypeError):
        estraces.traces.metadatas.Metadatas(reader="oups")


def test_metadatas_repr(metadatas):
    assert repr(metadatas)


def test_read_metadatas_key_throws_key_error_on_missing_key(metadatas):
    with pytest.raises(KeyError):
        metadatas["missing"]
    with pytest.raises(KeyError):
        metadatas[3]


def test_read_metadatas_key_throws_type_error_on_incorrect_key_type(metadatas):
    with pytest.raises(TypeError):
        metadatas[np.array(range(2))]
    with pytest.raises(TypeError):
        metadatas[np.array([1, 2, 3], dtype="uint8")]


def test_read_metadata_returns_array_of_values(metas_1_ths):
    assert isinstance(metas_1_ths["ciphertext"], np.ndarray)
    assert np.array_equal(metas_1_ths["ciphertext"], CIPHERS)


def test_read_metadata_returns_a_mapping(metadatas):
    assert [k for k in sorted(metadatas.keys())] == ["ciphertext", "indices", "plain_t", "plaintext"]
    if metadatas._trace_id is None:
        for k, v in metadatas.items():
            assert k in ("plaintext", "ciphertext", "indices", "plain_t")
            if k == 'ciphertext':
                assert np.array_equal(v, CIPHERS)
            elif k == 'plaintext' or k == 'plain_t':
                assert np.array_equal(v, PLAINS)
            else:
                assert np.array_equal(v, INDICES)
    else:
        for k, v in metadatas.items():
            assert k in ("plaintext", "ciphertext", "indices", "plain_t")
            if k == 'ciphertext':
                assert np.array_equal(v, CIPHERS[0])
            elif k == 'plaintext' or k == 'plain_t':
                assert np.array_equal(v, PLAINS[0])
            else:
                assert v == INDICES[0]


def test_read_metadata_return_values(metas_1_trace):
    assert isinstance(metas_1_trace["ciphertext"], np.ndarray)
    assert np.array_equal(metas_1_trace["ciphertext"], CIPHERS[0])
    assert isinstance(metas_1_trace["indices"], str)
    assert metas_1_trace["indices"] == INDICES[0]


def test_metadatas_cache(metas_1_ths):
    assert metas_1_ths._cache == {}
    metas_1_ths["ciphertext"]
    assert "ciphertext" in metas_1_ths._cache
    assert np.array_equal(metas_1_ths._cache['ciphertext'], CIPHERS)


def test_metadatas_set_new_attribute(metadatas):
    metadatas['new_attribute'] = 'foo'

    assert metadatas['new_attribute'] == 'foo'
