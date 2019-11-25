from .context import estraces  # noqa
import pytest
import numpy as np
from .conftest import _HEADERS


@pytest.fixture(params=['1 trace', 'all traces'])
def headers(request, fmt):
    return estraces.traces.headers.Headers(reader=fmt)


def test_headers_init_with_improper_reader_type_raises_exception():
    with pytest.raises(TypeError):
        estraces.traces.headers.Headers(reader="oups")


def test_headers_repr(headers):
    assert repr(headers) == str(headers)


def test_read_headers_key_throws_key_error_on_missing_key(headers):
    with pytest.raises(KeyError):
        headers["missing"]
    with pytest.raises(KeyError):
        headers["3"]


def test_read_headers_key_throws_type_error_on_incorrect_key_type(headers):
    with pytest.raises(TypeError):
        headers[np.array(range(2))]
    with pytest.raises(TypeError):
        headers[np.array([1, 2, 3], dtype="uint8")]


def test_read_headers_returns_a_mapping(headers):
    keys = [k for k in sorted(headers.keys())]
    assert 'foo' in keys
    assert 'key' in keys
    assert 'time' in keys
    for k, v in headers.items():
        if k == 'key':
            assert np.array_equal(v, _HEADERS[k])
        elif k == 'foo' or k == 'time':
            assert v == _HEADERS[k]


def test_headers_equality_with_not_mapping_type_raise_exception(headers):
    with pytest.raises(NotImplementedError):
        assert headers == 2


def test_headers_equality(headers, fmt):
    d = dict(headers)
    d_1 = dict(headers)
    d_1.update({'another_key': 122390490})
    assert headers == d
    assert headers != d_1
    h_2 = estraces.traces.headers.Headers(fmt)
    assert headers == h_2
    assert id(headers) != id(h_2)
