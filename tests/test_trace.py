from .context import estraces  # noqa
import pytest
import numpy as np
from .conftest import PLAINS, CIPHERS, KEY, TIME, FOO


@pytest.fixture
def ths(fmt):
    return estraces.traces.trace_header_set.build_trace_header_set(reader=fmt, name="Test THS")


@pytest.fixture
def trace(ths):
    return ths.traces[0]


def test_metadatas_is_metadatas_instance(trace):
    assert isinstance(trace.metadatas, estraces.traces.metadatas.Metadatas)


def test_read_metadatas_with_attribute(trace):
    assert np.array_equal(trace.metadatas["ciphertext"], trace.ciphertext)
    assert np.array_equal(trace.ciphertext, CIPHERS[0])
    assert np.array_equal(trace.metadatas["plaintext"], trace.plaintext)
    assert np.array_equal(trace.plaintext, PLAINS[0])


def test_read_metadatas_attributes_missing_raises_attribute_error(trace):
    with pytest.raises(AttributeError):
        trace.myproperty
    with pytest.raises(AttributeError):
        trace.ciphers


def test_trace_samples_is_samples_type(trace):
    assert isinstance(trace.samples, estraces.traces.samples.Samples)
    assert len(trace.samples) == 1000


def test_attribute_headers_returns_headers_type(trace):
    assert isinstance(trace.headers, estraces.traces.headers.Headers)


def test_attribute_headers(trace):
    assert np.array_equal(trace.headers['key'], KEY)
    assert trace.headers['time'] == TIME
    assert trace.headers['foo'] == FOO


def test_add_metadata_to_trace(trace):
    trace.metadatas['new_meta'] = 'foo'
    assert trace.metadatas['new_meta'] == 'foo'
    assert trace.new_meta == 'foo'


def test_add_metadata_to_trace_through_attribute(trace):
    trace.new_meta = 'foo'
    assert trace.new_meta == 'foo'
    assert trace.metadatas['new_meta'] == 'foo'


def test_add_existing_metadata_or_attribute_raises_exception(trace):
    with pytest.raises(AttributeError):
        trace.plaintext = 'foo'
    with pytest.raises(AttributeError):
        trace.samples = 'foo'
    trace.__doc__ = 'foo'
    assert trace.__doc__ == 'foo'
    with pytest.raises(KeyError):
        trace.metadatas['__doc__']
