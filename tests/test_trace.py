from .context import estraces  # noqa
import pytest
import numpy as np
from .conftest import PLAINS, CIPHERS


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
