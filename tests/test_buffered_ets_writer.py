from .context import estraces  # noqa
import numpy as np
import os
from estraces.formats.buffered_ets_writer import BufferedETSWriter, _Dataset, _SAMPLES_KEY
from estraces import read_ths_from_ets_file
import pytest

_output_ets_filename = 'BUFFERED_ETS-test.ets'


@pytest.fixture
def ets_reader():
    return read_ths_from_ets_file('tests/samples/test.ets')


@pytest.fixture
def output_ets_filename():
    try:
        os.remove(_output_ets_filename)
    except FileNotFoundError:
        pass
    yield _output_ets_filename
    try:
        os.remove(_output_ets_filename)
    except FileNotFoundError:
        pass


#############
#  Dataset  #
#############

def test_dataset_instantiation():
    _Dataset(42)


def test_dataset_raises_if_write_greater_than_1d_data():
    d = _Dataset(42)
    with pytest.raises(ValueError, match='Data must'):
        d.write(np.zeros((2, 3)))


def test_dataset_raises_if_write_when_full():
    d = _Dataset(42)
    data = np.zeros(42)
    for i in range(42):
        d.write(data)
    with pytest.raises(ValueError, match='Dataset is full'):
        d.write(data)


def test_dataset_write_and_get_vectors():
    d = _Dataset(42)
    data = np.random.rand(42, 123)
    for dat in data:
        d.write(dat)
    assert np.array_equal(d.get_data(), data)
    assert d.data is None
    assert d.current_index is None


def test_dataset_write_and_get_scalar():
    d = _Dataset(42)
    data = np.random.rand(42)
    for dat in data:
        d.write(dat)
    got_data = d.get_data()
    assert got_data.shape == (42, 1)
    assert np.array_equal(got_data.squeeze(), data)
    assert d.data is None
    assert d.current_index is None


def test_dataset_write_and_get_not_full():
    d = _Dataset(42)
    data = np.random.rand(33, 123)
    for dat in data:
        d.write(dat)
    assert np.array_equal(d.get_data(), data)
    assert d.data is None
    assert d.current_index is None


####################
#  BufferedWriter  #
####################

def test_buffered_writer_get_reader(output_ets_filename, ets_reader):
    scalar_data = np.arange(len(ets_reader))
    bw = BufferedETSWriter(output_ets_filename)
    bw.write_headers({'foo': 'bar', 'test': 42})
    for i, trace in enumerate(ets_reader):
        trace.scalar = scalar_data[i]
        samples = trace.samples[:]
        bw.write_trace_object_and_points(trace, samples)

    reader = bw.get_reader()
    assert np.array_equal(ets_reader.samples[:], reader.samples[:])
    assert np.array_equal(ets_reader.ciphertext[:], reader.ciphertext[:])
    assert np.array_equal(ets_reader.foo_bar[:], reader.foo_bar[:])
    assert np.array_equal(ets_reader.plaintext[:], reader.plaintext[:])
    assert np.array_equal(scalar_data, reader.scalar[:].squeeze())


def test_buffered_writer_close(output_ets_filename, ets_reader):
    scalar_data = np.arange(len(ets_reader))
    bw = BufferedETSWriter(output_ets_filename, buffer_length=42)
    bw.write_headers({'foo': 'bar', 'test': 42})
    for i, trace in enumerate(ets_reader):
        trace.scalar = scalar_data[i]
        samples = trace.samples[:]
        bw.write_trace_object_and_points(trace, samples)
    bw.close()

    reader = read_ths_from_ets_file(output_ets_filename)
    assert np.array_equal(ets_reader.samples[:], reader.samples[:])
    assert np.array_equal(ets_reader.ciphertext[:], reader.ciphertext[:])
    assert np.array_equal(ets_reader.foo_bar[:], reader.foo_bar[:])
    assert np.array_equal(ets_reader.plaintext[:], reader.plaintext[:])
    assert np.array_equal(scalar_data, reader.scalar[:].squeeze())


def test_buffered_writer_flush_when_empty(output_ets_filename, ets_reader):
    bw = BufferedETSWriter(output_ets_filename, buffer_length=42)
    for trace in ets_reader[:42]:
        samples = trace.samples[:]
        bw.write_trace_object_and_points(trace, samples)
    bw.flush()


def test_buffered_writer_raises_if_one_dataset_has_different_size(output_ets_filename, ets_reader):
    bw = BufferedETSWriter(output_ets_filename, buffer_length=42)
    for trace in ets_reader[:5]:
        samples = trace.samples[:]
        bw.write_trace_object_and_points(trace, samples)
    bw.datasets[_SAMPLES_KEY].write(ets_reader[42].samples[:])
    with pytest.raises(ValueError, match='All Datasets must have the same length'):
        bw.flush()


# https://gitlab.com/eshard/estraces/-/issues/40
def test_buffered_writer_ok_with_strings(output_ets_filename, ets_reader):
    bw = BufferedETSWriter(output_ets_filename, buffer_length=42)
    for trace in ets_reader[:5]:
        trace.foo = 'bar'
        samples = trace.samples[:]
        bw.write_trace_object_and_points(trace, samples)
    bw.flush()
