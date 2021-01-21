from .context import estraces # noqa
import pytest
import numpy as np
from estraces.formats import ets_format, ram_format
import cloudpickle

path = 'tests/samples/test.ets'


def load_data():
    ets = ets_format.read_ths_from_ets_file(filename=path)
    samples = ets.samples[:].copy()
    plaintext = ets.plaintext[:].copy()
    foo_bar = ets.foo_bar[:].copy()
    ciphertext = ets.ciphertext[:].copy()
    return samples, plaintext, foo_bar, ciphertext


@pytest.fixture
def ram1():
    samples, plaintext, foo_bar, ciphertext = load_data()
    return ram_format.read_ths_from_ram(samples=samples, plaintext=plaintext, foo_bar=foo_bar, ciphertext=ciphertext)


def test_instanciate(ram1):
    assert isinstance(ram1, estraces.TraceHeaderSet)
    assert 200 == len(ram1)


def test_raise_type_error_if_inputs_are_not_ndarray():
    samples, plaintext, foo_bar, ciphertext = load_data()
    with pytest.raises(TypeError):
        ram_format.read_ths_from_ram(samples.tolist(), plaintext=plaintext, foobar=foo_bar, ciphertext=ciphertext)
    with pytest.raises(TypeError):
        ram_format.read_ths_from_ram(samples, plaintext=plaintext.tolist(), foo_bar=foo_bar, ciphertext=ciphertext)


def test_raise_type_error_if_inputs_are_not_correct_arrays():
    _, plaintext, foo_bar, ciphertext = load_data()
    with pytest.raises(TypeError):
        ram_format.read_ths_from_ram(np.arange(200), plaintext=plaintext, foobar=foo_bar, ciphertext=ciphertext)


def test_incompatible_shapes_raise_value_error():
    samples, plaintext, foo_bar, ciphertext = load_data()
    with pytest.raises(ValueError):
        ram_format.read_ths_from_ram(samples[:10], plaintext=plaintext, foobar=foo_bar, ciphertext=ciphertext)
    with pytest.raises(ValueError):
        ram_format.read_ths_from_ram(samples, plaintext=plaintext[:10], foo_bar=foo_bar, ciphertext=ciphertext)
    with pytest.raises(ValueError):
        ram_format.read_ths_from_ram(samples, plaintext=np.arange(300), foo_bar=foo_bar, ciphertext=ciphertext)


def test_read_metadatas(ram1):
    assert isinstance(ram1.metadatas, estraces.traces.metadatas.Metadatas)
    for v in ram1.metadatas.keys():
        assert v in ['ciphertext', 'foo_bar', 'plaintext']
    assert [2, 9, 15, 8, 11, 8, 2, 13] == ram1.metadatas['plaintext'][0].tolist()
    assert [2, 9, 15, 8, 11, 8, 2, 13] == ram1.plaintext[0].tolist()
    assert [2, 9, 15, 8, 11, 8, 2, 13] == ram1[0].plaintext.tolist()
    for _, v in ram1.metadatas.items():
        assert 2 == v.ndim
    for _, v in ram1[0].metadatas.items():
        assert 1 == v.ndim
    assert [
        [3, 10, 16, 9, 12, 9, 3, 14],
        [4, 11, 17, 10, 13, 10, 4, 15],
        [5, 12, 18, 11, 14, 11, 5, 16]
    ] == ram1.plaintext[1: 4].tolist()


def test_read_samples(ram1):
    assert isinstance(ram1.samples, estraces.traces.samples.Samples)
    assert ram1.samples.ndim == 2
    assert isinstance(ram1[0].samples, estraces.traces.samples.Samples)
    assert ram1[0].samples.ndim == 1
    assert [
        [2, 11, 21],
        [4, 13, 23],
        [6, 15, 25]
    ] == ram1.samples[1: 6: 2, [1, 10, 20]].tolist()

    assert [
        [2, 11, 21],
        [4, 13, 23],
        [6, 15, 25]
    ] == ram1.samples[1: 6: 2, np.array([1, 10, 20])].tolist()

    assert [
        [2, 3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11, 12],
        [6, 7, 8, 9, 10, 11, 12, 13, 14]
    ] == ram1.samples[1: 6: 2, 1:10].tolist()

    # Manage limits
    assert 0 == ram1.samples[0, 0]
    assert [1, 2, 3, 4, 5, 6, 7, 8, 9] == ram1.samples[0, 1:10].tolist()
    assert [1, 2, 3, 4, 5, 6, 7, 8, 9] == ram1.samples[1:10, 0].tolist()

    assert [1, 3, 4, 7, 9] == ram1[0].samples[np.array([1, 3, 4, 7, 9])].tolist()

    assert [1, 4, 3, 7, 4, 9] == ram1[0].samples[np.array([1, 4, 3, 7, 4, 9])].tolist()

    assert list(range(33)) == ram1[0].samples[...].tolist()
    assert 12 == ram1[0].samples[12]


def test_slice_set(ram1):
    sub1 = ram1[2: 7]
    sub2 = ram1[[1, 10, 12]]
    assert isinstance(sub1, estraces.TraceHeaderSet)
    assert isinstance(sub2, estraces.TraceHeaderSet)
    assert 5 == len(sub1)
    assert 3 == len(sub2)
    assert [2, 3, 4, 5] == sub1.samples[0, 0:4].tolist()
    assert [6, 7, 8, 9] == sub1.samples[4, 0:4].tolist()
    assert [1, 2, 3, 4] == sub2.samples[0, 0:4].tolist()
    assert [10, 11, 12, 13] == sub2.samples[1, 0:4].tolist()
    assert [12, 13, 14, 15] == sub2.samples[2, 0:4].tolist()
    assert sub1.plaintext.shape[0] == 5

    sub_3 = ram1[0:1]
    assert sub_3.plaintext.shape[0] == 1


def test_split_preserves_length_consistency_of_sub_ths(ram1):
    parts = ram1.split(100)
    for _, p in enumerate(parts):
        assert len(p) in (100, 56)
        assert len(p) == p.samples[:].shape[0]


def test_serialization(ram1):
    ths = ram1
    dumped = cloudpickle.dumps(ths)
    loaded_ths = cloudpickle.loads(dumped)
    assert np.array_equal(loaded_ths.samples[:], ths.samples[:])
    assert np.array_equal(loaded_ths.plaintext[:], ths.plaintext[:])
    assert np.array_equal(loaded_ths.foo_bar[:], ths.foo_bar[:])
    assert np.array_equal(loaded_ths.ciphertext[:], ths.ciphertext[:])
