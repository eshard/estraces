from .context import estraces # noqa
import pytest
import numpy as np
from estraces.formats import ets_format
import cloudpickle

path = 'tests/samples/test.ets'


@pytest.fixture
def ets():
    return ets_format.read_ths_from_ets_file(filename=path)


def compare_ets(ets1, ets2, attribute_names=['plaintext', 'foo_bar', 'ciphertext']):
    assert np.array_equal(ets1.samples[:], ets2.samples[:])
    for attribute_name in attribute_names:
        assert np.array_equal(getattr(ets1, attribute_name), getattr(ets2, attribute_name)), f'{attribute_name} not equal'


def test_serialization_full(ets):
    dumped = cloudpickle.dumps(ets)
    loaded_ets = cloudpickle.loads(dumped)
    compare_ets(ets, loaded_ets)


def test_serialization_sub_ths_part_1(ets):
    dumped = cloudpickle.dumps(ets[10:20])
    loaded_ets = cloudpickle.loads(dumped)
    compare_ets(ets[10:20], loaded_ets)


def test_serialization_sub_ths_part_2(ets):
    ets_part = ets[50:80:2]
    dumped = cloudpickle.dumps(ets_part)
    loaded_ets = cloudpickle.loads(dumped)
    compare_ets(ets_part, loaded_ets)


def test_serialize_headers():
    ets = ets_format.read_ths_from_ets_file('tests/samples/test_with_headers.ets')
    headers_dict = {'foo': 'bar', 'bar': 42, 'list': [1, 2, 3], 'ndarray': np.arange(5, dtype='int8')}
    assert ets.headers == headers_dict


def test_cached_metadata_not_serialized(ets):
    assert ets._metadatas is None
    dumped1 = cloudpickle.dumps(ets)
    assert ets._metadatas is None
    _ = ets.plaintext[:]
    assert ets._metadatas is not None
    dumped2 = cloudpickle.dumps(ets)
    assert len(dumped1) == len(dumped2)
    loaded_ets = cloudpickle.loads(dumped2)
    assert loaded_ets._metadatas is None
