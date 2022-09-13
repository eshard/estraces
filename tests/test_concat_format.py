from .context import estraces # noqa
import pytest
from estraces.formats import concat_format, ets_format
import numpy as np
import cloudpickle


path = 'tests/samples/test.ets'


@pytest.fixture
def ets1():
    return ets_format.read_ths_from_ets_file(filename=path)


@pytest.fixture
def concat1():
    ets1 = ets_format.read_ths_from_ets_file(filename=path)
    ets2 = ets_format.read_ths_from_ets_file(filename=path)
    return concat_format.read_ths_from_multiple_ths(ets1, ets2)


def test_instanciate(concat1):
    assert isinstance(concat1, estraces.TraceHeaderSet)
    assert 400 == len(concat1)


def test_sub_ths(concat1):
    concat2 = concat1[:40]
    assert isinstance(concat2, estraces.TraceHeaderSet)
    assert 40 == len(concat2)


def test_raises_type_error_if_args_are_not_all_traceheadersets(ets1):
    with pytest.raises(TypeError):
        concat_format.read_ths_from_multiple_ths(ets1, "foo_bar")


def test_split_preserves_length_consistency_of_sub_ths(concat1):
    parts = concat1.split(100)
    for _, p in enumerate(parts):
        assert len(p) == 100
        assert len(p) == p.samples[:].shape[0]


inconsistent_trace_sets = ['test_without_plaintext.ets',
                           'test_with_plaintext_too_short.ets',
                           'test_with_plaintext_wrong_dtype.ets',
                           'test_with_plaintext_wrong_name.ets',
                           'test_with_samples_too_short.ets',
                           'test_with_samples_wrong_dtype.ets']


@pytest.fixture(params=inconsistent_trace_sets)
def ets2(request):
    path = 'tests/samples/' + request.param
    return ets_format.read_ths_from_ets_file(filename=path)


def test_wrong_meta_and_samples_consistency_raises(ets1, ets2):
    with pytest.raises(ValueError):
        concat_format.read_ths_from_multiple_ths(ets1, ets2)


def test_meta_strings_with_different_lengths():
    ets1 = ets_format.read_ths_from_ets_file(filename='tests/samples/test_str_meta_1.ets')
    ets2 = ets_format.read_ths_from_ets_file(filename='tests/samples/test_str_meta_2.ets')
    ets = concat_format.read_ths_from_multiple_ths(ets1, ets2)

    str_stack = np.concatenate((ets1.str_meta[:], ets2.str_meta[:]), axis=0)
    np_str_stack = np.concatenate((ets1.np_str_meta[:], ets2.np_str_meta[:]), axis=0)

    assert np.array_equal(ets.str_meta[:], str_stack)
    assert np.array_equal(ets.np_str_meta[:], np_str_stack)


def test_inconsistent_headers_raises_exception():
    ths_1 = estraces.read_ths_from_ram(
        np.random.randint(0, 255, (1000, 1200), dtype='uint8'),
        headers={'foo': 'ths_1', 'bar': 'bar'},
        plaintext=np.random.randint(0, 255, (1000, 16), dtype='uint8')
    )
    ths_2 = estraces.read_ths_from_ram(
        np.random.randint(0, 255, (1000, 1200), dtype='uint8'),
        headers={'foo': 'ths_2', 'bar': 'bar'},
        plaintext=np.random.randint(0, 255, (1000, 16), dtype='uint8')
    )
    with pytest.raises(ValueError):
        concat_format.read_ths_from_multiple_ths(ths_1, ths_2)
    ths_1 = estraces.read_ths_from_ram(
        np.random.randint(0, 255, (1000, 1200), dtype='uint8'),
        headers={'foo': 'ths_1', 'bar': 'bar'},
        plaintext=np.random.randint(0, 255, (1000, 16), dtype='uint8')
    )
    ths_2 = estraces.read_ths_from_ram(
        np.random.randint(0, 255, (1000, 1200), dtype='uint8'),
        headers={'foo_2': 'ths_2', 'bar': 'bar'},
        plaintext=np.random.randint(0, 255, (1000, 16), dtype='uint8')
    )
    with pytest.raises(ValueError):
        concat_format.read_ths_from_multiple_ths(ths_1, ths_2)


def test_headers_can_be_overridden():
    ths_1 = estraces.read_ths_from_ram(
        np.random.randint(0, 255, (1000, 1200), dtype='uint8'),
        headers={'foo': 'ths_1', 'bar': 'bar', 'spec': 1},
        plaintext=np.random.randint(0, 255, (1000, 16), dtype='uint8')
    )
    ths_2 = estraces.read_ths_from_ram(
        np.random.randint(0, 255, (1000, 1200), dtype='uint8'),
        headers={'foo': 'ths_2', 'bar': 'bar', 'true': 2},
        plaintext=np.random.randint(0, 255, (1000, 16), dtype='uint8')
    )
    with pytest.raises(ValueError):
        concat_format.read_ths_from_multiple_ths(ths_1, ths_2)

    ths = concat_format.read_ths_from_multiple_ths(
        ths_1, ths_2,
        headers={
            'foo': 'concat ths',
            'spec': 1,
            'true': 2,
            'new_header': 'new'
        }
    )
    assert sorted(list(ths.headers.keys())) == ['bar', 'foo', 'new_header', 'spec', 'true']
    assert ths.headers['foo'] == 'concat ths'
    assert ths.headers['bar'] == 'bar'
    assert ths.headers['spec'] == 1
    assert ths.headers['true'] == 2
    assert ths.headers['new_header'] == 'new'


def test_serialization(concat1):
    ths = concat1
    dumped = cloudpickle.dumps(ths)
    loaded_ths = cloudpickle.loads(dumped)
    assert np.array_equal(loaded_ths.samples[:], ths.samples[:])
    assert np.array_equal(loaded_ths.plaintext[:], ths.plaintext[:])
    assert np.array_equal(loaded_ths.foo_bar[:], ths.foo_bar[:])
    assert np.array_equal(loaded_ths.ciphertext[:], ths.ciphertext[:])


def test_concat_keeps_order():
    ths_list = [estraces.read_ths_from_ram(np.random.randint(0, 255, (100, 120), dtype='uint8')) for i in range(10)]

    ths = concat_format.read_ths_from_multiple_ths(*ths_list)
    for i in range(10):
        assert np.array_equal(ths_list[i].samples[:], ths.samples[i * 100:(i + 1) * 100]), 'Concatenation does not guarantee order.'
