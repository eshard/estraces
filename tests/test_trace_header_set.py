from .context import estraces  # noqa
import pytest
import numpy as np
from .format_test_implementation import Format
from .conftest import DATAS, PLAINS, CIPHERS, KEY, FOO, TIME


@pytest.fixture
def ths(fmt):
    return estraces.traces.trace_header_set.build_trace_header_set(reader=fmt, name="Test ths")


@pytest.fixture
def ths2():
    fmt = Format(metadatas={"other": range(10)})
    return estraces.traces.trace_header_set.build_trace_header_set(reader=fmt, name="Another ths")


def test_trace_header_set_cant_be_instantiate_directly():
    with pytest.raises(TypeError):
        estraces.traces.trace_header_set.TraceHeaderSet(reader={})


def test_read_metadatas_returns_metadatas_type(ths):
    assert isinstance(ths.metadatas, estraces.traces.metadatas.Metadatas)


def test_read_metadatas_with_attribute(ths, fmt):
    assert np.array_equal(ths.metadatas["ciphertext"], ths.ciphertext)
    assert np.array_equal(ths.ciphertext, CIPHERS)
    assert np.array_equal(ths.metadatas["plaintext"], ths.plaintext)
    assert np.array_equal(ths.plaintext, PLAINS)


def test_read_metadatas_attributes_missing_raises_attribute_error(ths, ths2):
    with pytest.raises(AttributeError):
        ths.myproperty
    with pytest.raises(AttributeError):
        ths2.ciphertext


def test_trace_header_set_initialization_with_improper_format():
    with pytest.raises(TypeError):
        estraces.traces.trace_header_set.build_trace_header_set(reader={}, name="")


def test_trace_header_set_initialization(ths):
    assert isinstance(ths, estraces.TraceHeaderSet)


def test_traces_attribute(ths):
    assert isinstance(ths.traces[0], estraces.Trace)
    assert len(ths.traces) == 10


def test_traces_length_when_slicing(ths):
    # See issue https://gitlab.eshard.int/esdynamic/estraces/issues/41
    # with a correct __len__ we are able to iterate over the sliced traces
    assert len(ths.traces[:]) == 10
    assert len(ths.traces[0:]) == 10
    assert len(ths.traces[2:]) == 8
    assert len(ths.traces[:10]) == 10
    assert len(ths.traces[:8]) == 8
    assert len(ths.traces[2:8]) == 6
    assert len(ths.traces[-2:10]) == 2
    assert len(ths.traces[2:-2]) == 6


def test_samples_returns_samples_type(ths):
    assert isinstance(ths.samples, estraces.traces.samples.Samples)


def test_ths_slice_is_a_new_ths(ths):
    new_ths = ths[:]
    assert isinstance(new_ths, estraces.TraceHeaderSet)
    assert id(new_ths) != id(ths)
    for idx, t in enumerate(new_ths.samples):
        assert id(t) != id(ths.samples[idx])
        assert np.array_equal(t[:], ths.samples[idx])


def test_ths_slicing_with_out_of_range_int_raises_index_error(ths):
    with pytest.raises(IndexError):
        ths[100000]


def test_ths_slicing_raises_error_with_improper_types(ths):
    with pytest.raises(IndexError):
        ths[1, 44]
    with pytest.raises(IndexError):
        ths["kdjfkldjflk"]
    with pytest.raises(IndexError):
        ths[range(1, 10)]
    with pytest.raises(IndexError):
        ths[np.array([[1, 2], [3, 4]])]
    with pytest.raises(IndexError):
        ths[np.array(["x", "tutu"])]
    with pytest.raises(IndexError):
        ths[np.array([1.4, 2.2])]
    with pytest.raises(IndexError):
        ths[None]
    with pytest.raises(IndexError):
        ths[np.float32(1.6)]


def test_ths_slicing_int_or_numpy_int(ths):
    expected = DATAS[1]
    for t in (1, np.int8(1), np.int32(1), np.int64(1)):
        assert np.array_equal(ths[t].samples[:], expected)


def test_ths_iteration_on_traces(ths):
    for t in ths:
        assert isinstance(t, estraces.Trace)


def test_ths_slicing_with_int_returns_a_trace(ths):
    assert isinstance(ths[1], estraces.Trace)
    assert np.array_equal(ths[1].samples[:], DATAS[1])


def test_ths_slicing_with_out_of_bounds_value_raise_exception(ths):
    new_ths = ths[[0, 2, 4, 6, 8, 9]]

    with pytest.raises(IndexError):
        new_ths[10]

    with pytest.raises(IndexError):
        new_ths[[1, 11]]

    with pytest.raises(IndexError):
        new_ths[np.array([1, 11])]


def test_ths_slicing_with_exotic_slices_returns_ths_of_max_size(ths):
    assert 10 == len(ths[:1000])
    assert 8 == len(ths[-11:8])
    assert 10 == len(ths[-1000:])
    assert 5 == len(ths[-1000:-5])
    assert 0 == len(ths[:-12])
    assert 3 == len(ths[-5: -2])
    assert 8 == len(ths[-12: -2])
    assert 0 == len(ths[-2: -12])
    assert 0 == len(ths[-2: 1])
    assert 1 == len(ths[-2: 9])
    assert 2 == len(ths[-2: 10])
    assert 10 == len(ths[-1000:1000])
    assert 0 == len(ths[1000:-1000])
    assert 0 == len(ths[1000:-5])
    assert 0 == len(ths[1000:])
    assert 0 == len(ths[1000:2])
    assert 0 == len(ths[1000:2000])


def test_ths_slicing_with_slice_returns_subths(ths):
    new_ths = ths[0:2:2]
    assert len(new_ths) == 1
    new_ths = ths[0:7:2]
    assert len(new_ths) == 4
    assert new_ths.plaintext.tolist() == ths.plaintext[0:7:2].tolist()

    new_ths = ths[3:]
    assert len(new_ths) == 7
    new_ths = ths[:8]
    assert len(new_ths) == 8


def test_succesive_ths_slicings_keeps_samples_metas_consistent(ths):
    sub_t_1 = ths[0:4]
    sub_t_2 = ths[4:8]
    assert len(sub_t_1) == 4
    assert len(sub_t_2) == 4
    assert sub_t_1.samples.tolist() == ths.samples[0:4].tolist()
    assert sub_t_2.samples.tolist() == ths.samples[4:8].tolist()
    assert sub_t_1.plaintext.tolist() == ths.plaintext[0:4].tolist()
    assert sub_t_2.plaintext.tolist() == ths.plaintext[4:8].tolist()
    assert sub_t_1.ciphertext.tolist() == ths.ciphertext[0:4].tolist()
    assert sub_t_2.ciphertext.tolist() == ths.ciphertext[4:8].tolist()
    assert sub_t_1.headers == sub_t_2.headers
    assert ths.headers == sub_t_1.headers

    sub_sub_t_1 = sub_t_1[0:2]
    sub_sub_t_2 = sub_t_2[0:2]
    assert len(sub_sub_t_1) == 2
    assert len(sub_sub_t_2) == 2
    assert sub_sub_t_1.samples.tolist() == ths.samples[0:2].tolist()
    assert sub_sub_t_2.samples.tolist() == ths.samples[4:6].tolist()
    assert sub_sub_t_1.plaintext.tolist() == ths.plaintext[0:2].tolist()
    assert sub_sub_t_2.plaintext.tolist() == ths.plaintext[4:6].tolist()
    assert sub_sub_t_1.plaintext.tolist() == ths.plaintext[0:2].tolist()
    assert sub_sub_t_2.plaintext.tolist() == ths.plaintext[4:6].tolist()
    assert sub_sub_t_1.ciphertext.tolist() == ths.ciphertext[0:2].tolist()
    assert sub_sub_t_2.ciphertext.tolist() == ths.ciphertext[4:6].tolist()
    assert sub_sub_t_1.headers == sub_sub_t_2.headers
    assert ths.headers == sub_sub_t_1.headers

    sub_sub_t_12 = sub_t_1[2:4]
    assert len(sub_sub_t_12) == 2
    assert sub_sub_t_12.samples.tolist() == ths.samples[2:4].tolist()
    assert sub_sub_t_12.plaintext.tolist() == ths.plaintext[2:4].tolist()
    assert sub_sub_t_12.ciphertext.tolist() == ths.ciphertext[2:4].tolist()
    assert sub_sub_t_12.headers == ths.headers

    sub_sub_t_22 = sub_t_2[2:4]
    assert len(sub_sub_t_22) == 2
    assert sub_sub_t_22.samples.tolist() == ths.samples[6:8].tolist()
    assert sub_sub_t_22.plaintext.tolist() == ths.plaintext[6:8].tolist()
    assert sub_sub_t_22.ciphertext.tolist() == ths.ciphertext[6:8].tolist()
    assert sub_sub_t_22.headers == ths.headers

    sub_sub_sub_t_1 = sub_sub_t_1[0:1]
    assert len(sub_sub_sub_t_1) == 1
    assert sub_sub_sub_t_1.samples.tolist() == ths.samples[0:1].tolist()
    assert sub_sub_sub_t_1.plaintext.tolist() == ths.plaintext[0:1].tolist()
    assert sub_sub_sub_t_1.ciphertext.tolist() == ths.ciphertext[0:1].tolist()
    assert sub_sub_sub_t_1.headers == ths.headers


def test_ths_slicing_with_list_returns_subths(ths):
    new_ths = ths[[1, 2, 4]]
    assert len(new_ths) == 3


def test_ths_slicing_with_nparray_returns_subths(ths):
    new_ths = ths[np.array([1, 2, 4])]
    assert len(new_ths) == 3


def test_ths_slicing_with_undeduped_unordered_list(ths):
    new_ths = ths[[3, 2, 1]]
    assert 3 == len(new_ths)
    assert ths[3].samples[:].tolist() == new_ths[0].samples[:].tolist()
    assert ths[1].samples[:].tolist() == new_ths[-1].samples[:].tolist()

    new_ths = ths[[3, 2, 1, 2]]
    assert 4 == len(new_ths)
    assert ths[3].samples[:].tolist() == new_ths[0].samples[:].tolist()
    assert ths[2].samples[:].tolist() == new_ths[-1].samples[:].tolist()
    assert ths[2].samples[:].tolist() == new_ths[1].samples[:].tolist()
    assert ths[1].samples[:].tolist() == new_ths[2].samples[:].tolist()


def test_ths_slicing_with_zero_len_slice_returns_empty_ths(ths):
    assert isinstance(ths[0:0], estraces.TraceHeaderSet)
    assert 0 == len(ths[0:0])


def test_ths_filter():
    ths = estraces.traces.trace_header_set.build_trace_header_set(
        name="test ths",
        reader=Format(datas=np.vstack([np.array(range(i, i + 100)) for i in range(100)])),
    )
    filtered_ths = ths.filter(lambda t: t.samples[0] % 2 == 0)
    assert len(filtered_ths) == 50
    assert id(ths) != filtered_ths
    for trc in filtered_ths:
        assert trc.samples[0] % 2 == 0


def test_split(ths):
    split = ths.split(part_size=2)
    assert len(split) == 5

    for t in split:
        assert len(t) == 2

    split = ths.split(part_size=3)
    assert len(split) == 4

    for idx in range(len(split) - 1):
        assert len(split[idx]) == 3
        assert len(split[idx]) == split[idx].samples[:].shape[0]
    assert len(split[-1]) == 1
    assert len(split[-1]) == split[-1].samples[:].shape[0]

    for t in split:
        assert isinstance(t, estraces.TraceHeaderSet)


def test_split_raises_error_with_improper_type(ths):
    with pytest.raises(TypeError):
        ths.split(part_size="eeze")
        ths.split(part_size=-4)

    split = ths.split(part_size=2)
    with pytest.raises(TypeError):
        split["jj"]
        split[-15]
        split[1:2]


def test_str_and_repr(ths):
    assert repr(ths) == str(ths)
    assert repr(ths[0]) == str(ths[0])


def test_attribute_headers_returns_headers_type(ths):
    assert isinstance(ths.headers, estraces.traces.headers.Headers)


def test_attribute_headers(ths):
    assert np.array_equal(ths.headers['key'], KEY)
    assert ths.headers['time'] == TIME
    assert ths.headers['foo'] == FOO


def test_add_metadata_to_ths(ths):
    datas = np.random.randint(0, 255, (len(ths), 1), dtype='uint8')
    ths.metadatas['new_meta'] = datas
    assert np.array_equal(ths.metadatas['new_meta'], datas)
    assert np.array_equal(ths.new_meta, datas)


def test_init_metadatas_and_slice_ths_works_as_expected(ths):
    ths.metadatas
    assert np.array_equal(ths[:1].metadatas['plaintext'], PLAINS[:1])


def test_add_metadata_to_ths_through_attribute(ths):
    datas = np.random.randint(0, 255, (len(ths), 1), dtype='uint8')
    ths.new_meta = datas
    assert np.array_equal(ths.metadatas['new_meta'], datas)
    assert np.array_equal(ths.new_meta, datas)

    sub_ths = ths[0:2]
    assert np.array_equal(sub_ths.metadatas['new_meta'], datas[:2])
    assert np.array_equal(sub_ths.new_meta, datas[:2])

    trace = ths[8]
    assert np.array_equal(trace.metadatas['new_meta'], datas[8])
    assert np.array_equal(trace.new_meta, datas[8])


def test_add_existing_metadata_or_attribute_raises_exception(ths):
    with pytest.raises(AttributeError):
        ths.plaintext = 'foo'
    with pytest.raises(AttributeError):
        ths.samples = 'foo'
    ths.__doc__ = 'foo'
    assert ths.__doc__ == 'foo'
    with pytest.raises(KeyError):
        ths.metadatas['__doc__']
