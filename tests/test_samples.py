from .context import estraces  # noqa
import pytest
import numpy as np
from .conftest import DATAS


@pytest.fixture(params=['1 trace', 'all traces'])
def samples(request, fmt):
    return estraces.traces.samples.Samples(reader=fmt, trace_id=0 if request.param == '1 trace' else None)


@pytest.fixture
def s1d(fmt):
    return estraces.traces.samples.Samples(reader=fmt, trace_id=0)


@pytest.fixture
def s2d(fmt):
    return estraces.traces.samples.Samples(reader=fmt)


def test_samples_repr_is_equal_to_wrapped_ndarray_repr(samples):
    assert repr(samples) == repr(samples[...])


def test_1d_samples_len_is_equal_to_ndarray_len(s1d):
    assert len(s1d) == len(s1d[...])


def test_2d_samples_len_is_equal_to_number_of_traces(s2d):
    assert 10 == len(s2d)


def test_samples_init_with_improper_reader_type_raises_exception():
    with pytest.raises(TypeError):
        estraces.traces.samples.Samples(reader=lambda s: s)


def test_samples_slicing_with_improper_index_type_raises_error(samples):
    with pytest.raises(IndexError):
        samples["jfkkjdl"]
    with pytest.raises(IndexError):
        samples[..., "klklkg"]
    with pytest.raises(IndexError):
        samples["hjkghkj", ()]
    with pytest.raises(IndexError):
        samples["hjkghkj", "fhfh"]


def test_trace_samples_slicing_with_too_large_index_raises_error(samples):
    with pytest.raises(IndexError):
        samples[4555555]
    with pytest.raises(IndexError):
        samples[[1454544, 98090]]
    with pytest.raises(IndexError):
        samples[[1454544, 98090], ...]
    with pytest.raises(IndexError):
        samples[..., [1454544, 98090]]
    with pytest.raises(IndexError):
        samples[..., range(1454544)]


def test_trace_slicing_with_more_than_available_dimensions_raises_error(s1d, s2d):
    with pytest.raises(IndexError):
        s2d[..., 1, ...]
    with pytest.raises(IndexError):
        s1d[..., 1]


def test_samples_array_attribute(samples):
    assert isinstance(samples.array, np.ndarray)
    assert np.array_equal(samples[:], samples.array)


def test_samples_slicing_with_explicit_indices_on_2_dim(s2d):
    for i in range(10):
        assert len(s2d[i].shape) == 1
        assert np.array_equal(s2d[i, :], DATAS[i, :])
        assert np.array_equal(s2d[i, 10:50], DATAS[i, 10:50])


def test_samples_slicing_with_int_slice_list_on_traces_index(s2d):
    assert np.array_equal(s2d[2], DATAS[2, :])
    assert np.array_equal(s2d[1:8:2], DATAS[1:8:2, :])
    assert np.array_equal(s2d[[1, 2, 3]], DATAS[[1, 2, 3], :])


def test_samples_slicing_with_duped_unordered_list(s1d, s2d):
    assert DATAS[0, np.array([1, 4, 3, 7, 4, 9])].tolist() == s1d[np.array([1, 4, 3, 7, 4, 9])].tolist()
    assert DATAS[[3, 1, 2], :].tolist() == s2d[[3, 1, 2], :].tolist()
    assert DATAS[[3, 1, 2, 1], :].tolist() == s2d[[3, 1, 2, 1], :].tolist()
    assert DATAS[[1, 2, 1], :][:, [7, 8, 7]].tolist() == s2d[[1, 2, 1], [7, 8, 7]].tolist()


def test_samples_slicing_traces_with_upper_bound_out_of_bounds_returns_samples_on_available_traces(s2d):
    assert s2d[5:1000].shape[0] == 5
    assert s2d[11:1000].shape[0] == 0


def test_samples_ranging_traces_with_upper_bound_out_of_bounds_returns_samples_on_available_traces(s2d):
    assert s2d[range(5, 1000)].shape[0] == 5
    assert s2d[range(11, 1000)].shape[0] == 0


def test_samples_slicing_with_int_slice_list_on_samples_index(s2d):
    assert DATAS[:, 1:8:2].tolist() == s2d[:, 1:8:2].tolist()
    assert DATAS[:, 2].tolist() == s2d[:, 2].tolist()
    assert DATAS[:, [1, 2, 3]].tolist() == s2d[:, [1, 2, 3]].tolist()
    assert DATAS[..., [1, 2, 13]].tolist() == s2d[..., [1, 2, 13]].tolist()


def test_samples_iterates_over_traces(s2d):
    for index, trc_samples in enumerate(s2d):
        assert np.array_equal(trc_samples[:], DATAS[index])


def test_2d_samples_with_one_trace_has_appropriate_shape(s2d):
    s2d_bis = s2d[0:1, [1, 2]]
    assert s2d_bis.ndim == 2
    assert s2d_bis.shape[0] == 1

    s2d_bis = s2d[0:1, 1:4]
    assert s2d_bis.ndim == 2
    assert s2d_bis.shape[0] == 1

    s2d_ter = s2d[0, 1:10]
    assert s2d_ter.ndim == 1

    s2d_ter = s2d[0, [1, 2]]
    assert s2d_ter.ndim == 1


def test_trace_samples_iteration(s1d):
    for s in s1d:
        assert s is not None


def test_trace_samples_tolist(s1d):
    assert isinstance(s1d.tolist(), list)


def test_trace_samples_slicing_empty_array_with_out_of_the_window_slice(s1d):
    smp = s1d[5000:4544444]
    assert np.array_equal(np.array([]), smp)


def test_trace_samples_slicing_with_ellipsis_or_none(s1d):
    assert isinstance(s1d[...], np.ndarray)
    assert np.array_equal(s1d[...], s1d[:])


def test_trace_samples_slicing_with_slice(s1d):
    assert isinstance(s1d[10:200:2], np.ndarray)
    assert len(s1d[0:200:2]) == 100


def test_trace_samples_slicing_with_range(s1d):
    assert isinstance(s1d[range(10, 200, 2)], np.ndarray)
    assert len(s1d[range(0, 200, 2)]) == 100


def test_trace_samples_slicing_with_list(s1d):
    assert isinstance(s1d[[10, 20, 40]], np.ndarray)
    assert len(s1d[[10, 20, 40]]) == 3


def test_trace_samples_slicing_with_int(s1d):
    assert s1d[3] == s1d[:][3]
    assert isinstance(s1d[3], np.uint8)
    assert isinstance(s1d[3], type(s1d[:][0]))


def test_trace_samples_slicing_with_out_of_bounds_int(s1d, s2d):
    with pytest.raises(IndexError, match='index 10000000 is out of bounds for axis 0 with size 1000'):
        s1d[10000000]
    with pytest.raises(IndexError, match='index 11 is out of bounds for axis 0 with size 10'):
        s2d[11, :]
    with pytest.raises(IndexError, match='index 10000000000 is out of bounds for axis 1 with size 1000'):
        s2d[:, 10000000000]
    with pytest.raises(IndexError, match='index 10000000000 is out of bounds for axis 1 with size 1000'):
        s2d[0, 10000000000]


def test_samples_slicing_with_slice_of_len_zero(s1d, s2d):
    assert isinstance(s1d[10:10], np.ndarray)
    assert 0 == len(s1d[10:10])

    assert isinstance(s2d[:, 10:10], np.ndarray)
    assert 0 == s2d[:, 10:10].shape[1]

    assert isinstance(s2d[10:10, :], np.ndarray)
    assert 0 == s2d[10:10, :].shape[0]


def test_samples_slicing_with_range_of_len_zero(s1d, s2d):
    assert isinstance(s2d[10:10, :], np.ndarray)
    assert 0 == s2d[10:10, :].shape[0]


def test_samples_frame_slicing_with_range_of_len_zero_raises_index_error(s1d, s2d):
    with pytest.raises(IndexError):
        s1d[range(10, 10)]
    with pytest.raises(IndexError):
        s2d[:, range(10, 10)]


def test_samples_slicing_with_negative_slices(s1d, s2d):
    assert s1d[800:900].tolist() == s1d[-200: -100].tolist()
    assert [] == s1d[-100: -200].tolist()

    assert s2d[:, 800:900].tolist() == s2d[:, -200: -100].tolist()
    assert [[] for i in range(s2d.shape[0])] == s2d[:, -100: -200].tolist()

    assert s2d[8:9, :].tolist() == s2d[-2: -1, :].tolist()
    assert [] == s2d[-1:-2, :].tolist()


def test_samples_slicing_with_negative_range(s1d, s2d):
    assert s1d[800:900].tolist() == s1d[range(-200, -100)].tolist()
    with pytest.raises(IndexError):
        s1d[range(-100, -200)]

    assert s2d[:, 800:900].tolist() == s2d[:, range(-200, -100)].tolist()
    with pytest.raises(IndexError):
        s2d[:, range(-100, -200)]

    assert s2d[8:9, :].tolist() == s2d[range(-2, -1), :].tolist()
    assert [] == s2d[range(-1, -2), :].tolist()


def test_samples_implements_array_interface(s1d, s2d):
    # See issue https://gitlab.eshard.int/esdynamic/estraces/issues/35
    assert s1d.__array_interface__ is not None
    assert s2d.__array_interface__ is not None
    assert s1d.__array_struct__ is not None
    assert s2d.__array_struct__ is not None
    assert isinstance(s1d.__array_interface__, type(s1d[:].__array_interface__)) is not None
    assert isinstance(s2d.__array_interface__, type(s2d[:].__array_interface__)) is not None


def test_samples_is_castable_to_numpy(s1d, s2d):
    assert np.array_equal(
        np.array(s1d),
        s1d[:]
    )
    assert np.array_equal(
        np.array(s2d),
        s2d[:]
    )


def test_samples_instantiate_array_data_in_last_resort(s1d, s2d):
    # See issue https://gitlab.eshard.int/esdynamic/estraces/issues/52
    assert s2d._array is None
    s2d[0]
    assert s2d._array is None
    s2d.array
    assert s2d._array is not None


def test_samples_support_boolean_indexing(samples):
    boolean = np.array([True] + [False] * (len(samples) - 2) + [True], dtype=bool)
    assert len(samples[boolean]) == 2
    assert np.array_equal(samples[0], samples[boolean][0])
    assert np.array_equal(samples[-1], samples[boolean][1])


def test_samples_support_boolean_indexing_2d(s2d):
    boolean = np.array([True] + [False] * (s2d.shape[1] - 2) + [True], dtype=bool)
    assert s2d[:, boolean].shape[1] == 2
    assert np.array_equal(s2d[:, 0], s2d[:, boolean][:, 0])
    assert np.array_equal(s2d[:, -1], s2d[:, boolean][:, 1])


def test_samples_shape_does_not_load_data(s1d, s2d):
    shape = s1d.shape
    assert s1d._array is None
    assert shape == s1d.array.shape

    shape = s2d.shape
    assert s2d._array is None
    assert shape == s2d.array.shape


def test_samples_dtype_does_not_load_data(s1d, s2d):
    dtype = s1d.dtype
    assert s1d._array is None
    assert dtype == s1d.array.dtype

    dtype = s2d.dtype
    assert s2d._array is None
    assert dtype == s2d.array.dtype


def test_samples_size_does_not_load_data(s1d, s2d):
    size = s1d.size
    assert s1d._array is None
    assert size == s1d.array.size

    size = s2d.size
    assert s2d._array is None
    assert size == s2d.array.size
