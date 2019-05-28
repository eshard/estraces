from .context import estraces  # noqa
import pytest
import numpy as np
from estraces import build_trace_header_set
from estraces._legacy_formats.bin_format import BinReader
from estraces._legacy_formats.ets_format import ETSReader


@pytest.fixture
def ths(fmt):
    return build_trace_header_set(reader=fmt, name="Test legacy ths")


def test_init_legacy_ths_raise_deprecation_warnings(ths):
    import warnings
    warnings.simplefilter('default', DeprecationWarning)
    with pytest.warns(DeprecationWarning):
        ths.get_points()


def test_ths_get_points(ths):
    assert np.array_equal(ths.get_points(), ths.samples[:, ...])

    # frame can be range, tuple
    assert np.array_equal(
        ths.samples[:, slice(2, 6, 2)],
        ths.get_points(frame=range(2, 6, 2))
    )

    assert np.array_equal(
        ths.samples[:, [2, 6, 8]],
        ths.get_points(frame=(2, 6, 8))
    )


def test_getters(ths):

    assert np.array_equal(
        ths.plain_t,
        ths.get_plain_t()
    )

    assert np.array_equal(
        ths.plaintext,
        ths.get_plaintext()
    )

    assert np.array_equal(
        ths.ciphertext,
        ths.get_ciphertext()
    )

    assert np.array_equal(
        ths.indices,
        ths.get_indices()
    )


def test_ths_get_method(ths):

    assert np.array_equal(
        ths.plaintext,
        ths.get('plaintext')
    )

    assert ths.samples.tolist() == ths.get('data').tolist()

    assert np.array_equal(
        ths.samples[:, [1, 5]],
        ths.get('points', frame=[1, 5])
    )

    with pytest.raises(AttributeError):
        ths.get('noneattr')


def test_get_attr_method(ths):

    assert np.array_equal(
        ths.plaintext,
        ths.get_attr('plaintext')
    )

    assert ths.samples.tolist() == ths.get_attr('data').tolist()

    assert np.array_equal(
        ths.samples[:, [1, 5]],
        ths.get_attr('points', frame=[1, 5])
    )

    with pytest.raises(AttributeError):
        ths.get_attr('noneattr')


def test_get_trace_by_index(ths):

    assert ths[4] == ths.get_trace_by_index(x=4)

    with pytest.raises(IndexError):
        ths.get_trace_by_index(x=444)


def test_trace_points_method(ths):
    trc = ths[0]
    assert np.array_equiv(trc.samples[:], trc.points())

    # frame can be range, tuple, int
    assert np.array_equal(
        trc.samples[slice(2, 6, 2)],
        trc.points(frame=range(2, 6, 2))
    )

    assert np.array_equal(
        trc.samples[[2, 6, 8]],
        trc.points(frame=(2, 6, 8))
    )

    assert trc.samples[12] == trc.points(frame=12)
    assert isinstance(trc.points(frame=12), np.uint8)


def test_trace_nb_point(ths):
    assert ths[0].nb_point == len(ths[0])


def test_bin_format_ths_filename_attribute():
    test_traces_fn_pattern = "./tests/samples/AESEncrypt.500MSs.cur.*"
    ths = BinReader(filename_pattern=test_traces_fn_pattern, dtype='uint8')
    assert ths[0].filename == ths[0]._reader._filenames[0]
    assert ths.get('filename').tolist() == ths._reader._filenames.tolist()


def test_legacy_ets_initialization():
    ets = ETSReader(filename='tests/samples/test.ets')
    assert ets.samples.tolist() == ets.get_points().tolist()


def test_get_trace_by_index_method(ths):
    assert ths.get_trace_by_index(2) == ths[2]


def test_get_writable_attributes(ths):
    assert list(ths[0].metadatas.keys()) == ths[0].__get_writable_attributes__()
    ths[0].new_prop = 'value'
    exp = [k for k in ths[0].metadatas.keys()]
    exp.append('new_prop')
    assert exp == ths[0].__get_writable_attributes__()


def test_h5_file_property(ths):
    try:
        assert ths.h5_file == ths._reader._h5_file
    except AttributeError:
        assert ths.h5_file is None


def test_h5_close_method(ths):
    assert ths.close() is None
