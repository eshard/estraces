from unittest.mock import patch

from estraces.formats.ets_writer import _get_optimal_batch_size

from .context import estraces  # noqa
from .conftest import _HEADERS
import numpy as np
import os
from estraces import ETSWriter
from estraces import read_ths_from_ets_file
import pytest

nb_trace = 1000
nb_points = 500
filename = 'ETS-test.ets'
compressed = 'ETS.etsz'
filename_1 = 'ETS1.ets'
filename_2 = 'ETS2.ets'
filename_3 = 'ETS3.ets'


@pytest.fixture
def ets_filenames():
    try:
        os.remove(filename)
        os.remove(compressed)
        os.remove(filename_1)
        os.remove(filename_2)
        os.remove(filename_3)
    except FileNotFoundError:
        pass
    yield [filename, filename_1, filename_2, filename_3, compressed]
    try:
        os.remove(filename)
        os.remove(compressed)
        os.remove(filename_1)
        os.remove(filename_2)
        os.remove(filename_3)
    except FileNotFoundError:
        pass


def test_ets_writer_raises_exception_if_trying_to_replace_existing_data_wo_overwrite_mode(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    out.write_samples(datas)
    out.close()
    out = ETSWriter(filename)
    with pytest.raises(estraces.ETSWriterError):
        out.write_samples(datas, index=0)
    out.close()
    out = ETSWriter(filename, overwrite=True)
    replacement = np.random.randint(100, size=(nb_trace - 5, nb_points), dtype=np.uint8)
    out.write_samples(replacement, index=0)
    ths = out.get_reader()
    assert np.array_equal(ths.samples[:nb_trace - 5], replacement)


def test_ets_writer_works_whatever_is_the_index_order(ets_filenames):
    base_trace = np.random.randint(0, 256, (nb_trace, nb_points), dtype='uint8')

    ets1 = ETSWriter(filename_1, overwrite=True)
    indexes = np.arange(nb_trace)
    for ind in indexes:
        ets1.write_samples(base_trace[ind], index=ind)
    ets1.close()

    ets2 = ETSWriter(filename_2, overwrite=True)

    for ind in reversed(indexes):
        ets2.write_samples(base_trace[ind], index=ind)
    ets2.close()

    ets3 = ETSWriter(filename_3, overwrite=True)
    indexes_3 = [nb_trace - 1, 2, 1]
    for ind in indexes_3:
        ets3.write_samples(base_trace[ind], index=ind)
    ets3.close()

    d1 = read_ths_from_ets_file(filename_1).samples[:]
    d2 = read_ths_from_ets_file(filename_2).samples[:]
    d3 = read_ths_from_ets_file(filename_3).samples[:]
    assert np.array_equal(d1[nb_trace - 1], d3[nb_trace - 1])
    assert np.array_equal(d1[2], d3[2])
    assert np.array_equal(d1[1], d3[1])

    assert d1.shape == d2.shape
    assert d1.shape == d3.shape

    s1 = os.path.getsize(filename_1)
    s2 = os.path.getsize(filename_2)

    assert s1 == s2


def test_write_ndarray(ets_filenames):
    out = ETSWriter(filename, overwrite=True)

    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(256, size=(nb_trace, 16), dtype=np.uint8)
    for index, data in enumerate(datas):
        out.write_samples(data, index=index)
        out.write_metadata('plaintext', plaintext[index], index=index)

    ths = out.get_reader()
    for i, t in enumerate(ths):
        assert np.array_equal(t.samples[:], datas[i])
        assert np.array_equal(t.plaintext, plaintext[i])


def test_write_2d_ndarray_metadata(ets_filenames):
    out = ETSWriter(filename, overwrite=True)

    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(256, size=(nb_trace, 16), dtype=np.uint8)
    for index, data in enumerate(datas):
        out.write_samples(data, index=index)
        out.write_metadata('plaintext', plaintext[index][None, :], index=index)

    ths = out.get_reader()
    for i, t in enumerate(ths):
        assert np.array_equal(t.samples[:], datas[i])
        assert np.array_equal(t.plaintext, plaintext[i])


def test_write_2d_ndarray_points(ets_filenames):
    out = ETSWriter(filename, overwrite=True)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    out.write_samples(datas)
    ths = out.get_reader()
    assert np.array_equal(ths.samples.array, datas)


def test_write_points_length_1(ets_filenames):
    out = ETSWriter(filename, overwrite=True)

    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    for index, data in enumerate(datas):
        out.write_samples(np.array([10]), index=index)
    ths = out.get_reader()
    assert np.array_equal(ths[0].samples[:], np.array([10]))


def test_write_meta_length_1(ets_filenames):
    out = ETSWriter(filename, overwrite=True)

    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    for index, data in enumerate(datas):
        out.write_samples(data, index=index)
        out.write_metadata('plaintext', 22, index=index)

    ths = out.get_reader()
    assert np.array_equal(ths[0].plaintext, np.array([22]))


def test_write_meta_string(ets_filenames):
    out = ETSWriter(filename, overwrite=True)

    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    for index, data in enumerate(datas):
        out.write_samples(data, index=index)
        out.write_metadata('plaintext', 'azerty', index=index)

    ths = out.get_reader()
    assert ths[0].plaintext == 'azerty'


def test_write_samples_1d(ets_filenames):
    out = ETSWriter(filename, overwrite=True)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    ths = estraces.formats.read_ths_from_ram(datas)
    out.add_samples(ths[0].samples)
    out.add_samples(ths[1].samples)
    ths_2 = out.get_reader()
    assert np.array_equal(ths[0].samples.array, ths_2[0].samples.array)
    assert np.array_equal(ths[1].samples.array, ths_2[1].samples.array)
    out.close()
    out = ETSWriter(filename)
    out.add_samples(ths[2].samples)
    ths_2 = out.get_reader()
    assert np.array_equal(ths[2].samples.array, ths_2[2].samples.array)


def test_add_samples_2d(ets_filenames):
    out = ETSWriter(filename, overwrite=True)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    ths = estraces.formats.read_ths_from_ram(datas)
    out.add_samples(ths.samples)
    ths_2 = out.get_reader()
    assert np.array_equal(ths.samples.array, ths_2.samples.array)


def test_add_samples_raise_exception_if_not_samples_instance(ets_filenames):
    out = ETSWriter(filename)
    with pytest.raises(TypeError):
        out.add_samples([1, 2, 3])


def test_add_samples_truncate_samples_to_first_inserted_data_size(ets_filenames):
    out = ETSWriter(filename, overwrite=True)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    ths = estraces.formats.read_ths_from_ram(datas)
    out.add_samples(ths.samples)
    ths_2 = estraces.formats.read_ths_from_ram(np.random.randint(100, size=(nb_trace, nb_points + 10), dtype=np.uint8))
    out.add_samples(ths_2.samples)
    ths_4 = estraces.formats.read_ths_from_ram(np.random.randint(100, size=(nb_trace, nb_points - 10), dtype=np.uint8))
    out.add_samples(ths_4.samples)
    ths_3 = out.get_reader()
    assert len(ths_3[0].samples) == nb_points


def test_writer_open_a_new_file(ets_filenames):
    out = ETSWriter(filename, overwrite=True)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    ths = estraces.formats.read_ths_from_ram(datas)
    out.add_samples(ths[0].samples)
    out.close()
    ths_2 = estraces.read_ths_from_ets_file(filename)
    assert np.array_equal(ths_2[0].samples.array, ths[0].samples.array)
    assert len(ths_2) == 1
    os.remove(filename)


def test_writer_open_an_existing_file_and_append_to(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    ths = estraces.formats.read_ths_from_ram(datas)
    out.add_samples(ths[0].samples)
    out.close()
    out = ETSWriter(filename)
    out.add_samples(ths[1].samples)
    out.close()
    ths_2 = estraces.read_ths_from_ets_file(filename)
    assert len(ths_2) == 2
    assert np.array_equal(ths_2.samples.array, ths[0:2].samples.array)
    os.remove(filename)


def test_writer_overwrite_an_existing_file(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    ths = estraces.formats.read_ths_from_ram(datas)
    out.add_samples(ths[0].samples)
    out.close()
    out = ETSWriter(filename, overwrite=True)
    out.add_samples(ths[1].samples)
    out.close()
    ths_2 = estraces.read_ths_from_ets_file(filename)
    assert len(ths_2) == 1
    assert np.array_equal(ths_2.samples.array, ths[1:2].samples.array)
    os.remove(filename)


def test_ets_writer_add_metadata_for_1_trace(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )

    out.add_samples(ths.samples)
    for i in range(nb_trace):
        out.add_metadata(ths[i].metadatas)
    out_ths = out.get_reader()
    assert np.array_equal(out_ths.plaintext, plaintext)
    assert np.array_equal(out_ths.ciphertext, ciphertext)
    assert np.array_equal(out_ths.chair, chair)


def test_ets_writer_add_metadata_for_several_traces(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )

    out.add_samples(ths.samples)
    out.add_metadata(ths.metadatas)
    out_ths = out.get_reader()
    assert np.array_equal(out_ths.plaintext, plaintext)
    assert np.array_equal(out_ths.ciphertext, ciphertext)
    assert np.array_equal(out_ths.chair.tolist(), chair.tolist())


def test_ets_writer_raises_exception_if_metadatas_is_not_proper_type(ets_filenames):
    out = ETSWriter(filename)
    with pytest.raises(TypeError):
        out.add_metadata({"dic": 1})


def test_ets_writer_add_metadata_with_inconsistent_sizes(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )

    out.add_samples(ths.samples)
    out.add_metadata(ths.metadatas)

    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 18), dtype='uint8')
    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )
    out.add_metadata(ths.metadatas)
    ths = out.get_reader()
    assert ths.plaintext.shape == (2 * nb_trace, 16)


def test_ets_writer_add_trace(ets_filenames):

    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )
    out.add_trace(ths[0])

    out_ths = out.get_reader()
    assert np.array_equal(out_ths[0].plaintext, plaintext[0])
    assert np.array_equal(out_ths[0].ciphertext, ciphertext[0])
    assert out_ths[0].chair == chair.tolist()[0]
    assert np.array_equal(out_ths[0].samples.array, datas[0])


def test_ets_writer_add_trace_raises_exception_if_trace_has_improper_types(ets_filenames):
    out = ETSWriter(filename)
    with pytest.raises(TypeError):
        out.add_trace({'not': 'a trace'})


def test_ets_writer_add_trace_raises_exception_if_metadata_are_inconsistent(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )
    out.add_trace_header_set(ths)

    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext_2 = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        plaintext_2=plaintext_2,
        ciphertext=ciphertext
    )
    with pytest.raises(estraces.ETSWriterError):
        out.add_trace(ths[0])


def test_ets_writer_add_trace_header_set(ets_filenames):

    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )
    new_meta = np.random.randint(0, 255, (len(ths), 1), dtype='uint8')
    ths.metadatas['new'] = new_meta
    out.add_trace_header_set(ths)
    out_ths = out.get_reader()
    assert np.array_equal(out_ths.plaintext, plaintext)
    assert np.array_equal(out_ths.ciphertext, ciphertext)
    assert out_ths.chair.tolist() == chair.tolist()
    assert np.array_equal(out_ths.samples.array, datas)
    assert np.array_equal(out_ths.new, new_meta)


def test_ets_writer_add_ths_raises_exception_if_ths_has_improper_types(ets_filenames):
    out = ETSWriter(filename)
    with pytest.raises(TypeError):
        out.add_trace_header_set({'not': 'a ths'})


def test_ets_writer_add_ths_raises_exception_if_metadata_are_inconsistent(ets_filenames):
    out = ETSWriter(filename)
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    chair = np.array(['abcd' for i in range(nb_trace)])
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        chair=chair,
        plaintext=plaintext,
        ciphertext=ciphertext
    )
    out.add_trace_header_set(ths)

    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext_2 = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        plaintext_2=plaintext_2,
        ciphertext=ciphertext
    )
    with pytest.raises(estraces.ETSWriterError):
        out.add_trace_header_set(ths)


def test_write_trace_with_new_metadata(ets_filenames):
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')
    ciphertext = np.random.randint(0, 256, (nb_trace, 16), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        plaintext=plaintext,
        ciphertext=ciphertext
    )

    for index, trace in enumerate(ths):
        trace.new_meta = 'oups'
        out = ETSWriter(filename)
        out.write_trace_object_and_points(trace, trace.samples[:], index)
    out.close()

    ths_2 = estraces.read_ths_from_ets_file(filename)

    assert ths_2[0].new_meta == 'oups'


def test_write_samples(ets_filenames):
    out = ETSWriter(filename, overwrite=True)

    datas = np.random.randint(100, size=(nb_trace, nb_points), dtype=np.uint8)
    plaintext = np.random.randint(256, size=(nb_trace, 16), dtype=np.uint8)
    for index, data in enumerate(datas):
        out.write_samples(data, index=index)
        out.write_metadata('plaintext', plaintext[index], index=index)

    ths = out.get_reader()
    for i, t in enumerate(ths):
        assert np.array_equal(t.samples[:], datas[i])
        assert np.array_equal(t.plaintext, plaintext[i])


def test_write_trace_with_scalar_metadata(ets_filenames):
    datas = np.random.randint(0, 256, size=(nb_trace, nb_points), dtype=np.uint8)
    scals = np.random.randint(0, 256, (nb_trace,), dtype='uint8')

    ths = estraces.formats.read_ths_from_ram(
        datas,
        scals=scals
    )

    out = ETSWriter(filename)
    out.add_trace_header_set(ths)

    ths_2 = out.get_reader()

    assert len(ths_2.scals) == nb_trace


def test_compress_ets_raises_exception_if_filename_not_provided(ets_filenames):
    with pytest.raises(TypeError):
        estraces.compress_ets(out_filename='foo')
    with pytest.raises(TypeError):
        estraces.compress_ets(filename='foo')


def test_compress_ets_raises_exception_on_bad_types(ets_filenames):
    with pytest.raises(TypeError):
        estraces.compress_ets(filename=1233, out_filename='foo')
    with pytest.raises(TypeError):
        estraces.compress_ets(filename='foo', out_filename=123)


def test_compress_ets_raises_exception_if_original_file_doesnt_exist(ets_filenames):
    with pytest.raises(AttributeError):
        estraces.compress_ets(filename='foo', out_filename='bar')


def test_compress_ets_raises_exception_if_original_outfile_exist(ets_filenames):
    ets = estraces.ETSWriter(filename)
    ets.add_trace_header_set(
        estraces.read_ths_from_ram(
            np.random.randint(0, 255, (100, 10), dtype='uint8')
        )
    )
    ets.close()
    with pytest.raises(ValueError):
        estraces.compress_ets(filename='foo', out_filename=filename)
    with pytest.raises(ValueError):
        estraces.compress_ets(filename=filename, out_filename=filename)


def test_compress_ets(ets_filenames):
    ets = estraces.ETSWriter(filename, overwrite=True)
    ths = estraces.read_ths_from_ram(
        np.random.randint(-55000, 55000, (nb_trace, nb_points), dtype='int32'),
        plaintext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        ciphertext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        chairs=np.array([f'chair{i}' for i in range(nb_trace)])
    )
    ets.add_trace_header_set(ths)
    ths = ets.get_reader()
    estraces.compress_ets(filename=filename, out_filename=compressed)
    ths_comp = estraces.read_ths_from_ets_file(compressed)

    assert np.array_equal(ths.samples[:], ths_comp.samples[:])
    assert np.array_equal(ths.plaintext, ths_comp.plaintext)
    assert np.array_equal(ths.ciphertext, ths_comp.ciphertext)
    assert np.array_equal(ths.chairs, ths_comp.chairs)
    assert os.path.getsize(filename) > 1.1 * os.path.getsize(compressed)


def test_write_headers(ets_filenames):
    ets = estraces.ETSWriter(filename, overwrite=True)
    ets.write_samples(np.random.randint(0, 255, (1000, 100), dtype='uint8'))
    ets.write_headers(_HEADERS)
    ths = ets.get_reader()
    assert np.array_equal(ths.headers['key'], _HEADERS['key'])
    assert ths.headers['foo'] == _HEADERS['foo']
    assert ths.headers['time'] == _HEADERS['time']


def test_add_trace_header_set_with_headers(ets_filenames):
    ets = estraces.ETSWriter(filename, overwrite=True)
    ths = estraces.read_ths_from_ram(
        np.random.randint(-55000, 55000, (nb_trace, nb_points), dtype='int32'),
        headers=_HEADERS,
        plaintext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        ciphertext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        chairs=np.array([f'chair{i}' for i in range(nb_trace)])
    )
    ets.add_trace_header_set(ths)
    ths = ets.get_reader()
    assert np.array_equal(ths.headers['key'], _HEADERS['key'])
    assert ths.headers['foo'] == _HEADERS['foo']
    assert ths.headers['time'] == _HEADERS['time']

    ets = estraces.ETSWriter(filename_1, overwrite=True)
    ets.add_trace_header_set(ths)
    ets.write_headers({'bar': 'bar'})
    ths = ets.get_reader()
    assert np.array_equal(ths.headers['key'], _HEADERS['key'])
    assert ths.headers['foo'] == _HEADERS['foo']
    assert ths.headers['time'] == _HEADERS['time']
    assert ths.headers['bar'] == 'bar'


def _get_virtual_memory_mock(v):
    def virtual_memory_wrapped():
        return 1, v
    return virtual_memory_wrapped


@patch("psutil.virtual_memory", side_effect=_get_virtual_memory_mock(20000000))
def test_get_optimal_batch_size_with_size_lower_available_memory(virtual_memory_func):
    ths = estraces.read_ths_from_ram(
        np.random.randint(-55000, 55000, (nb_trace, nb_points), dtype='int32'),
        headers=_HEADERS,
        plaintext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        ciphertext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        chairs=np.array([f'chair{i}' for i in range(nb_trace)])
    )

    batch_size = _get_optimal_batch_size(ths)

    assert batch_size > 1


@patch("psutil.virtual_memory", side_effect=_get_virtual_memory_mock(1))
def test_get_optimal_batch_size_with_size_higher_available_memory(virtual_memory_func):
    ths = estraces.read_ths_from_ram(
        np.random.randint(-55000, 55000, (nb_trace, nb_points), dtype='int32'),
        headers=_HEADERS,
        plaintext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        ciphertext=np.random.randint(0, 256, (nb_trace, 16), dtype='uint8'),
        chairs=np.array([f'chair{i}' for i in range(nb_trace)])
    )

    with pytest.raises(MemoryError):
        _get_optimal_batch_size(ths)
