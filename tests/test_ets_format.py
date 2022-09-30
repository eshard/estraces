from .context import estraces # noqa
import pytest
import shutil
import os
import numpy as np
import timeit
import memory_profiler
import h5py
from estraces.formats import ets_format
from pathlib import Path
import cloudpickle

path = 'tests/samples/test.ets'


@pytest.fixture
def ets1():
    return ets_format.read_ths_from_ets_file(filename=path)


def test_open_ets_file(ets1):
    assert isinstance(ets1, estraces.TraceHeaderSet)
    assert 200 == len(ets1)


def test_open_ets_file_with_path():
    ets1 = ets_format.read_ths_from_ets_file(filename=Path(path))
    assert isinstance(ets1, estraces.TraceHeaderSet)
    assert 200 == len(ets1)


def test_read_from_ets_raise_exception_if_file_does_not_exist():
    with pytest.raises(AttributeError):
        ets_format.read_ths_from_ets_file(filename='doesnotexist')


def test_read_from_incorrect_ets_file_raises_type_error():
    with pytest.raises(TypeError):
        ets_format.read_ths_from_ets_file(filename='tests/samples/AESEncrypt.500MSs.cur.00000')


def test_read_metadatas(ets1):
    assert isinstance(ets1.metadatas, estraces.traces.metadatas.Metadatas)
    for v in ets1.metadatas.keys():
        assert v in ['ciphertext', 'foo_bar', 'plaintext']
    assert [2, 9, 15, 8, 11, 8, 2, 13] == ets1.metadatas['plaintext'][0].tolist()
    assert [2, 9, 15, 8, 11, 8, 2, 13] == ets1.plaintext[0].tolist()
    assert [2, 9, 15, 8, 11, 8, 2, 13] == ets1[0].plaintext.tolist()
    for _, v in ets1.metadatas.items():
        assert 2 == v.ndim
    for _, v in ets1[0].metadatas.items():
        assert 1 == v.ndim
    assert [
        [3, 10, 16, 9, 12, 9, 3, 14],
        [4, 11, 17, 10, 13, 10, 4, 15],
        [5, 12, 18, 11, 14, 11, 5, 16]
    ] == ets1.plaintext[1: 4].tolist()


def test_read_samples(ets1):
    assert isinstance(ets1.samples, estraces.traces.samples.Samples)
    assert ets1.samples.ndim == 2
    assert isinstance(ets1[0].samples, estraces.traces.samples.Samples)
    assert ets1[0].samples.ndim == 1
    assert [
        [2, 11, 21],
        [4, 13, 23],
        [6, 15, 25]
    ] == ets1.samples[1: 6: 2, [1, 10, 20]].tolist()

    assert [
        [2, 11, 21],
        [4, 13, 23],
        [6, 15, 25]
    ] == ets1.samples[1: 6: 2, np.array([1, 10, 20])].tolist()

    assert [
        [2, 3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11, 12],
        [6, 7, 8, 9, 10, 11, 12, 13, 14]
    ] == ets1.samples[1: 6: 2, 1:10].tolist()

    # Manage limits
    assert 0 == ets1.samples[0, 0]
    assert [1, 2, 3, 4, 5, 6, 7, 8, 9] == ets1.samples[0, 1:10].tolist()
    assert [1, 2, 3, 4, 5, 6, 7, 8, 9] == ets1.samples[1:10, 0].tolist()

    assert [1, 3, 4, 7, 9] == ets1[0].samples[np.array([1, 3, 4, 7, 9])].tolist()

    # H5 doesn't manage unordered or dedup list, arrays in constrast to ndarray.
    assert [1, 4, 3, 7, 4, 9] == ets1[0].samples[np.array([1, 4, 3, 7, 4, 9])].tolist()

    assert list(range(33)) == ets1[0].samples[...].tolist()
    assert 12 == ets1[0].samples[12]

    # Test memory overhead optimization strategies
    assert ets1.samples[:, :][:, [0, 32, 28, 0]].tolist() == ets1.samples[:, [0, 32, 28, 0]].tolist()


def test_read_samples_with_range(ets1):
    assert [
        [2, 3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11, 12],
        [6, 7, 8, 9, 10, 11, 12, 13, 14]
    ] == ets1.samples[range(1, 6, 2), range(1, 10)].tolist()


def test_slice_ets(ets1):
    sub1 = ets1[2: 7]
    sub2 = ets1[[1, 10, 12]]
    assert isinstance(sub1, estraces.TraceHeaderSet)
    assert isinstance(sub2, estraces.TraceHeaderSet)
    assert 5 == len(sub1)
    assert 3 == len(sub2)
    assert [2, 3, 4, 5] == sub1.samples[0, 0:4].tolist()
    assert [6, 7, 8, 9] == sub1.samples[4, 0:4].tolist()
    assert [1, 2, 3, 4] == sub2.samples[0, 0:4].tolist()
    assert [10, 11, 12, 13] == sub2.samples[1, 0:4].tolist()
    assert [12, 13, 14, 15] == sub2.samples[2, 0:4].tolist()


def bench_func(ets):
    for t in ets:
        t.plaintext


def ref_func(ets):
    plaintext = ets.plaintext[:]
    for t in range(len(ets)):
        plaintext[t]


ets2 = ets_format.read_ths_from_ets_file(filename=path)
ets3 = ets_format.read_ths_from_ets_file(filename=path)


def test_fetch_samples_execution_time(ets1):
    # See issue https://gitlab.eshard.int/esdynamic/estraces/issues/26
    # Here we mesure execution time of bench_func versus an arbitrary ref time given by ref_func execution.
    # Initial ratio before issue #26 ~175
    t_ref = timeit.timeit('ref_func(ets2)', number=100, globals=globals())
    t = timeit.timeit('bench_func(ets3)', number=100, globals=globals())
    assert 200 > abs(t - t_ref) / t_ref


PATH = 'tests/samples/'


@pytest.fixture
def ets_filename():
    fn = 'big_ets'
    ets_filename = f'{PATH}{fn}.ets'
    ets_filename_bck = f'{PATH}{fn}.ets.bck'
    shutil.copyfile(ets_filename, ets_filename_bck)
    num_traces = 50000
    len_trc = 1000
    plains = np.array([np.random.randint(0, 256, 32, dtype="uint8") for i in range(num_traces)])
    datas = np.vstack([np.random.rand(len_trc) for i in range(num_traces)])

    file = h5py.File(name=ets_filename, mode='w')
    file.create_group('metadata')
    file.create_dataset('traces', dtype='float64', shape=(num_traces, len_trc))
    file['metadata'].create_dataset('plaintext', dtype='uint8', shape=(num_traces, 32))
    for i in range(num_traces):
        file['traces'][i] = datas[i]
        file['metadata']['plaintext'][i] = plains[i]
    file.flush()
    file.close()
    yield ets_filename
    shutil.copyfile(ets_filename_bck, ets_filename)


def test_ths_slicing_memory_consumption(ets_filename):
    # See issue https://gitlab.eshard.int/esdynamic/estraces/issues/30
    # Slicing a ths based on ets file without direct acces to data lead to important memory consumptions.
    # For example ths[0] or ths[0:100] can lead to the total size of the ets consummed in memory.
    # Here we test that slicing a ths with an integer or slice without access to the samples
    # doesn't load too much data === at most 10% memory consumption of the ths.samples reading operation.
    # We also ensure that slicing or getting only one trace has the same memory footprint.

    def tested_trace():
        ths = estraces.read_ths_from_ets_file(ets_filename)
        ths[0]

    trace_footprint = max(memory_profiler.memory_usage(tested_trace))

    def tested_slice():
        ths = estraces.read_ths_from_ets_file(ets_filename)
        ths[:10]

    slice_footprint = max(memory_profiler.memory_usage(tested_slice))

    def bgd():
        estraces.read_ths_from_ets_file(ets_filename)

    bgd_footprint = max(memory_profiler.memory_usage(bgd))

    def ref():
        ths = estraces.read_ths_from_ets_file(ets_filename)
        ths[:].samples[:]

    ref_footprint = max(memory_profiler.memory_usage(ref))
    assert abs(trace_footprint - bgd_footprint) / abs(ref_footprint - bgd_footprint) < 0.1, 'Memory consumption of trace selection is too high'
    assert abs(slice_footprint - bgd_footprint) / abs(ref_footprint - bgd_footprint) < 0.1, 'Memory consumption of ths slicing is too high'
    assert abs((trace_footprint - slice_footprint) / (slice_footprint)) < 0.01, 'Memory consumption of ths slicing and trace selection should be equivalent'


def test_split_preserves_length_consistency_of_sub_ths(ets1):
    parts = ets1.split(100)
    for _, p in enumerate(parts):
        assert len(p) in (100, 56)
        assert len(p) == p.samples[:].shape[0]


@pytest.fixture
def ets_with_name():
    data = np.random.randint(0, 255, (1000, 2000), dtype='uint8')
    name = np.array(['a great name' for name in range(1000)])
    _id = np.array([i for i in range(1000)], dtype='uint')
    ths = estraces.read_ths_from_ram(data, name=name, id=_id)
    out = estraces.ETSWriter('ets_with_name.ets')
    out.add_trace_header_set(ths)
    try:
        yield out.get_reader()
    finally:
        os.remove('ets_with_name.ets')


def test_ets_with_name_and_id_metadata(ets_with_name):
    assert ets_with_name.name[0] == 'a great name'
    assert len(ets_with_name.name) == 1000
    assert ets_with_name[0].name == 'a great name'
    assert len(ets_with_name.id) == 1000
    assert ets_with_name.id[0] == 0
    assert ets_with_name[100].id == 100


def test_headers():
    ets = ets_format.read_ths_from_ets_file('tests/samples/test_with_headers.ets')
    headers_dict = {'foo': 'bar', 'bar': 42, 'list': [1, 2, 3], 'ndarray': np.arange(5, dtype='int8')}
    assert ets.headers == headers_dict


def test_serialization(ets1):
    ths = ets1
    dumped = cloudpickle.dumps(ths)
    loaded_ths = cloudpickle.loads(dumped)
    assert np.array_equal(loaded_ths.samples[:], ths.samples[:])
    assert np.array_equal(loaded_ths.plaintext[:], ths.plaintext[:])
    assert np.array_equal(loaded_ths.foo_bar[:], ths.foo_bar[:])
    assert np.array_equal(loaded_ths.ciphertext[:], ths.ciphertext[:])


@pytest.mark.parametrize('ets_file', ['tests/samples/test_old.ets', 'tests/samples/test_new.ets'])
def test_read_with_strings(ets_file):
    ets = ets_format.read_ths_from_ets_file(ets_file)
    for metadata_name in ['string', 'variable_length_string', 'ascii_bytes', 'variable_length_ascii_bytes']:
        print(metadata_name, end=' ')
        metadata = getattr(ets, metadata_name)
        for value in metadata:
            assert isinstance(value, str)
        print('is of type ', type(value))
