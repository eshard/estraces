from .context import estraces  # noqa
import pytest
import numpy as np
from estraces.formats import random_reader
import cloudpickle


SMALL_SIZE = 10
BIG_SIZE = 9999999


@pytest.fixture
def rand(request):
    return random_reader.read_ths_from_random(**request.param)


@pytest.mark.parametrize('rand', [{'nb_traces': 0, 'trace_length': 0}], indirect=True)
def test_instanciate(rand):
    assert isinstance(rand, estraces.TraceHeaderSet)


@pytest.mark.parametrize('rand', [{'nb_traces': 0, 'trace_length': 0}], indirect=True)
def test_empty(rand):
    assert rand._reader.nb_traces == 0
    assert len(rand) == 0
    assert rand._reader.trace_length == 0


@pytest.mark.parametrize('rand', [{'nb_traces': SMALL_SIZE, 'trace_length': SMALL_SIZE}], indirect=True)
def test_small(rand):
    assert rand._reader.nb_traces == SMALL_SIZE
    assert len(rand) == SMALL_SIZE
    assert rand._reader.trace_length == SMALL_SIZE


@pytest.mark.parametrize('rand', [{'nb_traces': BIG_SIZE, 'trace_length': BIG_SIZE}], indirect=True)
def test_big(rand):
    assert rand._reader.nb_traces == BIG_SIZE
    assert len(rand) == BIG_SIZE
    assert rand._reader.trace_length == BIG_SIZE


def test_kwargtype_plainttext():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE,
                                               trace_length=SMALL_SIZE,
                                               plaintext=(16, 'uint8'))
    assert 'plaintext' in rand1.metadatas
    assert len(rand1.metadatas['plaintext'][0].tolist()) == 16
    assert isinstance(rand1.metadatas['plaintext'][0].tolist()[0], int)
    assert rand1.metadatas['plaintext'][0].dtype == 'uint8'


def test_kwargtype_shape():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE,
                                               trace_length=SMALL_SIZE,
                                               plaintext=(16, 'uint8'))
    assert rand1.metadatas['plaintext'].shape == (SMALL_SIZE, 16)


def test_missing_params_raise_type_error():
    samples = np.random.rand(10, 10)
    plaintext = (16, 'uint8')
    with pytest.raises(TypeError):
        random_reader.read_ths_from_random()
    with pytest.raises(TypeError):
        random_reader.read_ths_from_random(plaintext=plaintext)
    with pytest.raises(TypeError):
        random_reader.read_ths_from_random(samples[:10], plaintext=plaintext)
    with pytest.raises(TypeError):
        random_reader.read_ths_from_random(nb_traces=10, plaintext=plaintext)
    with pytest.raises(TypeError):
        random_reader.read_ths_from_random(trace_length=10, plaintext=plaintext)
    with pytest.raises(TypeError):
        random_reader.read_ths_from_random(trace_length='4445', plaintext=plaintext)


def test_incompatible_shapes():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE, trace_length=SMALL_SIZE, plaintext=(16, 'uint8'))
    assert rand1.samples[:].shape == (SMALL_SIZE, SMALL_SIZE)


def test_compatible_shapes_with_indexing():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE, trace_length=SMALL_SIZE, plaintext=(16, 'uint8'))
    assert rand1[:5].samples[:].shape == (5, SMALL_SIZE)


def test_incompatible_dtype():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE, trace_length=SMALL_SIZE, dtype='int16', plaintext=(16, 'uint8'))
    assert not rand1.samples[:].dtype == 'int8'


def test_compatible_default_dtype():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE, trace_length=SMALL_SIZE, plaintext=(16, 'uint8'))
    assert rand1.samples[:].dtype == 'int8'


def test_compatible_dtype():
    rand1 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE, trace_length=SMALL_SIZE, dtype='int16', plaintext=(16, 'uint8'))
    assert rand1.samples[:].dtype == 'int16'


@pytest.mark.parametrize('rand', [{'nb_traces': SMALL_SIZE, 'trace_length': SMALL_SIZE}], indirect=True)
def test_new_rand(rand):
    rand2 = random_reader.read_ths_from_random(nb_traces=SMALL_SIZE, trace_length=SMALL_SIZE)
    assert not np.array_equal(rand.samples[:], rand2.samples[:])


def test_indexing():
    rand = random_reader.read_ths_from_random(100, 1000, plaintext=(16, 'uint8'))
    assert len(rand[[3, 4, 5]]) == 3
    assert len(rand[np.array([10, 20, 15])]) == 3
    assert len(rand[10:20]) == 10
    assert len(rand[10:20:2]) == 5
    assert len(rand[90:]) == 10
    assert len(rand[:10]) == 10
    assert len(rand[:]) == 100


def test_indexing_2():
    rand = random_reader.read_ths_from_random(100, 1000, plaintext=(16, 'uint8'))
    assert rand._reader.fetch_samples(slice(None), slice(None)).shape == (100, 1000)
    assert rand[0:2].samples[:].shape == (2, 1000)
    assert rand[0:2].samples[:, ::2].shape == (2, 500)
    assert rand[0:2].samples[:, 100:].shape == (2, 900)
    assert rand.samples[:2, 100:].shape == (2, 900)
    assert rand[0:2].samples[:, 100:200].shape == (2, 100)


def test_multiple_indexing():
    rand = random_reader.read_ths_from_random(100, 1000, plaintext=(16, 'uint8'))
    sub1 = rand[:50]
    sub2 = sub1[40:60]
    assert len(sub1) == 50
    assert len(sub2) == 10
    assert sub1.samples[:].shape == (50, 1000)
    assert sub1.plaintext[:].shape == (50, 16)
    assert sub2.samples[:].shape == (10, 1000)
    assert sub2.plaintext[:].shape == (10, 16)


SEED = 101


@pytest.mark.parametrize('rand', [{'nb_traces': 2, 'trace_length': 2, 'seed': SEED}], indirect=True)
def test_fixed_seed(rand):
    sample1 = rand.samples[:]
    rand2 = random_reader.read_ths_from_random(nb_traces=2, trace_length=2, seed=SEED)
    assert np.array_equal(sample1, rand2.samples[:])


def test_serialization():
    ths = random_reader.read_ths_from_random(100, 1000, plaintext=(16, 'uint8'))
    dumped = cloudpickle.dumps(ths)
    loaded_ths = cloudpickle.loads(dumped)
    assert loaded_ths.samples[:].shape == ths.samples[:].shape
    assert loaded_ths.samples[:].dtype == ths.samples[:].dtype
    assert loaded_ths.plaintext[:].shape == ths.plaintext[:].shape
    assert loaded_ths.plaintext[:].dtype == ths.plaintext[:].dtype
