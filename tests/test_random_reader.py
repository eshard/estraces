from .context import estraces  # noqa
import pytest
import numpy as np
from estraces.formats import random_reader


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


SEED = 101


@pytest.mark.parametrize('rand', [{'nb_traces': 2, 'trace_length': 2, 'seed': SEED}], indirect=True)
def test_fixed_seed(rand):
    sample1 = rand.samples[:]
    rand2 = random_reader.read_ths_from_random(nb_traces=2, trace_length=2, seed=SEED)
    assert np.array_equal(sample1, rand2.samples[:])
