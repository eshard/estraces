from .context import estraces, patch_open, Counter  # noqa
import pytest
import numpy as np
from estraces.formats import bin_extractor, bin_format
import binascii
import cloudpickle


test_traces_fn_pattern = "./tests/samples/AESEncrypt.500MSs.cur.*"
test_trace_fn = "./tests/samples/AESEncrypt.500MSs.cur.00000"
test_trace_fn_pattern = "./tests/samples/AESEncrypt.500MSs.cur.0000*"
aes00000_100 = np.array([78, 82, 83, 87, 87, 88, 86, 90, 87, 87, 84, 89, 86, 86, 86, 88,
                         84, 86, 84, 87, 86, 85, 83, 88, 80, 78, 79, 80, 79, 77, 76, 79,
                         75, 76, 77, 78, 76, 74, 75, 73, 72, 72, 70, 70, 67, 69, 66, 69,
                         67, 73, 76, 80, 81, 87, 89, 94, 95, 100, 102, 105, 103, 105, 107, 105,
                         107, 105, 107, 106, 105, 103, 102, 105, 102, 104, 102, 102, 96, 97, 98, 97,
                         94, 97, 91, 94, 89, 92, 88, 90, 85, 85, 81, 87, 80, 80, 80, 76,
                         77, 74, 72, 70], dtype=np.uint8)
aes00000_last = np.array([49, 51, 56, 60, 63, 67, 73, 79, 80, 82, 85, 88, 89, 91, 90, 90,
                          89, 87, 92, 93, 88, 89, 87, 89, 86, 86, 85, 88, 81, 83, 81, 84,
                          78, 79, 78, 76, 74, 74, 74, 77, 72, 75, 71, 72, 70, 72, 68, 70,
                          67, 68, 66, 66, 61, 64, 61, 65, 61, 67, 69, 70, 71, 71, 75, 75,
                          78, 76, 76, 78, 77, 75, 76, 77, 78, 79, 82, 87, 89, 94, 95, 101,
                          103, 107, 111, 118, 119, 126, 128, 129, 131, 132, 138, 139, 138, 141, 139, 143,
                          144, 142, 145, 147], dtype=np.uint8)


METADATAS = {
    "my_property_2": bin_extractor.PatternExtractor(r"([0-9]{5})", num=0, unhexlify=False),
    "my_property_3": bytes([1, 10, 65]),  # bytes
    "my_property_4": np.array([4, 5, 6]),  # np.ndarray
    "my_property_5": (7, 8, 9),
}


@pytest.fixture
def ths():
    return bin_format.read_ths_from_bin_filenames_pattern(
        filename_pattern=test_traces_fn_pattern,
        dtype="uint8",
        metadatas_parsers=METADATAS,
        offset=0,
    )


@pytest.fixture
def ths_offset():
    return bin_format.read_ths_from_bin_filenames_pattern(
        filename_pattern=test_traces_fn_pattern,
        dtype="uint8",
        metadatas_parsers=METADATAS,
        offset=10,
    )


def test_bin_format_initialization_raises_error_with_improper_dtype():
    with pytest.raises(TypeError):
        bin_format.BinFormat(
            filenames=bin_format.get_sorted_filenames(pattern=test_traces_fn_pattern),
            dtype="dfjdlkfjdkljl",
            metadatas_parsers=METADATAS,
        )


def test_bin_format_initialization_raises_error_with_improper_filenames():
    with pytest.raises(TypeError):
        bin_format.BinFormat(
            filenames="ruieuirur", dtype="uint8", metadatas_parsers=METADATAS
        )


def test_empty_bin_format_returns_no_data():
    ths = bin_format.read_ths_from_bin_filenames_list(filenames_list=[], dtype='uint8', offset=0, metadatas_parsers={'meta1': lambda t: t})
    assert 0 == len(ths)
    assert (0,) == ths.samples.shape
    assert [('meta1', [])] == [(k, v.tolist()) for k, v in ths.metadatas.items()]


def test_bin_format_initialization():
    bin_engine = bin_format.BinFormat(
        filenames=bin_format.get_sorted_filenames(pattern=test_traces_fn_pattern),
        dtype="uint8",
        metadatas_parsers=METADATAS,
    )
    assert isinstance(bin_engine, bin_format.BinFormat)


def test_initialize_with_factory_function(ths):
    assert isinstance(ths, estraces.TraceHeaderSet)
    assert len(ths) == 100


def test_traces_attribute(ths):
    assert isinstance(ths.traces[0], estraces.Trace)
    assert len(ths.traces) == 100


def test_samples_slicing(ths, ths_offset):
    files = bin_format.get_sorted_filenames(pattern=test_traces_fn_pattern)
    for t in [ths, ths_offset]:
        for index, trc in enumerate(t.traces):
            filename = files[index]
            with open(filename, "rb") as fid:
                fid.seek(t._reader._offset)
                exp_data = fid.read()
                exp_data = np.frombuffer(exp_data, dtype="uint8")
                assert np.array_equal(trc.samples[:], exp_data)

    # Get with a frame
    for t in [ths, ths_offset]:
        for index, trc in enumerate(t.traces):
            filename = files[index]
            with open(filename, "rb") as fid:
                fid.seek(10 + t._reader._offset)
                exp_data = fid.read(40)
                exp_data = np.frombuffer(exp_data, dtype="uint8")
                assert np.array_equal(trc.samples[10:50], exp_data)


def test_samples_slicing_with_zero_length_slices(ths):
    assert isinstance(ths.samples[:, 100:100], np.ndarray)
    assert 0 == ths.samples[:, 100:100].shape[1]

    assert isinstance(ths.samples[100:100, :], np.ndarray)
    assert 0 == ths.samples[100:100, :].shape[0]

    assert isinstance(ths[0].samples[100:100], np.ndarray)
    assert 0 == ths.samples[100:100].shape[0]


def test_samples_slicing_with_negative_slices(ths):
    le = len(ths[0].samples)
    assert ths[0].samples[le - 200:le - 100].tolist() == ths[0].samples[-200: -100].tolist()
    assert [] == ths[0].samples[-100: -200].tolist()

    le = ths.samples.shape[1]
    assert ths.samples[:, le - 200: le - 100].tolist() == ths.samples[:, -200: -100].tolist()
    assert [[] for i in range(ths.samples.shape[0])] == ths.samples[:, -100: -200].tolist()

    le = ths.samples.shape[0]
    assert ths.samples[le - 2:le - 1, :].tolist() == ths.samples[-2: -1, :].tolist()
    assert [] == ths.samples[-1:-2, :].tolist()


def test_ths_slicing(ths):

    assert isinstance(ths[2], estraces.Trace)
    with pytest.raises(IndexError):
        ths[12:12, 0]
        ths["jfkj"]

    ths2 = ths[10:15]
    ths3 = ths[[1, 3]]
    ths4 = ths[:15]
    ths5 = ths[5:]
    assert isinstance(ths2, estraces.TraceHeaderSet)
    assert isinstance(ths3, estraces.TraceHeaderSet)
    assert isinstance(ths4, estraces.TraceHeaderSet)
    assert isinstance(ths5, estraces.TraceHeaderSet)

    assert len(ths2) == 5
    assert len(ths3) == 2
    assert len(ths4) == 15
    assert len(ths5) == len(ths) - 5

    assert id(ths) != id(ths2)

    origin_traces = ths.traces[10:15]
    for idx, trc in enumerate(ths2.traces):
        origin_trc = origin_traces[idx]
        assert id(trc) != id(origin_trc)
        assert trc.samples.tolist() == origin_trc.samples.tolist()
        for k, v in trc.metadatas.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, origin_trc.metadatas[k])
            else:
                assert v == origin_trc.metadatas[k]

    for idx, trc in enumerate(ths3.traces):
        o_ind = 1 if idx == 0 else 3
        origin_trc = ths.traces[o_ind]
        assert id(trc) != id(origin_trc)
        assert trc.samples.tolist() == origin_trc.samples.tolist()
        for k, v in trc.metadatas.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, origin_trc.metadatas[k])
            else:
                assert v == origin_trc.metadatas[k]


def test_bin_format_samples_api(ths, ths_offset):
    data = ths.samples[0, :]
    assert np.array_equal(data[0:100], aes00000_100)

    data = ths.samples[0, 10:20]
    assert data.ndim == 1
    assert np.array_equal(data, aes00000_100[10:20])

    data = ths.samples[0, [1, 3, 4, 9]]
    assert data.ndim == 1
    assert np.array_equal(data, aes00000_100[[1, 3, 4, 9]])

    # With a too large slice via indexing
    data = ths.samples[0, 99900:200000]
    assert np.array_equal(data, aes00000_last)

    # Requesting a single point via indexing (getitem)
    data = ths.samples[0, 7]
    assert np.array_equal(data, aes00000_100[7])

    # Requesting a too large single point via indexing (getitem)
    with pytest.raises(IndexError):
        ths.samples[0, 700000]

    # With a wrong index type
    with pytest.raises(IndexError):
        ths.samples[0, "abcd"]

    # With an offset
    data = ths_offset.samples[0]
    assert len(data) == 99990
    assert np.array_equal(data[0:90], aes00000_100[10:])

    trc = ths[0]
    assert aes00000_100[12] == trc.samples[12]

    assert isinstance(trc.samples[12], np.uint8)

    assert np.array_equal(
        aes00000_100[1:10:2],
        trc.samples[1:10:2]
    )


def test_metadatas(ths):

    prop2 = ths.my_property_2
    prop3 = ths.my_property_3
    prop4 = ths.my_property_4
    # Get a wrong attribute
    with pytest.raises(AttributeError):
        ths.fake

    for trc_idx in range(len(ths)):
        assert "{:05d}".format(trc_idx) == prop2[trc_idx]
        assert bytes([1, 10, 65]) == prop3[trc_idx]
        assert np.array_equal(np.array([4, 5, 6]), prop4[trc_idx])

    assert isinstance(ths.metadatas, estraces.traces.metadatas.Metadatas)
    assert ths.metadatas["my_property_2"].tolist() == ths.my_property_2.tolist()
    assert ths.metadatas["my_property_3"].tolist() == ths.my_property_3.tolist()


def test_bin_format_fetch_samples_always_returns_2d_array(ths):
    assert 2 == ths._reader.fetch_samples(traces=[1, 2, 3], frame=...).ndim
    assert 2 == ths._reader.fetch_samples(traces=range(len(ths._reader)), frame=1).ndim
    assert 2 == ths._reader.fetch_samples(traces=[1], frame=slice(10, 20, 1)).ndim
    assert 2 == ths._reader.fetch_samples(traces=[1], frame=slice(10, 10, 1)).ndim
    assert 2 == ths._reader.fetch_samples(traces=[1], frame=1).ndim
    assert 2 == ths._reader.fetch_samples(traces=[1, 4], frame=1).ndim


def test_direct_value():
    """Verify that we handle any kind of input from the user for plain, cipher and key."""
    expect = b'\x03\xd20\xbf\x92o\xd8\xf5'
    in_1 = "03D230BF926FD8F5"
    in_2 = [3, 210, 48, 191, 146, 111, 216, 245]

    out_1 = bin_extractor.DirectValue(in_1)
    assert expect == out_1.get_text(None)
    out_2 = bin_extractor.DirectValue(in_2)
    assert expect == out_2.get_text(None)


def test_direct_value_wrong_inputs():
    """Verify that we handle any kind of input from the user for plain, cipher and key."""
    in_1 = 0x03D230BF926FD8F5
    in_2 = "0x03D230BF926FD8F5"
    in_3 = "03D230BF926FD8F57"
    in_4 = [0, 1, 255, 257]

    with pytest.raises(ValueError) as exc:
        bin_extractor.DirectValue(in_1)
    assert str(exc.value) == "Invalid value format <class 'int'>. HexString without the '0x' prefix or a list of integer in range(0, 256) is expected."

    with pytest.raises(ValueError) as exc:
        bin_extractor.DirectValue(in_2)
    assert str(exc.value) == "Invalid character in input string."
    with pytest.raises(ValueError) as exc:
        bin_extractor.DirectValue(in_3)
    assert str(exc.value) == "Odd-length string."
    with pytest.raises(ValueError) as exc:
        bin_extractor.DirectValue(in_4)
    assert str(exc.value) == "bytes must be in range(0, 256)"


def test_header_extractor():
    filename = "tests/samples/DESi386__03D230BF926FD8F5_DB0F84C83D856553.bin"
    key = bin_extractor.HeaderExtractor(start=0, count=8)
    cipher = bin_extractor.HeaderExtractor(start=8, count=8)
    k = key.get_text(filename)
    c = cipher.get_text(filename)

    assert k == binascii.unhexlify('1c0000001cec7ff6')
    assert c == binascii.unhexlify('50cc7ef600e07ff6')


def test_header_extractor_wrong_inputs():
    # No arguments
    with pytest.raises(ValueError) as exc:
        bin_extractor.HeaderExtractor()
    assert str(exc.value) == "No 'start' offset specified."

    # No end or count specified
    with pytest.raises(ValueError) as exc:
        bin_extractor.HeaderExtractor(start=0)
    assert str(exc.value) == "No 'end' offset or 'count' specified."

    # Both end AND count specified
    with pytest.raises(ValueError) as exc:
        bin_extractor.HeaderExtractor(start=0, count=10, end=12)
    assert str(exc.value) == "Use either 'end' or 'count' option, not both."

    # Wrong start value
    with pytest.raises(ValueError) as exc0:
        bin_extractor.HeaderExtractor(start=-332, count=10)
    with pytest.raises(ValueError) as exc1:
        bin_extractor.HeaderExtractor(start=1.2, count=10)
    with pytest.raises(ValueError) as exc2:
        bin_extractor.HeaderExtractor(start=1, count=-2)
    with pytest.raises(ValueError) as exc3:
        bin_extractor.HeaderExtractor(start=1, count=1.2)
    with pytest.raises(ValueError) as exc4:
        bin_extractor.HeaderExtractor(start=1, end=-2)
    with pytest.raises(ValueError) as exc5:
        bin_extractor.HeaderExtractor(start=1, end=1.2)

    assert 'must be positive integer' in str(exc0.value)
    assert 'must be positive integer' in str(exc1.value)
    assert 'must be positive integer' in str(exc2.value)
    assert 'must be positive integer' in str(exc3.value)
    assert 'must be positive integer' in str(exc4.value)
    assert 'must be positive integer' in str(exc5.value)

    # Incoherent start, end values
    with pytest.raises(ValueError) as exc0:
        bin_extractor.HeaderExtractor(start=0, count=0)
    with pytest.raises(ValueError) as exc1:
        bin_extractor.HeaderExtractor(start=0, end=0)
    assert str(exc0.value) == 'The end offset must be greater than the start offset.'
    assert str(exc1.value) == 'The end offset must be greater than the start offset.'

    # Specified both cound and end arguments
    with pytest.raises(ValueError) as exc0:
        bin_extractor.HeaderExtractor(start=0, end=1, count=1)
    with pytest.raises(ValueError) as exc1:
        bin_extractor.HeaderExtractor(start=0, end=0, count=0)
    assert str(exc0.value) == "Use either 'end' or 'count' option, not both."
    assert str(exc1.value) == "Use either 'end' or 'count' option, not both."


def test_pattern_extractor():
    filename = "wave_aist-aes-agilent_2009-12-30_22-53-19_k=0000000000000003243f6a8885a308d3_m=0000000000000002b7e151628aed2a6a_c=e6a636e30c85f35e980f3546a04daff7.bin"  # noqa
    key = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{32})", num=0)
    msg = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{32})", num=1)
    cipher = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{32})", num=2)
    test_0 = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{16})")
    test_1 = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{16})", num=0)
    test_2 = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{16})", num=1)
    test_3 = bin_extractor.PatternExtractor(pattern=r"([A-Fa-f0-9]{35})")

    k = key.get_text(filename)
    m = msg.get_text(filename)
    c = cipher.get_text(filename)

    t0 = test_0.get_text(filename)
    t1 = test_1.get_text(filename)
    t2 = test_2.get_text(filename)

    assert k == binascii.unhexlify('0000000000000003243f6a8885a308d3')
    assert m == binascii.unhexlify('0000000000000002b7e151628aed2a6a')
    assert c == binascii.unhexlify('e6a636e30c85f35e980f3546a04daff7')
    assert t0 == binascii.unhexlify('0000000000000003')
    assert t1 == binascii.unhexlify('0000000000000003')
    assert t2 == binascii.unhexlify('243f6a8885a308d3')

    with pytest.raises(ValueError) as exc:
        test_3.get_text(filename)
    assert str(exc.value) == "Pattern '([A-Fa-f0-9]{35})' not found in 'wave_aist-aes-agilent_2009-12-30_22-53-19_k=0000000000000003243f6a8885a308d3_m=0000000000000002b7e151628aed2a6a_c=e6a636e30c85f35e980f3546a04daff7.bin'."  # noqa


def test_file_pattern_extractor():
    """Verify that the 5 first messages are what we expect."""
    expected = ['D9EDDB17D7B1B4F406685DA69CA71DA1',
                '8384F00F9869555D3D1401E9679A4A6B',
                '105FDD49C7EF0961BB993263B48ECF34',
                '9A870CAC7B9C38E99A1E88DA9BAE02D4',
                'E87303E1BE987A9CE1FB6B16F460AD14']
    msg = []
    tmp = bin_extractor.FilePatternExtractor(
        "tests/samples/plain.txt", pattern=r"([A-Fa-f0-9]{32})", replace=r"\1")
    for _ in range(5):
        msg.append(tmp.get_text(None))

    for it in enumerate(expected):
        m = binascii.unhexlify(it[1])
        assert m == msg[it[0]]


def test_file_pattern_extractor_wrong_inputs():
    """Verify that error are raised with wrong user inputs."""
    # Wrong file path
    with pytest.raises(FileNotFoundError) as exc:
        bin_extractor.FilePatternExtractor("tests/samples/plainFAKE.txt", pattern=r"([A-Fa-f0-9]{32})", replace=r"\1")
    assert str(exc.value) == "'tests/samples/plainFAKE.txt' not found"


def test_file_pattern_extractor_long():
    # Extract the meta data from the same ascii files
    file = 'tests/samples/metadatas.txt'
    key = bin_extractor.FilePatternExtractor(file, r"([A-Fa-f0-9]{32})", num=0)
    plain = bin_extractor.FilePatternExtractor(file, r"([A-Fa-f0-9]{32})", num=1)
    cipher = bin_extractor.FilePatternExtractor(file, r"([A-Fa-f0-9]{32})", num=2)
    mask1 = bin_extractor.FilePatternExtractor(file, r"([A-Fa-f0-9]{32})", num=3)
    mask2 = bin_extractor.FilePatternExtractor(file, r"([A-Fa-f0-9]{32})", num=4)
    mask3 = bin_extractor.FilePatternExtractor(file, r"([A-Fa-f0-9]{32})", num=5)

    ths = bin_format.read_ths_from_bin_filenames_pattern(
        filename_pattern=test_traces_fn_pattern,
        dtype='uint8',
        offset=0,
        metadatas_parsers={
            'key': key,
            'plain': plain,
            'cipher': cipher,
            'mask1': mask1,
            'mask2': mask2,
            'mask3': mask3
        }
    )

    with patch_open(Counter()) as counter:
        ths.metadatas['plain']
        ths.metadatas['mask3']
        ths.metadatas['cipher']
        ths.metadatas['key']
        ths.metadatas['mask1']
        ths.metadatas['mask2']
        ths.plain
        ths.mask3
        ths.cipher
        ths.key
        ths.mask1
        ths.mask2
        trc = ths[0]
        trc.plain
        trc.mask3
        trc.cipher
        trc.key
        trc.mask1
        trc.mask2
        for v in counter.counts.values():
            assert v < 7  # One file open for each meta data

    with patch_open(Counter()) as counter:
        ths.metadatas['plain']
        ths_2 = ths[0:50]
        ths_2.plain
        for v in counter.counts.values():
            assert v < 2


@pytest.fixture(params=[bin_format.BinFormat, bin_format.read_ths_from_bin_filenames_list])
def bin_factory(request):
    return request.param


def test_bin_reader_raises_exception_if_invalid_padding_mode(bin_factory):
    with pytest.raises(AttributeError):
        bin_factory([test_trace_fn], dtype='uint8', metadatas_parsers={}, offset=0, padding_mode='TRU')


def test_ths_bin_pattern_factory_raises_exception_if_invalid_padding_mode():
    with pytest.raises(AttributeError):
        bin_format.read_ths_from_bin_filenames_pattern('./tests/samples/*difflength.bin', dtype='uint8', metadatas_parsers={}, padding_mode='TRU')


def test_default_bin_reader_raise_exception_at_init_with_padding_mode_default_and_traces_not_of_the_same_size(bin_factory):
    filenames = bin_format.get_sorted_filenames('./tests/samples/*difflength.bin')
    with pytest.raises(ValueError):
        bin_factory(filenames, dtype='uint8', metadatas_parsers={})


def test_default_pattern_ths_factory_raise_exception_at_init_with_padding_mode_default_and_traces_not_of_the_same_size(bin_factory):
    with pytest.raises(ValueError):
        bin_format.read_ths_from_bin_filenames_pattern('./tests/samples/*difflength.bin', dtype='uint8', metadatas_parsers={})


def test_bin_reader_with_truncate_padding_mode_truncate_too_long_traces():
    filenames = bin_format.get_sorted_filenames('./tests/samples/*difflength.bin')
    binf = bin_format.BinFormat(filenames, 'uint8', {}, padding_mode=bin_format.PaddingMode.TRUNCATE)
    for i in range(3):
        assert 990 == binf.get_trace_size(i)
        assert 990 == binf.fetch_samples(traces=[i], frame=None).shape[1]
    assert 2 == len(binf[:2])


def test_bin_reader_with_pad_padding_mode_pad_too_short_traces():
    filenames = bin_format.get_sorted_filenames('./tests/samples/*difflength.bin')
    binf = bin_format.BinFormat(filenames, 'uint8', {}, padding_mode=bin_format.PaddingMode.PAD)
    for i in range(3):
        assert 1010 == binf.get_trace_size(i)
        assert 1010 == binf.fetch_samples(traces=[i], frame=None).shape[1]
    assert [0] * 10 == binf.fetch_samples(traces=[0], frame=None)[0, -10:].tolist()
    assert [0] * 20 == binf.fetch_samples(traces=[2], frame=None)[0, -20:].tolist()
    assert 2 == len(binf[:2])


def test_serialization(ths):
    dumped = cloudpickle.dumps(ths)
    loaded_ths = cloudpickle.loads(dumped)
    assert np.array_equal(loaded_ths.samples[:], ths.samples[:])
    assert np.array_equal(loaded_ths.my_property_2[:], ths.my_property_2[:])
    assert np.array_equal(loaded_ths.my_property_3[:], ths.my_property_3[:])
    assert np.array_equal(loaded_ths.my_property_4[:], ths.my_property_4[:])
    assert np.array_equal(loaded_ths.my_property_5[:], ths.my_property_5[:])
