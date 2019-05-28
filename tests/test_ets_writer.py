from .context import estraces  # noqa
import unittest
import numpy as np
import os
from estraces.formats.ets_writer import ETSWriter
from estraces import read_ths_from_ets_file


class TestETSWriter(unittest.TestCase):
    nb_trace = 10
    nb_points = 10
    filename = 'ETS-test.ets'
    filename_1 = 'ETS1.ets'
    filename_2 = 'ETS2.ets'
    filename_3 = 'ETS3.ets'

    def setUp(self):
        try:
            os.remove(self.filename)
            os.remove(self.filename_1)
            os.remove(self.filename_2)
            os.remove(self.filename_3)
        except FileNotFoundError:
            pass

    def test_ets_writer_works_whatever_is_the_index_order(self):
        # Tests for issue https://gitlab.eshard.int/side-channel/estoolkit/issues/237.
        # ETSWriter doesn't behave properly if ETSWriter.write_points is called several times with unordered index.

        base_trace = np.random.randint(0, 256, (3000))

        ets1 = ETSWriter(self.filename_1, overwrite=True)
        indexes = np.arange(5)
        for ind in indexes:
            ets1.write_points(base_trace, index=ind)
        ets1.close()

        ets2 = ETSWriter(self.filename_2, overwrite=True)

        for ind in reversed(indexes):
            ets2.write_points(base_trace, index=ind)
        ets2.close()

        ets3 = ETSWriter(self.filename_3, overwrite=True)
        indexes_3 = [4, 2, 1]
        for ind in indexes_3:
            ets3.write_points(base_trace, index=ind)
        ets3.close()

        d1 = read_ths_from_ets_file(self.filename_1).samples[:]
        d2 = read_ths_from_ets_file(self.filename_2).samples[:]
        d3 = read_ths_from_ets_file(self.filename_3).samples[:]
        self.assertEqual(d1.shape, d2.shape)
        self.assertEqual(d1.shape, d3.shape)

        self.assertTrue(np.array_equal(d1[2], d3[2]))

        s1 = os.path.getsize(self.filename_1)
        s2 = os.path.getsize(self.filename_2)

        self.assertEqual(s1, s2)

    def test_write_ndarray(self):
        out = ETSWriter(self.filename, overwrite=True)

        datas = np.random.randint(100, size=(self.nb_trace, self.nb_points), dtype=np.uint8)
        plaintext = np.random.randint(256, size=(self.nb_trace, 16), dtype=np.uint8)
        for index, data in enumerate(datas):
            out.write_points(data, index=index)
            out.write_meta(tag='plaintext', metadata=plaintext[index], index=index)

        ths = out.get_reader()
        for i, t in enumerate(ths):
            self.assertTrue(np.array_equal(t.samples[:], datas[i]))
            self.assertTrue(np.array_equal(t.plaintext, plaintext[i]))

    def test_write_2d_ndarray_metadata(self):
        out = ETSWriter(self.filename, overwrite=True)

        datas = np.random.randint(100, size=(self.nb_trace, self.nb_points), dtype=np.uint8)
        plaintext = np.random.randint(256, size=(self.nb_trace, 16), dtype=np.uint8)
        for index, data in enumerate(datas):
            out.write_points(data, index=index)
            out.write_meta(tag='plaintext', metadata=plaintext[index][None, :], index=index)

        ths = out.get_reader()
        for i, t in enumerate(ths):
            self.assertTrue(np.array_equal(t.samples[:], datas[i]))
            self.assertTrue(np.array_equal(t.plaintext, plaintext[i]))

    def test_write_2d_ndarray_points(self):
        out = ETSWriter(self.filename, overwrite=True)

        datas = np.random.randint(100, size=(self.nb_trace, self.nb_points), dtype=np.uint8)
        for index, data in enumerate(datas):
            with self.assertRaises(Exception):
                out.write_points(data[None, :], index=index)

    def test_write_points_length_1(self):
        out = ETSWriter(self.filename, overwrite=True)

        datas = np.random.randint(100, size=(self.nb_trace, self.nb_points), dtype=np.uint8)
        for index, data in enumerate(datas):
            out.write_points(np.array([10]), index=index)
        ths = out.get_reader()
        self.assertTrue(np.array_equal(ths[0].samples[:], np.array([10])))

    def test_write_meta_length_1(self):
        out = ETSWriter(self.filename, overwrite=True)

        datas = np.random.randint(100, size=(self.nb_trace, self.nb_points), dtype=np.uint8)
        for index, data in enumerate(datas):
            out.write_points(data, index=index)
            out.write_meta(tag='plaintext', metadata=22, index=index)

        ths = out.get_reader()
        self.assertTrue(np.array_equal(ths[0].plaintext, np.array([22])))

    def test_write_meta_string(self):
        out = ETSWriter(self.filename, overwrite=True)

        datas = np.random.randint(100, size=(self.nb_trace, self.nb_points), dtype=np.uint8)
        for index, data in enumerate(datas):
            out.write_points(data, index=index)
            out.write_meta(tag='plaintext', metadata='azerty', index=index)

        ths = out.get_reader()
        self.assertEqual(ths[0].plaintext, 'azerty')

    def tearDown(self):
        try:
            os.remove(self.filename)
            os.remove(self.filename_1)
            os.remove(self.filename_2)
            os.remove(self.filename_3)
        except FileNotFoundError:
            pass
