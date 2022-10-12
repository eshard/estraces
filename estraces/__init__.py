# -*- coding: utf-8 -*-
"""eshard's Trace set Python library.

esTraces provides the basic tools to handle side-channel traces in a unified way, through:
   * classes dedicated to trace and trace set manipulation: :class:`estraces.TraceHeaderSet`, :class:`estraces.Trace`
   * specific file formats reader: binaries formats and eShard's format ETS
   * abstract base class to easily implement a reader for any specific file format

Trace and TraceHeaderSet
========================

:class:`estraces.Trace` and :class:`estraces.TraceHeaderSet` are the classes which abstract your samples and the corresponding metadatas.
Through these objects, you can:
* manipulate your trace set with TraceHeaderSet - filter it, slice it, ...
* manipulate your samples data through the `samples` attribute of both TraceHeaderSet and Trace, which provides a Numpy array-like API.
* manipulate your metadata through the `metadatas` attribute of both TraceHeaderSet and Trace, which provides a dict-like interface to your metadatas.

All the underlying mechanics to read datas on your trace set files are managed by a format reader.

`Trace` class
-------------
.. autoclass:: estraces.Trace

`TraceHeaderSet` class
----------------------
.. autoclass:: estraces.TraceHeaderSet
    :members:
    :member-order: bysource

`Samples` class
---------------
.. autoclass:: estraces.Samples
    :members:
    :member-order: bysource

`Metadatas` class
-----------------
.. autoclass:: estraces.Metadatas
    :members:
    :member-order: bysource


Get a trace header set from files - Format readers
==================================================

Basically, formats reader are classes which implements methods to read samples and metadatas from a trace set files. For the sake of simplicity,
a format should always comes with a simple factory function to get a ths from basic parameters. esTraces provides:

   * a bin format reader, which comes with two factory functions
   * the eshard ETS format reader which comes with its factory functions :func:`estraces.read_ths_from_ets_file`

Bin reader format: `read_ths_from_bin_...` functions
----------------------------------------------------

.. autofunction:: estraces.read_ths_from_bin_filenames_pattern


.. autofunction:: estraces.read_ths_from_bin_filenames_list


.. automodule:: estraces.formats.bin_extractor
    :members:


ETS reader format:  `read_ths_from_ets_file`
--------------------------------------------

.. autofunction:: estraces.read_ths_from_ets_file

TRS reader format:  `read_ths_from_trs_file`
--------------------------------------------

.. autofunction:: estraces.read_ths_from_trs_file

Get a trace header set from numpy arrays
========================================

The `read_ths_from_ram` function allows to build a `TraceHeaderSet` from any arrays, in memory.

RAM reader format: `read_ths_from_ram` function
-----------------------------------------------

.. autofunction:: estraces.read_ths_from_ram

Get a trace header set concatenated from multiple trace header set instances
============================================================================

The `read_ths_from_multiple_ths` function allows to build a `TraceHeaderSet` from several others `TraceHeaderSet` instance,
as a concatenation.

.. autofunction:: estraces.read_ths_from_multiple_ths

Writing an ETS file
===================

The class `ETSWriter` provides API to create Eshard Trace Set file.

.. autoclass:: estraces.ETSWriter

Implementing a new format reader
================================

To handle your own trace set format, you must create a format reader class inheriting from :class:`estraces.AbstractReader`.
A reader implementation must basically implement two methods:
* a `fetch_samples` method which takes a traces id list and samples frame as inputs, and returns a 2 dimension numpy array
* a `fetch_metadatas` method which takes a trace id and a metadata name, and returns the metadata values

When developing your own format reader, you should also provide a factory function to get a TraceHeaderSet instance from the files.

The base format reader `AbstractReader`
---------------------------------------

.. autoclass:: estraces.AbstractReader
    :members:
    :member-order: bysource

How to create a `TraceHeaderSet` instance
-----------------------------------------

.. autofunction:: estraces.build_trace_header_set


:copyright: (c) 2022 ESHARD

=========

"""
from .traces import Trace, TraceHeaderSet, Samples, Metadatas, build_trace_header_set, AbstractReader
from .formats import (
    read_ths_from_bin_filenames_list,
    read_ths_from_bin_filenames_pattern,
    read_ths_from_ets_file,
    read_ths_from_trs_file,
    read_ths_from_ram,
    read_ths_from_multiple_ths,
    read_ths_from_sqlite,
    bin_extractor
)
from .formats.ets_writer import ETSWriter, ETSWriterError, compress_ets
from .formats.bin_format import PaddingMode
from . import _version

__all__ = [
    "Trace",
    "TraceHeaderSet",
    "Samples",
    "Metadatas",
    "read_ths_from_bin_filenames_list",
    "read_ths_from_bin_filenames_pattern",
    "read_ths_from_trs_file",
    "bin_extractor",
    "read_ths_from_ets_file",
    "read_ths_from_ram",
    "ETSWriter",
    "ETSWriterError",
    "AbstractReader",
    "build_trace_header_set",
    "PaddingMode",
    "compress_ets",
    "read_ths_from_multiple_ths",
    "read_ths_from_sqlite"
]

# Set default logging handler to avoid "No handler found" warnings.
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = _version.get_versions()['version']
VERSION = _version.get_versions()['version']
