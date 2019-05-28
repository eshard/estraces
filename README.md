# estraces - Traces and trace sets Python library for side-channel attacks

[![pipeline status](https://gitlab.com/eshard/estraces/badges/master/pipeline.svg)](https://gitlab.com/eshard/estraces/commits/master)
[![PyPI version](https://badge.fury.io/py/estraces.svg)](https://badge.fury.io/py/estraces)
[![Conda installer](https://anaconda.org/eshard/estraces/badges/installer/conda.svg)](https://anaconda.org/eshard/estraces)
[![Latest Conda release](https://anaconda.org/eshard/estraces/badges/latest_release_date.svg)](https://anaconda.org/eshard/estraces)

estraces is a Python library to manipulate side-channel trace sets. It aims at giving a clear and uniform API to handle
traces samples and metadata for various persistency and file formats.
It uses [Numpy](https://www.numpy.org) to handle data.

estraces was originally developped and maintain by [eshard](https://www.eshard.com), and is heavily used in the open-source
side-channel analysis framework.

## Getting started

### Prerequisites

You will need **Python 3.6+** to use and install estraces. You can use pip (or any pip based tool like pipenv) or conda to install it.

### Installation

To install estraces, you can use pip (or pipenv, or any other pip based-tool) or conda:

```bash
$ pip install estraces
# or with Conda
$ conda install -c eshard estraces
```

### Opens a trace set

If you have a trace set as binary files, you can get a trace header set by using the binary reader:

```python
# First import the lib
import estraces

# We suppose the binary files are under traces/ and are named something.bin
my_traces = estraces.read_ths_from_bin_filenames_pattern(
    'traces/*.bin', # First indicate the filename pattern for the bin file
    dtype='uint8', # Indicate the numpy dtype of the data
    metadatas_parsers={} # This dict allows to associate metadata
)
```

You can then read your samples:

```python
# This will return the data for the first 100 traces
my_traces.samples[:100]

# This will return the frame 0 - 1000 of all the traces as a numpy array
my_traces.samples[:, :1000]

# You can iterate on traces
for trace in my_traces:
    # do something
```

## Documentation

To go further and learn all about estraces, please go to [the full documentation](https://eshard.gitlab.io/estraces).

## Contributing

All contributions, starting with feedbacks, are welcomed.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) if you wish to contribute to the project.

## License

This library is licensed under LGPL V3 license. See the [LICENSE](LICENSE) file for details.

It is mainly intended for non-commercial use, by academics, students or professional willing to learn the basics of side-channel analysis.

If you wish to use this library in a commercial or industrial context, eshard provides commercial licenses under fees. Contact us!

## Authors

See [AUTHORS](AUTHORS.md) for the list of contributors to the project.
