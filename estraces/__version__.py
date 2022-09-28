def _get_version():
    with open("./version.txt") as f:
        v = f.readline()
        return tuple([int(x) for x in v.split(".")])


VERSION = _get_version()

__version__ = ".".join(map(str, VERSION))
__copyright__ = "(c) 2015-2022 ESHARD"
