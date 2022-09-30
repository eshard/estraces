import numpy as _np


def is_array_equivalent_to_a_slice(array):
    return array is not None and _array_has_enough_items(array) and _items_are_in_ascending_order(array) and _items_are_equally_spaced(array)


def _items_are_in_ascending_order(array):
    return _np.any(_np.diff(array) > 0)


def _array_has_enough_items(array):
    return len(array) >= 3


def _items_are_equally_spaced(array):
    return not _np.any(_np.diff(array, 2))


def build_equivalent_slice(array):
    if len(array) == 0:
        return slice(0, 0, 1)
    start = _np.min(array)
    stop = _np.max(array) + 1
    step = array[1] - array[0] if len(array) > 1 else 1
    return slice(start, stop, step)
