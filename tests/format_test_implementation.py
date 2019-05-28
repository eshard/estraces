from .context import estraces  # noqa
import numpy as np
from estraces import AbstractReader


class Format(AbstractReader):

    def __init__(self, datas=None, metadatas=None):
        self._datas = (
            np.vstack(
                [np.random.randint(0, 256, 10000, dtype="uint8") for i in range(10)]
            )
            if datas is None else datas
        )
        self._size = self._datas.shape[0]

        self._raw_metadatas = {
            "key1": np.array(range(self._size)),
            3: np.array(range(self._size))
        } if metadatas is None else metadatas

    def fetch_samples(self, traces, frame=None):
        if len(traces) == self._size:
            traces = slice(None, None)
        res = self._datas[traces, :][:, frame]
        try:
            if res.ndim == 1:
                return np.array([res], dtype=self._datas.dtype)
            elif isinstance(res, int):
                return np.array([[res]], dtype=self._datas.dtype)
            return res
        except Exception:
            return res

    def fetch_metadatas(self, key, trace_id=None):
        if trace_id is None:
            return self._raw_metadatas[key]
        else:
            return self._raw_metadatas[key][trace_id]

    @property
    def metadatas_keys(self):
        return self._raw_metadatas.keys()

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            key = [len(self) + k if k < 0 else k for k in key]
            key = np.array(key, dtype='uint8')
        elif isinstance(key, int):
            key = np.array([key], dtype='uint8')
        return Format(datas=self._datas[key, :], metadatas={k: v[key] for k, v in self._raw_metadatas.items()})

    def get_trace_size(self, trace_id):
        return self._datas[trace_id].shape[0]
