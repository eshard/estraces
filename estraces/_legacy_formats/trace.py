import warnings


class TraceDeprecated:

    @property
    def nb_point(self):
        warnings.warn(
            "Trace.nb_point method is deprecated. Use len(trace) instead.",
            DeprecationWarning
        )
        return len(self)

    def points(self, frame=None):
        warnings.warn(
            "Trace.points method is deprecated. Use .samples attribute instead.",
            DeprecationWarning
        )
        if frame is None:
            return self.samples[:]
        if isinstance(frame, range):
            frame = slice(frame.start, frame.stop, frame.step)
        if isinstance(frame, tuple):
            frame = list(frame)
        return self.samples[frame]

    @property
    def filename(self):
        try:
            return self._reader._filenames[self._id]
        except Exception:
            return AttributeError('filename attribute does not exist.')

    def __get_writable_attributes__(self):
        res = list(self.metadatas.keys())
        for k, v in self.__dict__.items():
            if k not in ('points', 'filename', 'nb_point', 'samples', 'metadatas', 'id', 'name') and k[0] != '_' and k[0:1] != '__':
                res.append(k)
        return res
