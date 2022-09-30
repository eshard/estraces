# -*- coding: utf-8 -*-
"""Provides metaclass for Trace and TraceHeaderSet classes.

:copyright: (c) 2015-2022 ESHARD
"""


class ClassWithMetadatas(type):
    """Metaclass used to create classes providing attribute-like access to metadata associated to a primary data sourced wrapped by the class.

    It is used to create specific classes inheriting from :class:`Trace` class or from :class:`TraceHeaderSet` class, with attribute accessor to metadata
    suited to the specific Format implementation provided.
    """

    def __new__(cls, name, bases, namespace, metadatas_keys):
        res = type.__new__(cls, name, bases, namespace)
        res._metadatas_keys = metadatas_keys
        res._set_metadatas_properties()
        res._is_valid_trace_class = True
        res.__doc__ = bases[0].__doc__
        return res

    def _set_metadatas_properties(cls):
        for k in cls._metadatas_keys:
            setattr(cls, f"{k}", cls._create_metadata_property(metadata_key=k))

    def _create_metadata_property(cls, metadata_key):

        def _(self):
            return self.metadatas[metadata_key]

        return property(fget=_, doc="""Return f'{metadata_key}' array of values.""")
