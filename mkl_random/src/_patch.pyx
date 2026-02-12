# Copyright (c) 2019, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# distutils: language = c
# cython: language_level=3

"""
Patch NumPy's `numpy.random` symbols to use mkl_random implementations.

This is attribute-level monkey patching. It can replace legacy APIs like
`numpy.random.RandomState` and global distribution functions, but it does not
replace NumPy's `Generator`/`default_rng()` unless mkl_random provides fully
compatible replacements.
"""

from threading import local as threading_local
from contextlib import ContextDecorator

import numpy as _np
from . import mklrand as _mr


cdef tuple _DEFAULT_NAMES = (
    # Legacy seeding / state
    "seed",
    "get_state",
    "set_state",
    "RandomState",

    # Common global sampling helpers
    "random",
    "random_sample",
    "sample",
    "rand",
    "randn",
    "bytes",

    # Integers
    "randint",

    # Common distributions (only patched if present on both sides)
    "standard_normal",
    "normal",
    "uniform",
    "exponential",
    "gamma",
    "beta",
    "chisquare",
    "f",
    "lognormal",
    "laplace",
    "logistic",
    "multivariate_normal",
    "poisson",
    "power",
    "rayleigh",
    "triangular",
    "vonmises",
    "wald",
    "weibull",
    "zipf",

    # Permutations / choices
    "choice",
    "permutation",
    "shuffle",
)


cdef class patch:
    cdef bint _is_patched
    cdef object _numpy_module
    cdef object _originals   # dict: name -> original object
    cdef object _patched     # list of names actually patched

    def __cinit__(self):
        self._is_patched = False
        self._numpy_module = None
        self._originals = {}
        self._patched = []

    def do_patch(self, numpy_module=None, names=None, bint strict=False):
        """
        Patch the given numpy module (default: imported numpy) in-place.

        Parameters
        ----------
        numpy_module : module, optional
            The numpy module to patch (e.g. `import numpy as np; use_in_numpy(np)`).
        names : iterable[str], optional
            Attributes under `numpy_module.random` to patch. Defaults to _DEFAULT_NAMES.
        strict : bool
            If True, raise if any requested symbol cannot be patched.
        """
        if numpy_module is None:
            numpy_module = _np
        if names is None:
            names = _DEFAULT_NAMES

        if not hasattr(numpy_module, "random"):
            raise TypeError("Expected a numpy-like module with a `.random` attribute.")

        # If already patched, only allow idempotent re-entry for the same numpy module.
        if self._is_patched:
            if self._numpy_module is numpy_module:
                return
            raise RuntimeError("Already patched a different numpy module; call restore() first.")

        np_random = numpy_module.random

        originals = {}
        patched = []
        missing = []

        for name in names:
            if not hasattr(np_random, name) or not hasattr(_mr, name):
                missing.append(name)
                continue
            originals[name] = getattr(np_random, name)
            setattr(np_random, name, getattr(_mr, name))
            patched.append(name)

        if strict and missing:
            # revert partial patch before raising
            for n, v in originals.items():
                setattr(np_random, n, v)
            raise AttributeError(
                "Could not patch these names (missing on numpy.random or mkl_random.mklrand): "
                + ", ".join([str(x) for x in missing])
            )

        self._numpy_module = numpy_module
        self._originals = originals
        self._patched = patched
        self._is_patched = True

    def do_unpatch(self):
        """
        Restore the previously patched numpy module.
        """
        if not self._is_patched:
            return
        numpy_module = self._numpy_module
        np_random = numpy_module.random
        for n, v in self._originals.items():
            setattr(np_random, n, v)

        self._numpy_module = None
        self._originals = {}
        self._patched = []
        self._is_patched = False

    def is_patched(self):
        return self._is_patched

    def patched_names(self):
        """
        Returns list of names that were actually patched.
        """
        return list(self._patched)


_tls = threading_local()


def _is_tls_initialized():
    return (getattr(_tls, "initialized", None) is not None) and (_tls.initialized is True)


def _initialize_tls():
    _tls.patch = patch()
    _tls.initialized = True


def monkey_patch(numpy_module=None, names=None, strict=False):
    """
    Enables using mkl_random in the given NumPy module by patching `numpy.random`.

    Examples
    --------
    >>> import numpy as np
    >>> import mkl_random
    >>> mkl_random.is_patched()
    False
    >>> mkl_random.monkey_patch(np)
    >>> mkl_random.is_patched()
    True
    >>> mkl_random.restore()
    >>> mkl_random.is_patched()
    False
    """
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_patch(numpy_module=numpy_module, names=names, strict=bool(strict))


def use_in_numpy(numpy_module=None, names=None, strict=False):
    """
    Backward-compatible alias for monkey_patch().
    """
    monkey_patch(numpy_module=numpy_module, names=names, strict=strict)


def restore():
    """
    Disables using mkl_random in NumPy by restoring the original `numpy.random` symbols.
    """
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_unpatch()


def is_patched():
    """
    Returns whether NumPy has been patched with mkl_random.
    """
    if not _is_tls_initialized():
        _initialize_tls()
    return bool(_tls.patch.is_patched())


def patched_names():
    """
    Returns the names actually patched in `numpy.random`.
    """
    if not _is_tls_initialized():
        _initialize_tls()
    return _tls.patch.patched_names()


class mkl_random(ContextDecorator):
    """
    Context manager and decorator to temporarily patch NumPy's `numpy.random`.

    Examples
    --------
    >>> import numpy as np
    >>> import mkl_random
    >>> with mkl_random.mkl_random():
    ...     x = np.random.normal(size=10)
    """
    def __init__(self, numpy_module=None, names=None, strict=False):
        self._numpy_module = numpy_module
        self._names = names
        self._strict = strict

    def __enter__(self):
        monkey_patch(numpy_module=self._numpy_module, names=self._names, strict=self._strict)
        return self

    def __exit__(self, *exc):
        restore()
        return False
