import numpy as np
import mkl_random
import pytest

def test_is_patched():
    """
    Test that is_patched() returns correct status.
    """
    assert not mkl_random.is_patched()
    mkl_random.monkey_patch(np)
    assert mkl_random.is_patched()
    mkl_random.restore()
    assert not mkl_random.is_patched()

def test_monkey_patch_and_restore():
    """
    Test that monkey_patch replaces and restore brings back original functions.
    """
    # Store original functions
    orig_normal = np.random.normal
    orig_randint = np.random.randint
    orig_RandomState = np.random.RandomState

    try:
        mkl_random.monkey_patch(np)

        # Check that functions are now different objects
        assert np.random.normal is not orig_normal
        assert np.random.randint is not orig_randint
        assert np.random.RandomState is not orig_RandomState

        # Check that they are from mkl_random
        assert np.random.normal is mkl_random.mklrand.normal
        assert np.random.RandomState is mkl_random.mklrand.RandomState

    finally:
        mkl_random.restore()

    # Check that original functions are restored
    assert mkl_random.is_patched() is False
    assert np.random.normal is orig_normal
    assert np.random.randint is orig_randint
    assert np.random.RandomState is orig_RandomState

def test_context_manager():
    """
    Test that the context manager patches and automatically restores.
    """
    orig_uniform = np.random.uniform
    assert not mkl_random.is_patched()

    with mkl_random.mkl_random(np):
        assert mkl_random.is_patched() is True
        assert np.random.uniform is not orig_uniform
        # Smoke test inside context
        arr = np.random.uniform(size=10)
        assert arr.shape == (10,)

    assert not mkl_random.is_patched()
    assert np.random.uniform is orig_uniform

def test_patched_functions_callable():
    """
    Smoke test to ensure some patched functions can be called without error.
    """
    mkl_random.monkey_patch(np)
    try:
        # These calls should now be routed to mkl_random's implementations
        x = np.random.standard_normal(size=100)
        assert x.shape == (100,)

        y = np.random.randint(0, 100, size=50)
        assert y.shape == (50,)
        assert np.all(y >= 0) and np.all(y < 100)

        st = np.random.RandomState(12345)
        z = st.rand(10)
        assert z.shape == (10,)

    finally:
        mkl_random.restore()

def test_patched_names():
    """
    Test that patched_names() returns a list of patched symbols.
    """
    try:
        mkl_random.monkey_patch(np)
        names = mkl_random.patched_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "normal" in names
        assert "RandomState" in names
    finally:
        mkl_random.restore()
