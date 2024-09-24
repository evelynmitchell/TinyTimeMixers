""" Tests for the main module. """

import tinytimemixers
import pytest


def test_tiny_time_mixers_initialization():
    """ Test the TinyTimeMixers class initialization. """
    ttm = TinyTimeMixers()
    assert isinstance(ttm, TinyTimeMixers)
    assert ttm._mixers == []     # noqa W0212


def test_add_mixer():
    """ Test the add method. """
    ttm = TinyTimeMixers()
    ttm.add(5)
    assert ttm._mixers == [5]   # noqa W0212


def test_mix_mixers():
    """ Test the mix method. """
    ttm = TinyTimeMixers()
    ttm.add(5)
    ttm.add(10)
    assert ttm.mix() == 15


def test_mix_no_mixers():
    """ Test the mix method with no mixers. """
    ttm = TinyTimeMixers()
    with pytest.raises(TypeError):
        ttm.mix()
