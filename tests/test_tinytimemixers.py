""" Tests for the main module. """

from tinttimemixers import TinyTimeMixers

def test_tiny_time_mixers_initialization():
    ttm = TinyTimeMixers()
    assert isinstance(ttm, TinyTimeMixers)
    assert ttm._mixers == []

def test_add_mixer():
    ttm = TinyTimeMixers()
    ttm.add(5)
    assert ttm._mixers == [5]

def test_mix_mixers():
    ttm = TinyTimeMixers()
    ttm.add(5)
    ttm.add(10)
    assert ttm.mix() == 15

def test_mix_no_mixers():
    ttm = TinyTimeMixers()
    with pytest.raises(TypeError):
        ttm.mix()
