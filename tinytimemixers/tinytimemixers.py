""" TinyTimeMixers main module. """


class TinyTimeMixers:
    """ TinyTimeMixers class. """

    def __init__(self):
        """ TinyTimeMixers constructor. """
        self._mixers = []

    def add(self, mixer):
        """ Add a mixer to the list. """
        self._mixers.append(mixer)

    def mix(self):
        """ Mix the mixers. """
        return sum(self._mixers)
