import time
from typing import Union

import winsound


class SoundGeneration:
    """Class for sound generation object."""

    def __init__(self, freqs: list[Union[float, int]],
                 on: Union[float, int] = 5, off: Union[float, int] = 5):
        """
        :param freqs: list of frequencies to be tested
        :param on: on time in seconds
        :param off: off time in seconds
        """
        self.freqs = freqs
        self.on = on
        self.off = off
        self._validate()

    def _validate(self) -> None:
        assert len(self.freqs) > 0, "Must provide at least one frequency."
        for i in self.freqs:
            assert isinstance(i, float) or isinstance(i, int), \
                "Frequency must be float or int"
        assert self.on >= 3, "On must be greater than 3 s"
        assert self.off >= 3, "Off must be greater than 5 s"

    def play(self) -> None:
        """Plays the sound."""

        for f in self.freqs:
            print(f"\nNow playing frequency {f} Hz:")
            winsound.Beep(int(f), int(self.on * 1000))  # NB this time in ms
            print(f"Done playing frequency {f} Hz, sleeping {self.off} "
                  f"seconds.")
            time.sleep(self.off)  # NB this time in s

        print("Done playing all.")

    def get_predicted_time(self) -> float:
        """Returns the predicted time for a single complete play."""
        counter = 0
        counter += (len(self.freqs) * self.on)
        counter += ((len(self.freqs) - 1) * self.off)
        print(f"Predicted time is {round(counter)} seconds.")
        return counter
