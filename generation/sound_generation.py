import time
from typing import Union

import numpy as np
import winsound
from pysinewave import SineWave


class SoundGeneration:
    """Class for sound generation object."""

    def __init__(self, freqs: list[Union[float, int]],
                 on: Union[float, int] = 5, off: Union[float, int] = 5):
        """
        :param freqs: list of frequencies to be tested
        :param on: minimum on time in seconds
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

    def _calculate_play_duration(self, frequency: float) -> int:
        """Calculate a play duration (in ms) that is a multiple of the period."""
        period = 1 / frequency  # Period in seconds
        cycles_needed = int(
            self.on / period) + 1  # Minimum integer cycles to exceed on-time
        duration = cycles_needed * period  # Total duration in seconds
        return int(duration * 1000)  # Convert to milliseconds

    def play(self) -> None:
        """Plays each frequency with a smooth zero-ending cycle using
        winsound."""

        for f in self.freqs:
            play_duration = self._calculate_play_duration(f)
            print(
                f"\nNow playing frequency {f} Hz for {play_duration / 1000:.2f} seconds:")
            winsound.Beep(int(f),
                          play_duration)  # Play the beep for calculated duration
            print(
                f"Done playing frequency {f} Hz, sleeping for {self.off} seconds.")
            time.sleep(self.off)  # Rest between notes

        print("Done playing all frequencies.")

    def play_with_decay(self, decay_rate: float = 0.1) -> None:
        for f in self.freqs:
            sinewave = SineWave(pitch_per_second=10000)
            sinewave.set_frequency(f)
            sinewave.set_volume(0)  # Start at max volume
            sinewave.play()

            play_duration = self.on
            decay_time = min(play_duration * decay_rate, play_duration)
            print("Decay time ", decay_time)
            print(f"\nPlaying {f} Hz with exponential volume decay:")
            # Let the sound play before starting decay
            time.sleep(2)

            sinewave.set_volume(-60)  # Set target volume to silence
            time.sleep(decay_time)  # Allow time for decay to complete

            sinewave.stop()
            print(f"Done playing {f} Hz, resting for {self.off} seconds.")
            time.sleep(self.off)

        print("Done playing all frequencies with decay.")

    def get_predicted_time(self) -> float:
        """Returns the predicted total play time, including pauses between frequencies."""
        total_time = sum(
            self._calculate_play_duration(f) / 1000 for f in self.freqs) + (
                             len(self.freqs) - 1) * self.off
        print(f"Predicted total time: {round(total_time)} seconds.")
        return total_time
