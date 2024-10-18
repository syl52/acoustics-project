from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def plot_fft_frequencies(frequencies: np.ndarray, fft_magnitudes: np.ndarray) -> None:
    """Plot the FFT frequency spectrum."""
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, fft_magnitudes)
    plt.title('FFT Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


def plot_rt60_vs_frequency(prominent_freqs: Union[np.ndarray, list[float]], rt60_values: Union[np.ndarray, list[float]]) -> None:
    """Plot RT60 as a function of frequency."""
    plt.figure(figsize=(10, 6))
    plt.plot(prominent_freqs, rt60_values, 'o-')
    plt.title('RT60 as a Function of Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RT60 (seconds)')
    plt.grid(True)
    plt.show()
