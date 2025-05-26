import importlib
import os
from typing import Optional

import numpy as np
from scipy.fftpack import fft
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, butter, filtfilt


def convert_to_db(array):
    return 20 * np.log10(np.abs(array) / np.max(np.abs(array)) + 1e-12)


def remove_ambient_noise(audio_data: np.ndarray, frame_rate: int, ambient_duration: float) -> np.ndarray:
    """Remove the ambient noise from the beginning of the audio."""
    ambient_samples = int(frame_rate * ambient_duration)
    ambient_noise = np.mean(audio_data[:ambient_samples])  # Estimate average noise
    processed_audio = audio_data[ambient_samples:] - ambient_noise
    return processed_audio


def calculate_rt60(audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
    """Calculate RT60 (reverberation time)."""
    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize between -1 and 1

    # Energy calculation
    energy = audio_data ** 2
    cumulative_energy = np.cumsum(energy[::-1])[::-1]  # Reverse cumulative sum for energy decay

    # Prevent issues with log zero
    cumulative_energy = np.clip(cumulative_energy, a_min=1e-10, a_max=None)

    # Convert to dB
    cumulative_energy_db = 10 * np.log10(cumulative_energy / np.max(cumulative_energy))

    # Smoothing the curve (reduce smoothing window size)
    smoothed_energy = uniform_filter1d(cumulative_energy_db, size=200)

    # Find where the energy drops by 60 dB
    rt60_index = np.where(smoothed_energy <= -60)[0]
    if len(rt60_index) == 0:
        return None  # No RT60 found

    rt60_time = rt60_index[0] / sample_rate  # Convert index to time in seconds
    return rt60_time


def bandpass_filter(audio_data: np.ndarray, target_freq: float, sample_rate: int, bandwidth: float = 50) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter using filtfilt for zero-phase distortion.

    :param audio_data: Time-domain signal.
    :param target_freq: Centre frequency in Hz.
    :param sample_rate: Sampling rate of signal.
    :param bandwidth: Bandwidth around the centre frequency in Hz.
    :return: Bandpass filtered signal.
    """
    nyquist = 0.5 * sample_rate
    low = (target_freq - bandwidth / 2) / nyquist
    high = (target_freq + bandwidth / 2) / nyquist

    if low <= 0:
        low = 1e-5  # Prevent invalid filter

    b, a = butter(N=4, Wn=[low, high], btype='band')
    filtered_audio = filtfilt(b, a, audio_data)
    return filtered_audio


def get_prominent_frequencies(audio_data: np.ndarray, sample_rate: int, min_freq_distance: float = 4.0, max_freq: float = 4000) -> np.ndarray:
    """Identify the prominent frequencies in the audio data, filtering out those that are too close and out of range, and round to nearest integer."""
    n = len(audio_data)
    audio_fft = fft(audio_data)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)

    # Only positive frequencies
    pos_freqs = freqs[:n // 2]
    pos_fft = np.abs(audio_fft[:n // 2])

    # Limit to frequencies below max_freq
    valid_indices = np.where(pos_freqs <= max_freq)[0]
    pos_freqs = pos_freqs[valid_indices]
    pos_fft = pos_fft[valid_indices]

    # Find prominent frequencies
    peaks, _ = find_peaks(pos_fft, height=np.max(pos_fft) * 0.1)
    prominent_freqs = pos_freqs[peaks]

    # Filter out frequencies that are too close
    filtered_freqs = [prominent_freqs[0]]  # Start with the first prominent frequency
    for freq in prominent_freqs[1:]:
        if freq - filtered_freqs[-1] > min_freq_distance:
            filtered_freqs.append(freq)

    # Round the prominent frequencies to the nearest integer
    rounded_freqs = np.round(filtered_freqs).astype(int)

    return rounded_freqs


def calculate_bandwise_rt60s(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Calculate RT60 for octave-spaced frequency bands from 64 Hz to 2048 Hz.

    Parameters:
        :param audio_data: (np.ndarray): Input audio signal (1D).
        :param sample_rate: (int): Sampling rate in Hz.

    Returns:
        dict: Mapping of center frequency to RT60 value (in seconds), or None if not measurable.
    """
    target_freqs = [64, 128, 256, 512, 1024, 2048]
    # same freqs as in single-freq tests
    rt60_results = {}

    for freq in target_freqs:
        filtered = bandpass_filter(audio_data, freq, sample_rate,
                                   bandwidth=freq * 0.25)
        # 32 Hz lower limit BW for stability - this is fine since we only go
        # down to 64 Hz
        rt60 = calculate_rt60(filtered, sample_rate)
        rt60_results[freq] = rt60

    return rt60_results


def get_wav_files_from_folder(folder_path: str) -> list[str]:
    """Retrieve all WAV file paths from a given folder."""
    wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


def get_config(file_name: str, config_module: str =
"regression_start_end_times") -> list:
    """
    Fetch the corresponding list from the configuration module
    based on the file name.

    :param file_name: Name of the file to match.
    :param config_module: Module where lists are stored.
    :return: Matching list if found, else None.
    """
    # Dynamically import the config module
    config = importlib.import_module(config_module)

    # Attempt to fetch the corresponding list
    return getattr(config, file_name, None)
