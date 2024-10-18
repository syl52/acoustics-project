import os
from typing import Optional

import numpy as np
from scipy.fftpack import fft
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


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
    """Apply a simple bandpass filter to isolate the target frequency."""
    n = len(audio_data)
    fft_data = fft(audio_data)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)

    # Bandpass filter mask
    filter_mask = (np.abs(freqs - target_freq) <= bandwidth)
    filtered_fft = fft_data * filter_mask

    # Inverse FFT to get filtered audio
    filtered_audio = np.fft.ifft(filtered_fft).real
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


def get_wav_files_from_folder(folder_path: str) -> list[str]:
    """Retrieve all WAV file paths from a given folder."""
    wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files
