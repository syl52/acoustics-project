from typing import Optional

import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def remove_ambient_noise(audio_data: np.ndarray, frame_rate: int, ambient_duration: float) -> np.ndarray:
    """Remove the ambient noise from the beginning of the audio."""
    ambient_samples = int(frame_rate * ambient_duration)
    ambient_noise = np.mean(audio_data[:ambient_samples])  # Estimate average noise
    processed_audio = audio_data[ambient_samples:] - ambient_noise
    return processed_audio

def calculate_rt60(audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
    """Calculate RT60 (reverberation time)."""
    energy = audio_data ** 2
    cumulative_energy = np.cumsum(energy[::-1])[::-1]  # Reverse cumulative sum for energy decay
    cumulative_energy_db = 10 * np.log10(cumulative_energy / np.max(cumulative_energy))

    # Smooth energy curve
    smoothed_energy = uniform_filter1d(cumulative_energy_db, size=1000)

    # Find RT60 (time to drop 60 dB)
    rt60_index = np.where(smoothed_energy <= -60)[0]
    if len(rt60_index) == 0:
        return None  # No RT60 found
    rt60_time = rt60_index[0] / sample_rate
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

def get_prominent_frequencies(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Identify the prominent frequencies in the audio data."""
    n = len(audio_data)
    audio_fft = fft(audio_data)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)

    # Only positive frequencies
    pos_freqs = freqs[:n//2]
    pos_fft = np.abs(audio_fft[:n//2])

    # Find prominent frequencies
    peaks, _ = find_peaks(pos_fft, height=np.max(pos_fft) * 0.1)
    return pos_freqs[peaks]
