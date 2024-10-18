import wave

import numpy as np

from wav.wav_graphers import plot_rt60_vs_frequency
from wav.wav_helpers import remove_ambient_noise, calculate_rt60, bandpass_filter, get_prominent_frequencies


class Wav:
    """Class representation for Wav files."""

    def __init__(self, filepath: str):
        self.filepath = filepath

        self.audio_array = None

        self.__on_startup()

    def __on_startup(self) -> None:
        with wave.open(self.filepath, 'rb') as wav_file:
            # Extract Raw Audio from Wav File
            self.n_channels = wav_file.getnchannels()
            self.sample_width = wav_file.getsampwidth()
            self.frame_rate = wav_file.getframerate()
            self.n_frames = wav_file.getnframes()
            self.audio_data = wav_file.readframes(self.n_frames)

    def output_info(self) -> None:
        """Output basic information about the wav file, should the user wish."""

        # TODO possibly change from print to logger?

        print(f"Number of channels: {self.n_channels}")
        print(f"Sample width (bytes): {self.sample_width}")
        print(f"Frame rate (samples per second): {self.frame_rate}")
        print(f"Total number of frames: {self.n_frames}")
        print(f"Duration (seconds): {self.n_frames / self.frame_rate:.2f}")

    def generate_audio_array(self) -> np.ndarray:
        """Read audio data as stream, convert to usable numpy array."""

        audio_array = np.frombuffer(self.audio_data, dtype=np.int16)

        # If stereo, reshape the array to split channels
        if self.n_channels == 2:
            self.audio_array = audio_array.reshape(-1, 2)
        else:
            self.audio_array = audio_array

        return self.audio_array

    def plot_rt60_vs_frequency(self, ambient_duration: float) -> None:
        """Analyse and plot RT60 as a function of frequency."""
        # Step 1: Generate the audio array
        self.generate_audio_array()

        # Step 2: Remove ambient noise
        processed_audio = remove_ambient_noise(self.audio_array, self.frame_rate, ambient_duration)

        # Step 3: Identify prominent frequencies
        prominent_freqs = get_prominent_frequencies(processed_audio, self.frame_rate)

        # Step 4: Calculate RT60 for each prominent frequency
        rt60_values = []
        for freq in prominent_freqs:
            filtered_audio = bandpass_filter(processed_audio, freq, self.frame_rate)
            rt60 = calculate_rt60(filtered_audio, self.frame_rate)
            rt60_values.append(rt60)

        # Step 5: Plot RT60 as a function of frequency
        plot_rt60_vs_frequency(prominent_freqs, rt60_values)
