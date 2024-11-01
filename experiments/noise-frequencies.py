import numpy as np

from wav.wav import Wav
from wav.wav_graphers import plot_fft_frequencies
from wav.wav_helpers import get_wav_files_from_folder

if __name__ == "__main__":

    folder_path = r"C:\Users\Shawn\PycharmProjects\acoustics-project\test-audio-files"

    # Get all wav files from the folder
    ALL: list[str] = get_wav_files_from_folder(folder_path)
    # prune to only noise files
    ALL = [file for file in ALL if 'noise' in file]
    # plot fourier transforms of noise1 and noise2 files

    for file_path in ALL:
        wav_file = Wav(file_path)
        audio_array = wav_file.generate_audio_array()
        sample_rate = wav_file.frame_rate

        # Calculate FFT and plot
        audio_fft = np.fft.fft(audio_array)
        frequencies = np.fft.fftfreq(len(audio_fft), 1 / sample_rate)
        fft_magnitudes = np.abs(audio_fft[:len(
            audio_fft) // 2])  # Take only positive frequencies
        plot_fft_frequencies(frequencies[:len(frequencies) // 2],
                             fft_magnitudes)
