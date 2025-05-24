from typing import Union

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress


def plot_fft_frequencies(frequencies: np.ndarray,
                         fft_magnitudes: np.ndarray) -> None:
    """Plot the FFT frequency spectrum."""
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, fft_magnitudes)
    plt.title('FFT Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


def plot_rt60_vs_frequency(prominent_freqs: Union[np.ndarray, list[float]],
                           rt60_values: Union[
                               np.ndarray, list[float]]) -> None:
    """Plot RT60 as a function of frequency."""
    plt.figure(figsize=(10, 6))
    plt.plot(prominent_freqs, rt60_values, 'o-')
    plt.title('RT60 as a Function of Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RT60 (seconds)')
    plt.grid(True)
    plt.show()


def convert_to_db(array):
    return 20 * np.log10(np.abs(array) / np.max(np.abs(array)) + 1e-12)


def plot_time_domain_form(audio_array: np.ndarray, sample_rate: int) -> None:
    """Plot the time-domain representation of the audio signal in dB scale."""
    # Check if audio is stereo or mono
    if len(audio_array.shape) == 2:  # Stereo or multi-channel
        num_channels = audio_array.shape[1]
    else:  # Mono
        num_channels = 1

    time_axis = np.linspace(0, len(audio_array) / sample_rate,
                            len(audio_array))
    # Plot each channel on a separate subplot
    plt.figure(figsize=(12, 4 * num_channels))
    for i in range(num_channels):
        audio_db = convert_to_db(audio_array[:, i])
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(time_axis, audio_db)
        plt.title(f"Channel {i + 1} (dB Scale)")
        plt.xlabel("Time (seconds)")
        plt.ylim(-60, 0)  # only interested in -60dB upwards
        plt.ylabel("Amplitude (dB)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def regress_and_plot_time_domain(
        audio_array: np.ndarray,
        sample_rate: int,
        regions: list[list[tuple[float, float]]],
        f_name: str
) -> None:
    """
    Analyse and plot the time-domain representation of the audio signal in
    dB scale, with regression lines for specified regions.

    :param f_name: File name, used in chart title
    :param audio_array: The audio data as a numpy array.
    :param sample_rate: Sampling rate of the audio file.
    :param regions: List of lists where each sublist corresponds to one channel,
                    and contains tuples (t_start, t_end) for analysis regions.
    """
    # Check if audio is stereo or mono
    if len(audio_array.shape) == 2:  # Stereo or multi-channel
        num_channels = audio_array.shape[1]
    else:  # Mono
        num_channels = 1
        audio_array = audio_array[:,
                      np.newaxis]  # Reshape to 2D for uniform processing

    time_axis = np.linspace(0, len(audio_array) / sample_rate,
                            len(audio_array))

    # Plot each channel
    plt.figure(
        figsize=(12, 4 * num_channels))  # Adjust height for multiple channels
    # header for copying tabulated data to excel as csv
    print("Channel, Start(s), End(s), RT60(s)    , R²     , Decay_Rate(dB/s)")

    for i in range(num_channels):
        audio_db = convert_to_db(audio_array[:, i])
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(time_axis, audio_db, label=f"Channel {i + 1}", alpha=0.5)
        plt.ylim(-60, 0)
        plt.title(f"{f_name}/ Channel {i + 1} (dB Scale)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (dB)")
        plt.grid(True)

        # Process each region for this channel
        for t_start, t_end in regions[i]:
            # Convert timestamps to sample indices
            start_idx = int(t_start * sample_rate)
            end_idx = int(t_end * sample_rate)

            # Extract the region of interest
            region_time = time_axis[start_idx:end_idx]
            region_db = audio_db[start_idx:end_idx]

            # Find peaks in the region, prominence 5dB Change this!
            peaks, _ = find_peaks(region_db,
                                  prominence=1.5,
                                  distance=int(0.005 * sample_rate),
                                  )
            # min distance of peaks 0.02 seconds converted to samples
            # this greatly improved R^2 values
            peak_times = region_time[peaks]
            peak_values = region_db[peaks]

            # keep only peaks above -40dB (change this)
            threshold_db = -50
            valid_indices = peak_values > threshold_db
            peak_times = peak_times[valid_indices]
            peak_values = peak_values[valid_indices]

            if len(peak_values) < 2:
                print(
                    f"Channel {i + 1}, Region {t_start}-{t_end}s: Not enough peaks detected.")
                continue

            # Perform linear regression on the peaks
            slope, intercept, r_value, _, _ = linregress(peak_times,
                                                         peak_values)
            best_fit_line = slope * region_time + intercept

            # include this for some reason to prevent errors

            # Plot the peaks and the best-fit line
            plt.plot(peak_times, peak_values, 'r^', label="Peaks", alpha=0.3)
            plt.plot(region_time, best_fit_line,
                     label=f"Fit {t_start}-{t_end}s (R²={r_value ** 2:.2f})")

            # Print details to stdout
            decay_rate = slope  # dB/s
            RT60 = 60 / decay_rate
            """
            print(f"Channel {i + 1}, Region {t_start}-{t_end}s:")
            print(
                f"  R² = {r_value ** 2:.4f}, Decay Rate = {decay_rate:.4f} dB/s")
            """
            # can now copy to excel

            # Print the data rows, aligned with headers
            print(
                f"{i + 1},       {t_start:.4f},  {t_end:.4f}   ,"
                f"{-RT60:.4f},   "
                f"{r_value ** 2:.4f},   "
                f"{-decay_rate:.4f}")

        # plt.legend()

    plt.tight_layout()
    # Showing plot makes computer very laggy
    # Comment this line out if too slow
    plt.show()


def calculate_and_plot_c80(
        audio_array: np.ndarray,
        sample_rate: int,
        regions: list[list[tuple[float, float]]],
        f_name: str
) -> None:
    """
    Calculate the Clarity Index (C80) and plot the waveform with markers.
    Code partly shared from `regress_and_plot_time_domain`.

    :param audio_array: The audio data as a numpy array.
    :param sample_rate: Sampling rate of the audio file.
    :param regions: List of lists of (t_start, t_end) tuples per channel.
    :param f_name: File name for title purposes.
    """
    if len(audio_array.shape) == 2:
        num_channels = audio_array.shape[1]
    else:
        num_channels = 1
        audio_array = audio_array[:, np.newaxis]

    time_axis = np.linspace(0, len(audio_array) / sample_rate,
                            len(audio_array))
    plt.figure(figsize=(12, 4 * num_channels))

    print("Channel, Start(s), End(s), C80 (dB)")

    for i in range(num_channels):
        signal = audio_array[:, i]
        signal_abs = np.abs(signal)
        signal_db = convert_to_db(signal_abs)
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(time_axis, signal_db, label=f"Channel {i + 1}", alpha=0.6)
        plt.ylim(-60, 0)
        plt.title(f"{f_name}/ Channel {i + 1} - Time Domain")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (dB)")
        plt.grid(True)

        for t_start, t_end in regions[i]:
            start_idx = int(t_start * sample_rate)
            end_idx = int(t_end * sample_rate)
            split_idx = start_idx + int(0.08 * sample_rate)  # 80 ms later

            if split_idx >= end_idx:
                print(
                    f"Channel {i + 1}, Region {t_start}-{t_end}s: too short for 80 ms window.")
                continue

            signal_region = signal[start_idx:end_idx]
            signal_squared = signal_region ** 2

            early_energy = np.sum(signal_squared[:split_idx - start_idx])
            late_energy = np.sum(signal_squared[split_idx - start_idx:])

            if late_energy == 0:
                print(
                    f"Channel {i + 1}, Region {t_start}-{t_end}s: late energy is zero.")
                continue

            C80 = 10 * np.log10(early_energy / late_energy)

            print(f"{i + 1},       {t_start:.4f},  {t_end:.4f},  {C80:.2f} dB")

            # Flags to add legend labels only once per type
            label_flags = {'start': True, '80ms': True, 'end': True}

            for t_start, t_end in regions[i]:
                ...
                # Plot vertical lines with labels only once
                plt.axvline(x=t_start, color='g', linestyle='--',
                            label='Start of Decay' if label_flags[
                                'start'] else None)
                label_flags['start'] = False

                plt.axvline(x=t_start + 0.08, color='r', linestyle='--',
                            label='End of 80 ms' if label_flags['80ms'] else
                            None)
                label_flags['80ms'] = False

                plt.axvline(x=t_end, color='k', linestyle='--',
                            label='End of Decay' if label_flags[
                                'end'] else None)
                label_flags['end'] = False

        # Manually filter duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict()
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        plt.legend(unique.values(), unique.keys(), loc='lower right')

    plt.tight_layout()
    plt.show()
