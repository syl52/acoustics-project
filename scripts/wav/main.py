from wav.wav import Wav
from wav.wav_helpers import get_wav_files_from_folder

if __name__ == "__main__":

    folder_path = r"C:\Users\Shawn\PycharmProjects\acoustics-project\test-audio-files"

    # Get all wav files from the folder
    ALL: list[str] = get_wav_files_from_folder(folder_path)

    # Process each file
    for fp in ALL:
        print(f"Processing: {fp}")
        temp_wav = Wav(fp)
        temp_wav.output_info()

        ambient_duration = 1.0  # duration of ambient noise beginning of file
        temp_wav.plot_rt60_vs_frequency(ambient_duration)
