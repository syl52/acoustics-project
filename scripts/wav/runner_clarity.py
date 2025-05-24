from wav.wav import Wav
from wav.wav_graphers import calculate_and_plot_c80
from wav.wav_helpers import get_wav_files_from_folder, get_config

if __name__ == "__main__":

    folder_path = r"C:\Users\Shawn\PycharmProjects\acoustics-project\audio-files"

    # Get wav files
    ALL: list[str] = get_wav_files_from_folder(folder_path)

    # Process each file
    for fp in ALL:
        f_name = fp.split("\\")[-1].split(".")[0]
        print(f"Processing: {f_name}")
        temp_wav = Wav(fp)
        temp_wav.output_info()
        temp_wav.generate_audio_array()

        calculate_and_plot_c80(
            temp_wav.audio_array,
            temp_wav.frame_rate,
            get_config(f_name),
            f_name
        )
