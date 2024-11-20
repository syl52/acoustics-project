from regression_start_end_times import A1
from wav.wav import Wav
from wav.wav_graphers import regress_and_plot_time_domain
from wav.wav_helpers import get_wav_files_from_folder

if __name__ == "__main__":

    folder_path = r"C:\Users\Shawn\PycharmProjects\acoustics-project\test-audio-files"

    # Get all wav files from the folder
    ALL: list[str] = get_wav_files_from_folder(folder_path)
    # Process each file
    for fp in ALL:
        f_name = fp.split("\\")[-1]
        print(f"Processing: {f_name}")
        temp_wav = Wav(fp)
        temp_wav.output_info()

        temp_wav.generate_audio_array()

        regress_and_plot_time_domain(temp_wav.audio_array,
                                     temp_wav.frame_rate, A1)
