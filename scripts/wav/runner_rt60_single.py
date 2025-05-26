from wav.wav import Wav
from wav.wav_graphers import regress_and_plot_time_domain
from wav.wav_helpers import get_wav_files_from_folder, get_config



if __name__ == "__main__":

    folder_path = (r"C:\Users\Shawn\PycharmProjects\acoustics-project\audio"
                   r"-files-single")

    # Get all wav files from the folder
    ALL: list[str] = get_wav_files_from_folder(folder_path)
    # Process each file
    for fp in ALL:
        f_name = fp.split("\\")[-1]
        f_name = f_name.split(".")[0]
        print(f"Processing: {f_name}")
        temp_wav = Wav(fp)
        temp_wav.output_info()

        temp_wav.generate_audio_array()

        regress_and_plot_time_domain(temp_wav.audio_array,
                                     temp_wav.frame_rate,
                                     get_config(f_name),
                                     f_name
                                     )
