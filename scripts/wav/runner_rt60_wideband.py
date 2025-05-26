from wav.wav import Wav
from wav.wav_graphers import calculate_bandwise_rt60s_with_regions
from wav.wav_helpers import get_wav_files_from_folder

if __name__ == "__main__":
    folder_path = (r"C:\Users\Shawn\PycharmProjects\acoustics-project\audio"
                   r"-files-wideband")
    all_files = get_wav_files_from_folder(folder_path)

    for fp in all_files:
        f_name = fp.split("\\")[-1].split(".")[0]
        print(f"Processing: {f_name}")

        wav_obj = Wav(fp)
        wav_obj.output_info()
        wav_obj.generate_audio_array()

        # if stereo, choose first channel:
        audio = wav_obj.audio_array
        if audio.ndim > 1:
            audio = audio[:, 0]

        rt60s = calculate_bandwise_rt60s_with_regions(audio,
                                                      wav_obj.frame_rate,
                                                      f_name)
        print(f"Avg RT60 per frequency for {f_name}:")
        for freq, rt in rt60s.items():
            print(f"  {freq} Hz : {rt if rt else 'N/A'} s")
