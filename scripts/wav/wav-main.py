from wav.wav import Wav

# Provide absolute filepaths here of wav files to be analysed.
# TODO : add option to provide folder paths
ALL: list[str] = []

if __name__ == "__main__":
    for fp in ALL:
        temp_wav = Wav(fp)
        temp_wav.output_info()
        temp_wav.generate_audio_array()
