import time

from pysinewave import SineWave

OCTAVES = [64, 128, 256, 512, 1024, 2048]

WHOLETONE_BASS = [
    65.40639,
    73.41619,
    82.40689,
    92.49861,
    103.8262,
    116.5409
]

ALL = []
for octave in range(5):
    ALL += [(2 ** octave) * f for f in WHOLETONE_BASS]
    # get 5 octaves of the whole tone scale, then add top C
ALL.append(2093.005)
assert len(ALL) == 31


if __name__ == '__main__':
    sinewave = SineWave(pitch_per_second=10000)
    sinewave.set_frequency(400)
    sinewave.play()
    time.sleep(3.000625)
    sinewave.stop()
"""
    octave_playthrough = SoundGeneration(OCTAVES, 3, 3)
    octave_playthrough.get_predicted_time()
    octave_playthrough.play()

    full_playthrough = SoundGeneration(ALL, 3, 3)
    full_playthrough.get_predicted_time()
    full_playthrough.play()
"""
