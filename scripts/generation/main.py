from generation.sound_generation import SoundGeneration

OCTAVES = [64, 128, 256, 512, 1024, 2048]

WHOLETONE_BASS = [
    65.40639,
    73.41619,
    82.40689,
    92.49861,
    103.8262,
    116.5409
]
# TODO 2*pi/f to create whole waves, try integers (to counter the pop at the
#  end), choose f to be a good multiple of 2*pi
# TODO try to look at multiple different libraries
# TODO take a frequency around 1 s, vary the length very slightly to see if
#  the pop goes away. NB Need to get better than 0.1 s on reverb time
ALL = []
for octave in range(5):
    ALL += [(2 ** octave) * f for f in WHOLETONE_BASS]
    # get 5 octaves of the whole tone scale, then add top C
ALL.append(2093.005)
assert len(ALL) == 31


if __name__ == '__main__':
    octave_playthrough = SoundGeneration(OCTAVES, 3, 3)
    octave_playthrough.get_predicted_time()
    octave_playthrough.play()

    full_playthrough = SoundGeneration(ALL, 3, 3)
    full_playthrough.get_predicted_time()
    full_playthrough.play()
