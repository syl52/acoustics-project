import time

import numpy as np
import winsound
from pysinewave import SineWave

if __name__ == "__main__":
    # flags for running main
    winsound_flag = False
    pysinewave_flag = True
    if winsound_flag:
        # winsound
        f = 250  # in Hz, so period in ms is 4
        times = [i for i in range(3000, 3005, 1)]
        for t in times:
            print(f"Now playing frequency {f} Hz for {t} ms")
            winsound.Beep(f, t)
            time.sleep(3)
    """Results ( on my bluetooth speaker)
    time | click?  
    3000 | medium - pop - pop
    3001 | pop - pop 
    3002 | medium - low - pop
    3003 | medium - low - pop
    3004 | pop - pop - pop
    Conclusions: stochastic, not same every time
    pop content varies (low freq pop, white noise pop)
    """
    if pysinewave_flag:
        f = 250
        times = np.linspace(1998, 2000, 20)  # in ms, remember the period is
        # 4 ms so T/2 is 2 ms
        sinewave = SineWave(pitch_per_second=10000)  # having a high pitch per
        # second basically ensures it goes straight to the desired frequency
        sinewave.set_frequency(f)
        for t in times:
            print(f"Now playing frequency {f} Hz for {t / 1000} s")
            sinewave.play()
            time.sleep(t / 1000)
            sinewave.stop()

            time.sleep(1.5)
    """Results on my bluetooth speaker
    Some were duller than others, still couldn't fully get rid of the pop
    Popping is still stochastic."""
