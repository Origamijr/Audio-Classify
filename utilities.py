import librosa
from config import CONFIG


def show_spectrogram(s):
    librosa.display.specshow(s,sr=CONFIG['preprocessing']['sr'])


class MovingAverage:
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.value = 0
        self.n = 0

    def add(self, x):
        self.n += 1
        if self.alpha == 0:
            self.value += (x - self.value) / self.n
        else:
            self.value *= 1 - self.alpha
            self.value += self.alpha * x
        return self.value
