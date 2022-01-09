import librosa
from config import CONFIG
from torch.nn.utils.rnn import pad_sequence


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


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens