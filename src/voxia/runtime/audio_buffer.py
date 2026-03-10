import numpy as np


class AudioBuffer:

    def __init__(self):

        self.buffer = []

    def add(self, chunk):

        self.buffer.append(chunk)

    def get(self):

        if not self.buffer:
            return None

        return np.concatenate(self.buffer)