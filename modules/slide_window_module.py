from . import *
import random

class SlideWindowModule:

    def __init__(self, config):
        self.size = config['sliding_window']
        self.curr = 0

    def __call__(self, dataset, retr_result):
        tmp = []
        if self.curr >= len(max(retr_result, key=len)):
            return []

        for doc_list in retr_result:
            window = doc_list[self.curr:self.curr + self.size]

            # Fill with Noise
            # missing = self.size - len(window)

            # if missing > 0:
            #     filler = random.choices(dataset, k=missing)
            #     window.extend(filler)

            if window:
                tmp.append(window)

        self.curr += self.size
        return tmp