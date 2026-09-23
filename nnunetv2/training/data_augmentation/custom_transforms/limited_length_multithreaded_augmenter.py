import random

import batchgenerators.dataloading.nondet_multi_threaded_augmenter as nondet_augmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter


_batchgenerators_producer = nondet_augmenter.producer


def _fully_seeded_producer(queue, data_loader, transform, thread_id, seed,
                           abort_event, pause_event=None, wait_time=0.02):
    # batchgenerators seeds NumPy and Torch but not Python's random module. Some
    # augmentations use random.uniform, so paired runs still diverge without this.
    if seed is not None:
        random.seed(seed)
    return _batchgenerators_producer(
        queue, data_loader, transform, thread_id, seed,
        abort_event, pause_event, wait_time
    )


class LimitedLenWrapper(NonDetMultiThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len

    def _start(self):
        # The parent resolves its worker target from its own module global. Swap
        # that target only while processes are created, then restore it.
        original_producer = nondet_augmenter.producer
        nondet_augmenter.producer = _fully_seeded_producer
        try:
            super()._start()
        finally:
            nondet_augmenter.producer = original_producer
