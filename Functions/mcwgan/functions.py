import sys

import numpy as np
import torch
import signal
import csv
import os
import time
def implements_iterator(cls):
    '''
    From jinja2/_compat.py. License: BSD.

    Use as a decorator like this::

        @implements_iterator
        class UppercasingIterator(object):
            def __init__(self, iterable):
                self._iter = iter(iterable)
            def __iter__(self):
                return self
            def __next__(self):
                return next(self._iter).upper()

    '''
    return cls

get_next = lambda x: x.__next__

def to_cuda_if_available(*tensors):
    if torch.cuda.is_available():
        tensors = [tensor.cuda() if tensor is not None else None for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors

def to_cpu_if_available(*tensors):
    if torch.cuda.is_available():
        tensors = [tensor.cpu() if tensor is not None else None for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors

class DelayedKeyboardInterrupt(object):

    SIGNALS = [signal.SIGINT, signal.SIGTERM]

    def __init__(self):
        self.signal_received = {}
        self.old_handler = {}

    def __enter__(self):
        self.signal_received = {}
        self.old_handler = {}
        for sig in self.SIGNALS:
            self.old_handler[sig] = signal.signal(sig, self.handler)

    def handler(self, sig, frame):
        self.signal_received[sig] = frame
        print('Delaying received signal', sig)

    def __exit__(self, type, value, traceback):
        for sig in self.SIGNALS:
            signal.signal(sig, self.old_handler[sig])
        for sig, frame in self.signal_received.items():
            old_handler = self.old_handler[sig]
            print('Resuming received signal', sig)
            if callable(old_handler):
                old_handler(sig, frame)
            elif old_handler == signal.SIG_DFL:
                sys.exit(0)
        self.signal_received = {}
        self.old_handler = {}



class Logger(object):

    PRINT_FORMAT = "epoch {:d}/{:d} {}-{}: {:.05f} Time: {:.2f} s"
    CSV_COLUMNS = ["epoch", "model", "metric_name", "metric_value", "time"]

    start_time = None

    def __init__(self, output_path, append=False):
        if append and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            self.output_file = open(output_path, "a")
            self.output_writer = csv.DictWriter(self.output_file, fieldnames=self.CSV_COLUMNS)
        else:
            self.output_file = open(output_path, "w")
            self.output_writer = csv.DictWriter(self.output_file, fieldnames=self.CSV_COLUMNS)
            self.output_writer.writeheader()

        self.start_timer()

    def start_timer(self):
        self.start_time = time.time()

    def log(self, epoch_index, num_epochs, model_name, metric_name, metric_value):
        elapsed_time = time.time() - self.start_time

        self.output_writer.writerow({
            "epoch": epoch_index + 1,
            "model": model_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "time": elapsed_time
        })

        print(self.PRINT_FORMAT
              .format(epoch_index + 1,
                      num_epochs,
                      model_name,
                      metric_name,
                      metric_value,
                      elapsed_time
                      ))

    def flush(self):
        self.output_file.flush()

    def close(self):
        self.output_file.close()

        self.output_file = None
        self.output_writer = None

        import numpy as np

class Dataset(object):

    def __init__(self, features):
        self.features = features

    def split(self, proportion):
        assert 0 < proportion < 1, "Proportion should be between 0 and 1."

        limit = int(np.floor(len(self.features) * proportion))

        return Dataset(self.features[:limit, :]), Dataset(self.features[limit:, :])

    def batch_iterator(self, batch_size, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(self.features))
        else:
            indices = np.arange(len(self.features))
        return DatasetIterator(self.features, indices, batch_size)

@implements_iterator
class DatasetIterator(object):

    def __init__(self, features, indices, batch_size):
        self.features = features
        self.indices = indices
        self.batch_size = batch_size

        self.batch_index = 0
        self.num_batches = int(np.ceil(len(features) / batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_index >= self.num_batches:
            raise StopIteration
        else:
            batch_start = self.batch_index * self.batch_size
            batch_end = (self.batch_index + 1) * self.batch_size
            self.batch_index += 1
            return self.features[self.indices[batch_start:batch_end]]

