import math
from random import randrange
import os

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

cache = {'model': None, 'X': None, 'y': None}


def cached_model_predict(model, X):
    if model != cache['model'] or not np.array_equal(cache['X'], X):
        cache['model'] = model
        cache['X'] = X
        cache['y'] = model.predict(X)
    return cache['y']


def cached_model_predict_clear():
    cache['model'] = None
    cache['X'] = None
    cache['y'] = None


def find_elbow(x_values, y_values):
    origin = (x_values[0], y_values[0])
    baseline_vec = np.subtract((x_values[-1], y_values[-1]), origin)
    baseline_vec = (-baseline_vec[1], baseline_vec[0])  # rotate 90 degree
    baseline_vec = normalize(np.array(baseline_vec).reshape(1, -1))[0]

    idx = -1
    max_distance = 0
    for i, point in enumerate(zip(x_values, y_values)):
        point_vec = np.subtract(point, origin)
        distance = abs(np.dot(point_vec, baseline_vec))
        max_distance = max(max_distance, distance)

        if max_distance == distance:
            idx = i

    return idx


def load_set_info(path):
    SEP = '\t'
    data = []
    with open(path, 'r') as lines:
        headers = None

        for line in lines:
            fields = line.strip('\n').split(SEP)
            if headers is None:
                headers = fields
            else:
                tag_idx = len(headers) - 1
                current = fields[:tag_idx]
                current.append(fields[tag_idx:])
                data.append(current)

    return pd.DataFrame(data, columns=headers)


def get_windows(sample, window, window_size, num_windows, repeating=False):
    if repeating:
        sample = _repeat_sample(sample=sample, min_size=window_size)

    assert sample.shape[1] >= window_size

    windows = []
    for i in range(num_windows):
        if window == 'center':
            start_idx = int((sample.shape[1] - window_size) / 2)
        elif window == 'random':
            start_idx = randrange(sample.shape[1] - window_size)
        elif window == 'sliding':
            step = int((sample.shape[1] - window_size) / num_windows)
            start_idx = step * i
        else:
            raise ValueError('Unknown window type.')

        end_idx = start_idx + window_size

        windows.append(sample[:, start_idx:end_idx])

    return windows


def _repeat_sample(sample, min_size):
    if sample.shape[1] < min_size:
        count = math.ceil(min_size / sample.shape[1])
        return np.tile(sample, count)
    else:
        return sample


class EarlyStopper(object):

    def __init__(self, num_trials, save_path='./tmp.pt', accuracy=0):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = accuracy
        self.save_path = save_path

    def is_continuable(self, model, accuracy, epoch, optimizer, loss):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy
            }, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False