from os import path
import pickle

from dbispipeline.base import TrainValidateTestLoader
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from .. import utils


class MelSpectrogramsLoader(TrainValidateTestLoader):
    """
    Loads the mel-spectrograms provided by the task organizers and the labels
    for both the training and test set.
    """

    def __init__(self,
                 data_path,
                 training_path,
                 test_path,
                 validate_path=None,
                 window='center',
                 window_size=1366,
                 num_windows=1,
                 flatten=False,
                 repeating=False):
        self.training_path = training_path
        self.test_path = test_path
        self.validate_path = validate_path
        self.data_path = data_path
        self.window = window
        self.window_size = window_size
        self.num_windows = num_windows
        self.mlb = MultiLabelBinarizer()
        self.mlb_fit = True
        self.flatten = flatten
        self.repeating = repeating

    def _load_set(self, set_path, sub_sampling):
        sample_set = utils.load_set_info(set_path)[['PATH', 'TAGS']]
        X, y = self._load_data(sample_set, sub_sampling)

        # TODO: Remove workaround
        if self.mlb_fit:
            y = self.mlb.fit_transform(y)
            self.mlb_fit = False
        else:
            y = self.mlb.transform(y)

        if self.flatten:
            return X.flatten(), y
        return X, y

    def _load_data(self, sample_set, sub_sampling):
        X = []
        y = []

        for _, sample in sample_set.iterrows():
            sample_path = sample['PATH'].replace('.mp3', '.npy')
            sample_data = np.load(path.join(self.data_path, sample_path))

            if sub_sampling:
                sample_data = utils.get_windows(sample=sample_data,
                                                window=self.window,
                                                window_size=self.window_size,
                                                num_windows=self.num_windows,
                                                repeating=self.repeating)
                X.extend(sample_data)
                for _ in range(self.num_windows):
                    y.append(sample['TAGS'])
            else:
                sample_data = utils.get_windows(sample=sample_data,
                                                window=self.window,
                                                window_size=self.window_size,
                                                num_windows=1,
                                                repeating=self.repeating)
                X.append(sample_data[0])
                y.append(sample['TAGS'])

        assert len(X) == len(y)

        X = np.array(X)
        y = np.array(y)
        X = X.reshape(*X.shape, 1)
        return X, y

    def load_train(self):
        """Returns the train data."""
        return self._load_set(self.training_path, sub_sampling=True)

    def load_validate(self):
        """Returns the validate data."""
        if self.validate_path:
            return self._load_set(self.validate_path, sub_sampling=True)
        else:
            raise NotImplementedError()

    def load_test(self):
        """Returns the test data."""
        return self._load_set(self.test_path, sub_sampling=False)

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration of this loader.

        This is for storing its state in the database.
        """
        if isinstance(self.mlb.classes_, np.ndarray):
            classes = self.mlb.classes_.tolist()
        else:
            classes = self.mlb.classes_
        return {
            'training_path': self.training_path,
            'validate_path': self.validate_path,
            'test_path': self.test_path,
            'data_path': self.data_path,
            'window': self.window,
            'window_size': self.window_size,
            'num_windows': self.num_windows,
            'classes': classes,
        }


class MelSpectPickleLoader(TrainValidateTestLoader):
    """Loads the data from a pickle file."""

    def __init__(self, pickle_file):
        self.data = pickle.load(open(pickle_file, 'rb'))

    def load_train(self):
        """Returns the train data."""
        return self.data['train']

    def load_validate(self):
        """Returns the validate data."""
        return self.data['validate']

    def load_test(self):
        """Returns the test data."""
        return self.data['test']

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration of this loader.

        This is for storing its state in the database.
        """
        return self.data['configuration']


def labels_to_indices(dataloader, label_list):
    """Converts the label_list to an index array."""
    classes = dataloader.configuration['classes']
    class_prefix = 'mood/theme---'

    indices = []
    for label in label_list:
        if not label.startswith(class_prefix):
            label = class_prefix + label

        indices.append(classes.index(label))

    return np.array(indices)
