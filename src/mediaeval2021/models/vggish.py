"""This module contains the implementation of VGG-ish models."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from skorch import NeuralNetClassifier
import torch


class VGGishBaseline(BaseEstimator, ClassifierMixin):
    """The VGGish baseline wrapt as a sklearn classifier."""

    def __init__(
        self,
        epochs=10,
    ):
        """Creates the model."""
        if torch.cuda.device_count() > 0:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._model = NeuralNetClassifier(
            CNN(num_class=len(self.label_split)),
            max_epochs=epochs,
            criterion=torch.nn.BCELoss,
            optimizer=torch.optim.Adam,
            lr=1e-4,
            iterator_train__shuffle=True,
            train_split=False,
            device=device,
        )
        self.epochs = epochs

    def fit(self, features, target, epochs=None):
        """Fits the model for a given number of epochs."""
        features = self._reshape_data(features)
        if epochs:
            self._model.fit_loop(features, target, epochs=epochs)
        else:
            self._model.fit(features, target)

    def validate(self, features, target):
        """Validates the model."""

    def predict(self, features):
        """Returns the classes predicted by the model."""
        features = self._reshape_data(features)
        pred = self._model.predict(features)
        pred = pred.reshape(len(pred), 1)

        return np.apply_along_axis(
            lambda v: self._label_to_vector(v[0]),
            1,
            pred,
        )

    def predict_proba(self, features):
        """Returns the class probabilities predicted by the model."""
        features = self._reshape_data(features)
        return self._model.predict_proba(features)

    def _reshape_data(self, data):
        return data.reshape(data.shape[:-1])

    def _label_to_vector(self, label):
        return [1 if label == elem else 0 for elem in range(self.num_classes)]


class CNN(torch.nn.Module):
    """The VGGish baseline model used since the MediaEval 2019.

    The implementation is taken from the provided github repository.
    https://github.com/MTG/mtg-jamendo-dataset/blob/master/scripts/baseline/model.py
    """

    def __init__(self, num_class=15):
        """Creates the pytorch model."""
        super(CNN, self).__init__()

        # init bn
        self.bn_init = torch.nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = torch.nn.BatchNorm2d(64)
        self.mp_1 = torch.nn.MaxPool2d((2, 4))

        # layer 2
        self.conv_2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = torch.nn.BatchNorm2d(128)
        self.mp_2 = torch.nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = torch.nn.BatchNorm2d(128)
        self.mp_3 = torch.nn.MaxPool2d((2, 4))

        # layer 4
        self.conv_4 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = torch.nn.BatchNorm2d(128)
        self.mp_4 = torch.nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = torch.nn.BatchNorm2d(64)
        self.mp_5 = torch.nn.MaxPool2d((4, 4))

        # classifier
        self.dense = torch.nn.Linear(64, num_class)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        """Computation performed by the model."""
        x = x.unsqueeze(1)

        # init bn
        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(torch.nn.ELU()(self.bn_1(self.conv_1(x))))

        # layer 2
        x = self.mp_2(torch.nn.ELU()(self.bn_2(self.conv_2(x))))

        # layer 3
        x = self.mp_3(torch.nn.ELU()(self.bn_3(self.conv_3(x))))

        # layer 4
        x = self.mp_4(torch.nn.ELU()(self.bn_4(self.conv_4(x))))

        # layer 5
        x = self.mp_5(torch.nn.ELU()(self.bn_5(self.conv_5(x))))

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = torch.nn.Sigmoid()(self.dense(x))

        return logit
