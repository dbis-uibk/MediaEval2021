"""This module contains the implementation of VGG-ish models."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve, roc_auc_score
import torch

from ..utils import find_elbow, EarlyStopper


class VGGishBaseline(BaseEstimator, ClassifierMixin):
    """The VGGish baseline wrapt as a sklearn classifier."""

    def __init__(
        self,
        epochs=100,
        dataloader=None,
        batch_size=64,
    ):
        """Creates the model."""
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.epochs = epochs
        self.dataloader = dataloader
        self._model = None
        self.threshold = None
        self.batch_size = batch_size

    def _init_model(self):
        model = CNN(num_class=len(self.label_split))

        self._model = model.to(device=self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)

        self.early_stopper = EarlyStopper(num_trials=10)

    def fit(self, features, target, epochs=None):
        """Fits the model for a given number of epochs."""
        if not self._model:
            self._init_model()

        t_features = torch.Tensor(self._reshape_data(features))
        t_target = torch.Tensor(target.astype(np.float32))

        train_dataset = torch.utils.data.TensorDataset(t_features, t_target)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        if self.dataloader:
            validation_data = self.dataloader.load_validate()
            valid_features, valid_targets = validation_data

        if not epochs:
            epochs = self.epochs

        for epoch in range(epochs):

            total_loss = 0
            self._model.train()

            for x, y in train_loader:

                # variables to cuda
                x = x.to(self.device)
                y = y.to(self.device)

                # predict
                out = self._model(x)
                loss = self.criterion(out, y)

                # back propagation
                self._model.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{epochs}] train loss: {total_loss / len(train_loader):.6f}')
            train_preds = self.predict_proba(features)
            train_auc = roc_auc_score(target, train_preds)
            print(f'\ttrain roc-auc: {train_auc:.6f}')

            # validation
            valid_preds = self.predict_proba(valid_features)
            valid_auc = roc_auc_score(valid_targets, valid_preds)
            print(f'\tvalid roc-auc: {valid_auc:.6f}')
            if not self.early_stopper.is_continuable(self._model, valid_auc, epoch, self.optimizer, loss):
                print(f'\tvalidation: best auc: {self.early_stopper.best_accuracy}')
                break
        
        # load best model
        checkpoint = torch.load(self.early_stopper.save_path)
        self._model.load_state_dict(checkpoint['model_state_dict'])

        output_shape = target.shape[1]
        if self.dataloader:
            try:
                validation_data = self.dataloader.load_validate()
            except (NotImplementedError, AttributeError):
                validation_data = None

            if validation_data and self.label_split is not None:
                data, labels = validation_data
                assert len(self.label_split) == output_shape
                self.validate(data, labels[..., self.label_split])
            elif validation_data and self.label_split is None:
                labels = validation_data[1]
                try:
                    assert labels.shape[1] == output_shape
                except IndexError:
                    assert output_shape == 1
                self.validate(*validation_data)
            else:
                self.threshold = np.full(output_shape, .5)
        else:
            self.threshold = np.full(output_shape, .5)

    def validate(self, features, target):
        """Validates the model."""
        features = self._reshape_data(features)
        y_pred = self._model.predict_proba(features)

        threshold = []
        for label_idx in range(y_pred.shape[1]):
            fpr, tpr, thresholds = roc_curve(target[..., label_idx],
                                             y_pred[..., label_idx])
            try:
                idx = find_elbow(tpr, fpr)
            except ValueError as ex:
                print(ex)
                idx = -1

            if idx >= 0:
                threshold.append(thresholds[idx])
            else:
                threshold.append(0.5)

        self.threshold = np.array(threshold)

    def predict(self, features):
        """Returns the classes predicted by the model."""
        predictions = self.predict_proba(features)
        if self.threshold is not None:
            return np.greater(predictions, self.threshold)
        else:
            return predictions > 0.5

    def predict_proba(self, features):
        """Returns the class probabilities predicted by the model."""
        features = torch.Tensor(self._reshape_data(features))
        dataloader = torch.utils.data.DataLoader(features, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self._model.eval()

        full_data = []
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                full_data.append(self._model(x).detach().cpu().numpy())

        return np.concatenate(full_data, axis=0)

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
