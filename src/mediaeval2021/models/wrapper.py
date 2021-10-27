"""This module contains the implementation of VGG-ish models."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from mediaeval2021.models.resnet import ResNet
from mediaeval2021.models.vggish import CNN
from mediaeval2021.models.custom_vggish import CustomCNN

from ..utils import find_elbow, EarlyStopper


class TorchWrapper(BaseEstimator, ClassifierMixin):
    """Wrapping torch based models."""

    def __init__(
        self,
        model_name,
        epochs=100,
        dataloader=None,
        batch_size=64,
        early_stopping=False
    ):

        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model_name = model_name
        self._model = None

        self.epochs = epochs
        self.dataloader = dataloader
        self.threshold = None
        self.batch_size = batch_size
        self.early_stopping = early_stopping

    def _init_model(self):
        if self.model_name == 'ResNet-18':
            model = ResNet(num_layers=18, num_classes=len(self.label_split))
        elif self.model_name == 'ResNet-34':
            model = ResNet(num_layers=34, num_classes=len(self.label_split))
        elif self.model_name == 'CNN':
            model = CNN(num_classes=len(self.label_split))
        elif self.model_name == 'CustomCNN':
            model = CustomCNN(num_classes=len(self.label_split))
        else:
            raise ValueError('unknown model name: ' + self.model_name)

        self._model = model.to(device=self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.1)
        self.early_stopper = EarlyStopper(num_trials=30)

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

            # validation
            valid_preds = self.predict_proba(valid_features)
            valid_auc = roc_auc_score(valid_targets, valid_preds)
            valid_loss = self.criterion(torch.Tensor(valid_preds), torch.Tensor(valid_targets))

            # LR scheduling
            self.scheduler.step(valid_loss)

            print(f'Epoch [{epoch+1}/{epochs}] train loss: {total_loss / len(train_loader):.6f} valid roc-auc: {valid_auc:.6f}')

            if self.early_stopping and not self.early_stopper.is_continuable(self._model, valid_auc, epoch, self.optimizer, loss):
                print(f'\tvalidation: best auc: {self.early_stopper.best_accuracy}')
                break
        
        # load best model
        if self.early_stopping:
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
        y_pred = self.predict_proba(features)

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
