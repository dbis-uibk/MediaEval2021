import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve, roc_auc_score

from ..utils import find_elbow, EarlyStopper


class ResNetModel(BaseEstimator, ClassifierMixin):
    """ResNet wrapt as a sklearn classifier."""

    def __init__(
        self,
        epochs=10,
        dataloader=None,
        batch_size=32,
        num_layers=18
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
        self.num_layers = num_layers

    def _init_model(self):
        model = ResNet(num_layers=self.num_layers, block=Block, image_channels=1, num_classes=len(self.label_split))

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


class Block(nn.Module):
    """
    Building block for ResNet.

    Implementation taken from: https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded
    """

    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet architecture in pyTorch.

    Implementation taken from: https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded
    """

    def __init__(self, num_layers=18, block=Block, image_channels=1, num_classes=56):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = torch.nn.Sigmoid()(x)

        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)