"""Ensemble plan manually split by type moode/theme."""
import json

from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import numpy as np
from sklearn.pipeline import Pipeline

from mediaeval2021 import common
from mediaeval2021.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2021.models.ensemble import Ensemble
from mediaeval2021.models.wrapper import TorchWrapper

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = [
    np.arange(0, 14, 1),
    np.arange(14, 28, 1),
    np.arange(28, 42, 1),
    np.arange(42, 56, 1),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=TorchWrapper(
             model_name='ResNet-18',
             dataloader=dataloader,
             batch_size=64,
             early_stopping=True,
         ),
         label_splits=label_splits,
         epochs=100,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    lambda results: print(json.dumps(results, indent=4)),
]
