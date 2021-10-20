"""Ensemble plan manually split by type moode/theme."""
import json

from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
from sklearn.pipeline import Pipeline

import numpy as np

from mediaeval2021 import common
from mediaeval2021.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2021.dataloaders.melspectrograms import labels_to_indices
from mediaeval2021.models.ensemble import Ensemble
from mediaeval2021.models.vggish import VGGishBaseline

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_TODO.pickle')

label_splits = [
    np.arange(0, 19, 1),
    np.arange(19, 38, 1),
    np.arange(38, 56, 1),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=VGGishBaseline(dataloader=dataloader),
         label_splits=label_splits,
         epochs=30,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    lambda results: print(json.dumps(results, indent=4)),
]
