"""Ensemble plan manually split by type moode/theme."""
import json

from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
from sklearn.pipeline import Pipeline

from mediaeval2021 import common
from mediaeval2021.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2021.dataloaders.melspectrograms import labels_to_indices
from mediaeval2021.models.ensemble import Ensemble
from mediaeval2021.models.wrapper import TorchWrapper

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 0
            'film', 'heavy', 'holiday', 'drama', 'summer', 'upbeat', 'relaxing', 'groovy', 'fun', 'inspiring', 'space', 'game', 'motivational', 'dream'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 1
            'meditative', 'party', 'christmas', 'nature', 'energetic', 'retro', 'sad', 'emotional', 'commercial', 'movie', 'happy', 'background', 'trailer', 'advertising'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 2
            'calm', 'powerful', 'soft', 'sexy', 'action', 'children', 'fast', 'soundscape', 'hopeful', 'corporate', 'cool', 'dramatic', 'melodic', 'travel'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 3
            'love', 'deep', 'ballad', 'sport', 'dark', 'melancholic', 'positive', 'funny', 'romantic', 'epic', 'uplifting', 'documentary', 'slow', 'adventure'
        ],
    )
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=TorchWrapper(model_name="ResNet-18", dataloader=dataloader, batch_size=64, early_stopping=True),
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
