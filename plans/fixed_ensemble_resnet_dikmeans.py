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
            'hopeful', 'heavy', 'holiday', 'nature', 'summer', 'fast', 'emotional', 'corporate', 'space', 'dramatic', 'melodic', 'adventure'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 1
            'motivational', 'powerful', 'meditative', 'christmas', 'energetic', 'action', 'romantic', 'happy', 'epic', 'cool', 'advertising'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 2
            'calm', 'party', 'soft', 'sport', 'dark', 'relaxing', 'fun', 'movie', 'background', 'game', 'slow'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 3
            'ballad', 'sexy', 'drama', 'children', 'retro', 'sad', 'commercial', 'funny', 'documentary', 'love', 'travel'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 4
            'film', 'upbeat', 'deep', 'melancholic', 'positive', 'groovy', 'soundscape', 'inspiring', 'uplifting', 'trailer', 'dream'
        ],
    )
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=TorchWrapper(model_name="ResNet-18", dataloader=dataloader, batch_size=64),
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
