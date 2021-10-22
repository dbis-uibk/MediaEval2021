"""Ensemble plan manually split by type moode/theme."""
import json

from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
from sklearn.pipeline import Pipeline

from mediaeval2021 import common
from mediaeval2021.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2021.dataloaders.melspectrograms import labels_to_indices
from mediaeval2021.models.ensemble import Ensemble
from mediaeval2021.models.vggish import VGGishBaseline

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 0
            'powerful', 'holiday', 'deep', 'nature', 'fast', 'groovy', 'romantic', 'soundscape', 'happy', 'game', 'slow', 'dream'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 1
            'calm', 'heavy', 'ballad', 'children', 'summer', 'sad', 'commercial', 'movie', 'epic', 'love', 'advertising'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 2
            'upbeat', 'meditative', 'sexy', 'energetic', 'melancholic', 'emotional', 'funny', 'corporate', 'documentary', 'dramatic', 'melodic'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 3
            'soft', 'party', 'drama', 'positive', 'retro', 'film', 'hopeful', 'uplifting', 'trailer', 'background', 'travel'
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
         base_estimator=VGGishBaseline(dataloader=dataloader),
         label_splits=label_splits,
         epochs=25,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    lambda results: print(json.dumps(results, indent=4)),
]
