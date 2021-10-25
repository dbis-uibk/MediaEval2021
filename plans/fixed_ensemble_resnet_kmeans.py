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
            'background',
            'ballad',
            'children',
            'christmas',
            'drama',
            'emotional',
            'holiday',
            'hopeful',
            'inspiring',
            'love',
            'meditative',
            'melancholic',
            'nature',
            'relaxing',
            'romantic',
            'sad',
            'soft'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 1
            'advertising',
            'commercial',
            'corporate',
            'fun',
            'game',
            'happy',
            'melodic',
            'motivational',
            'positive',
            'upbeat',
            'uplifting'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 2
            'action',
            'adventure',
            'calm',
            'cool',
            'dark',
            'documentary',
            'dramatic',
            'dream',
            'epic',
            'film',
            'movie',
            'slow',
            'soundscape',
            'space',
            'trailer',
            'travel'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # cluster 3
            'deep',
            'energetic',
            'fast',
            'funny',
            'groovy',
            'heavy',
            'party',
            'powerful',
            'retro',
            'sexy',
            'sport',
            'summer'
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
