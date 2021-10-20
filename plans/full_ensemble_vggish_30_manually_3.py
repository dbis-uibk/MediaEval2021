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

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_TODO.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # unsure
            'action',
            'adventure',
            'background',
            'drama',
            'dream',
            'love',
            'melodic',
            'motivational',
            'nature',
            'party',
            'retro',
            'soundscape',
            'space',
            'sport',
            'summer',
            'upbeat',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # theme
            'advertising',
            'ballad',
            'children',
            'christmas',
            'commercial',
            'corporate',
            'documentary',
            'film',
            'game',
            'holiday',
            'movie',
            'trailer',
            'travel',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # mood
            'calm',
            'cool',
            'dark',
            'deep',
            'dramatic',
            'emotional',
            'energetic',
            'epic',
            'fast',
            'fun',
            'funny',
            'groovy',
            'happy',
            'heavy',
            'hopeful',
            'inspiring',
            'meditative',
            'melancholic',
            'positive',
            'powerful',
            'relaxing',
            'romantic',
            'sad',
            'sexy',
            'slow',
            'soft',
            'uplifting',
        ],
    ),
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
