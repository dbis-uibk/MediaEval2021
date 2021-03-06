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
        label_list=[  # theme
            'action',
            'adventure',
            'advertising',
            'background',
            'ballad',
            'children',
            'christmas',
            'commercial',
            'corporate',
            'documentary',
            'drama',
            'dream',
            'film',
            'game',
            'holiday',
            'love',
            'movie',
            'nature',
            'party',
            'retro',
            'soundscape',
            'space',
            'sport',
            'summer',
            'trailer',
            'travel',
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
            'melodic',
            'motivational',
            'positive',
            'powerful',
            'relaxing',
            'romantic',
            'sad',
            'sexy',
            'slow',
            'soft',
            'upbeat',
            'uplifting',
        ],
    )
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=TorchWrapper(model_name="CNN", dataloader=dataloader, batch_size=64),
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
