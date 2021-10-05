"""Ensemble plan manually split by type moode/theme."""
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
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
    ),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=VGGishBaseline(
             epochs=20,
             classes=dataloader.configuration['classes'],
         ),
         label_splits=label_splits,
         epochs=20,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
