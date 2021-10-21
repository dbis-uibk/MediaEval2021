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
        label_list=[
            'children', 'heavy', 'nature', 'corporate', 'cool', 'adventure'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
            'meditative', 'party', 'positive', 'emotional', 'trailer', 'motivational'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[ 
            'drama', 'sexy', 'energetic', 'romantic', 'calm', 'melodic'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
            'sad', 'powerful', 'hopeful', 'film', 'uplifting', 'travel'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'groovy', 'holiday', 'dark', 'fun', 'background', 'dream'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[ 
          'funny', 'soft', 'retro', 'action', 'happy', 'advertising'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'soundscape', 'upbeat', 'melancholic', 'movie', 'game'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'space', 'christmas', 'summer', 'epic', 'love'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'documentary', 'sport', 'relaxing', 'commercial', 'dramatic'
        ],
    )
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=VGGishBaseline(dataloader=dataloader),
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
