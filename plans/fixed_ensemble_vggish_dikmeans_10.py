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
        label_list=[
            'deep', 'holiday', 'sexy', 'sad', 'inspiring', 'game'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
            'documentary', 'heavy', 'ballad', 'romantic', 'funny', 'adventure'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[ 
            'positive', 'meditative', 'summer', 'fast', 'calm', 'travel'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
            'trailer', 'soft', 'energetic', 'soundscape', 'background', 'melodic'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'commercial', 'drama', 'retro', 'fun', 'epic', 'dream'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[ 
          'emotional', 'party', 'nature', 'film', 'uplifting', 'advertising'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'hopeful', 'powerful', 'relaxing', 'corporate', 'space'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'motivational', 'dark', 'children', 'groovy', 'dramatic'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'action', 'christmas', 'upbeat', 'movie', 'love'
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  
           'slow', 'sport', 'melancholic', 'happy', 'cool'
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
