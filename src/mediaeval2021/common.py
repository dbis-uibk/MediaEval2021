"""Module containing common functions."""
from os.path import basename
from os.path import splitext

from dbispipeline import store
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def grid_params():
    """Gridsearch parameters that can be used in all plans."""
    return {
        'scoring': {
            'f1_micro':
                make_scorer(f1_score, average='micro'),
            'f1_macro':
                make_scorer(f1_score, average='macro'),
            'roc_auc':
                make_scorer(roc_auc_score, average='macro', needs_proba=True),
            'average_precision':
                make_scorer(average_precision_score,
                            average='macro',
                            needs_proba=True),
            'precision_micro':
                make_scorer(precision_score, average='micro'),
            'precision_macro':
                make_scorer(precision_score, average='macro'),
            'recall_micro':
                make_scorer(recall_score, average='micro'),
            'recall_macro':
                make_scorer(recall_score, average='macro'),
        },
        'verbose': 100,
        'n_jobs': -1,
        'iid': True,
        'refit': False,
    }


def fixed_split_params():
    """Fixed split parameters that can be used in all plans."""
    return {
        'scoring': {
            'f1_micro':
                make_scorer(f1_score, average='micro'),
            'f1_macro':
                make_scorer(f1_score, average='macro'),
            'roc_auc':
                make_scorer(roc_auc_score, average='macro', needs_proba=True),
            'average_precision':
                make_scorer(average_precision_score,
                            average='macro',
                            needs_proba=True),
            'precision_micro':
                make_scorer(precision_score, average='micro'),
            'precision_macro':
                make_scorer(precision_score, average='macro'),
            'recall_micro':
                make_scorer(recall_score, average='micro'),
            'recall_macro':
                make_scorer(recall_score, average='macro'),
            'average_precision_all':
                make_scorer(average_precision_score,
                            average=None,
                            needs_proba=True),
            'roc_auc_all':
                make_scorer(roc_auc_score, average=None, needs_proba=True),
            'confusion_matrix':
                make_scorer(multilabel_confusion_matrix),
        },
    }


def store_prediction(model, dataloader, file_name_prefix=None):
    """Function extracting the predictions."""
    if not file_name_prefix:
        file_name_prefix = type(model).__name__
    elif file_name_prefix[-1] != '_':
        file_name_prefix += '_'

    if store['plan_path']:
        file_name_prefix += splitext(basename(store['plan_path']))[0]

    result_folder = 'results/'

    x_test, _ = dataloader.load_test()
    y_pred = model.predict(x_test)
    np.save(
        result_folder + file_name_prefix + '_decisions.npy',
        y_pred.astype(bool),
    )
    y_pred = model.predict_proba(x_test)
    np.save(
        result_folder + file_name_prefix + '_predictions.npy',
        y_pred.astype(np.float64),
    )
